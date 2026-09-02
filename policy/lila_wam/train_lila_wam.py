#!/usr/bin/env python
"""Train LiLa-WAM on a RoboSynChallenge LeRobot v2.1 dataset.

    python policy/lila_wam/train_lila_wam.py --config policy/lila_wam/configs/robosyn_3cam.yaml

Same schedule, optimizer and checkpoint format as upstream ``train.py`` (so
``--resume`` / ``--init_from`` behave identically and checkpoints stay
interchangeable); the only substitutions are the LeRobot v2.1 dataset and the
multi-camera wrapper. The run directory also gets a copy of the config and the
normalization stats, so evaluation only needs to be pointed at the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from _bootstrap import add_repo_root, resolve_path

add_repo_root()

from omegaconf import OmegaConf  # noqa: E402

from policy.lila_wam._upstream import upstream_collate_fn, upstream_models  # noqa: E402
from policy.lila_wam.lila_dataset import create_dataset, ensure_frame_cache  # noqa: E402
from policy.lila_wam.lila_model import build_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def build_train_config(config) -> dict:
    training = config.training
    future_cfg = config.model.get("future_feat", {}) or {}
    return {
        "time_mu": training.time_mu,
        "time_sigma": training.time_sigma,
        "use_vel_weight": training.use_vel_weight,
        "vel_weight_alpha": training.vel_weight_alpha,
        "vel_weight_sigma": training.vel_weight_sigma,
        "use_future_feat": bool(future_cfg.get("enabled", False)),
        "lambda_future_feat": training.get("lambda_future_feat", 0.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="policy/lila_wam/configs/robosyn_3cam.yaml")
    parser.add_argument(
        "--norm_stats_path",
        default=None,
        help="normalization stats json; defaults to <save_dir>/../norm_stats.json next to the config",
    )
    parser.add_argument("--save_dir", default="policy/lila_wam/checkpoints")
    parser.add_argument(
        "--resume",
        default=None,
        help="continue an interrupted run (model + optimizer + scheduler + epoch)",
    )
    parser.add_argument(
        "--init_from",
        default=None,
        help="start a new stage from these weights (fresh optimizer and lr schedule)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="override training.batch_size")
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="override training.learning_rate"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="override system.num_workers"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="stop after this many optimizer steps (smoke tests)",
    )
    parser.add_argument(
        "--run_name", default=None, help="run directory name (default: sft_<timestamp>)"
    )
    parser.add_argument(
        "--keep_last_n",
        type=int,
        default=0,
        help="keep only the N newest checkpoints (0 = keep all, upstream behaviour). "
             "A checkpoint carries the model plus AdamW's two moment buffers, so a long "
             "run with save_interval_epoch=1 can easily fill a disk.",
    )
    args = parser.parse_args()
    if args.resume and args.init_from:
        raise SystemExit(
            "--resume and --init_from are mutually exclusive: --resume continues an "
            "interrupted run (and its lr schedule), --init_from starts a new stage."
        )
    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    logger.info("device=%s precision=%s", device, dtype)

    config = OmegaConf.load(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.num_workers is not None:
        config.system.num_workers = args.num_workers

    norm_stats_path = resolve_path(
        args.norm_stats_path or resolve_path(args.config).parent / "norm_stats.json"
    )
    if not norm_stats_path.exists():
        raise SystemExit(
            f"normalization stats not found: {norm_stats_path}\n"
            f"Generate them first: python policy/lila_wam/compute_norm_stats.py "
            f"--config {args.config} --output {norm_stats_path}"
        )

    epochs = int(config.training.epochs)
    batch_size = int(config.training.batch_size)
    grad_accum_steps = int(config.training.grad_accum_steps)
    grad_clip_norm = float(config.training.grad_clip_norm)
    save_interval_epoch = int(config.training.save_interval_epoch)
    lr = float(config.training.learning_rate)
    lr_min = float(config.training.lr_min)
    if lr_min > lr:
        logger.warning(
            "lr_min (%.2e) > learning_rate (%.2e): CosineAnnealingLR would raise the lr", lr_min, lr
        )

    # ------------------------------------------------------------------ data
    ensure_frame_cache(config)
    train_dataset = create_dataset(config, val=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.system.num_workers),
        pin_memory=bool(config.system.pin_memory),
        collate_fn=upstream_collate_fn(),
        drop_last=True,
        persistent_workers=int(config.system.num_workers) > 0,
    )
    logger.info(
        "dataset frames=%d batches/epoch=%d batch=%d",
        len(train_dataset),
        len(train_dataloader),
        batch_size,
    )

    # ----------------------------------------------------------------- model
    model, action_model = build_model(
        config,
        norm_stats_path=norm_stats_path,
        device=device,
        dtype=dtype,
        train_config=build_train_config(config),
    )
    action_model.train()

    _, _, _ = upstream_models()
    from lila_upstream.utils.train_utils import count_parameters

    count_parameters(action_model, model_name="LiLa-WAM action model")

    optimizer = AdamW(
        action_model.parameters(),
        lr=lr,
        betas=tuple(config.training.betas),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_dataloader), eta_min=lr_min
    )

    start_epoch = 0
    global_step = 0
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        action_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info(
            "initialized weights from %s (epoch %s); fresh optimizer, lr %.2e -> %.2e",
            args.init_from,
            checkpoint.get("epoch", "?"),
            lr,
            lr_min,
        )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        action_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(
            checkpoint.get("global_step", start_epoch * len(train_dataloader) // grad_accum_steps)
        )
        logger.info("resumed from %s at epoch=%d step=%d", args.resume, start_epoch, global_step)

    # ------------------------------------------------------------------- run
    run_name = args.run_name or f"sft_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = resolve_path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, run_dir / "config.yaml")
    shutil.copyfile(norm_stats_path, run_dir / "norm_stats.json")
    logger.info("checkpoints -> %s", run_dir)

    loss_log = run_dir / "train_loss.csv"
    loss_log.write_text("epoch,step,global_step,loss\n")

    use_future_feat = bool((config.model.get("future_feat", {}) or {}).get("enabled", False))
    stop = False

    def prune_checkpoints():
        if args.keep_last_n <= 0:
            return
        saved = sorted(
            run_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        for stale in saved[: -args.keep_last_n]:
            stale.unlink()
            logger.info("pruned %s (--keep_last_n=%d)", stale.name, args.keep_last_n)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        seen_batches = 0
        optimizer.zero_grad()
        started = time.time()

        for step, batch in enumerate(train_dataloader):
            if batch is None:
                continue
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                loss, info = model(batch)
                loss = loss / grad_accum_steps
            loss.backward()

            step_loss = loss.item() * grad_accum_steps
            epoch_loss += step_loss
            seen_batches += 1

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                with loss_log.open("a") as handle:
                    handle.write(f"{epoch + 1},{step + 1},{global_step},{step_loss:.6f}\n")

                if global_step % 20 == 0:
                    message = (
                        f"epoch [{epoch + 1}/{epochs}] step [{step + 1}/{len(train_dataloader)}] "
                        f"loss {info['loss_mse'] * grad_accum_steps:.4f} "
                        f"lr {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    if use_future_feat:
                        message += f" future_feat {info.get('loss_future_feat', 0.0):.4f}"
                    logger.info(message)

                if args.max_steps is not None and global_step >= args.max_steps:
                    logger.info("reached --max_steps=%d, stopping", args.max_steps)
                    stop = True
                    break

        average_loss = epoch_loss / max(seen_batches, 1)
        logger.info(
            "=== epoch %d done. avg loss %.4f | %.1fs ===",
            epoch + 1,
            average_loss,
            time.time() - started,
        )

        is_last = (epoch + 1) == epochs
        if stop or is_last or (epoch + 1) % save_interval_epoch == 0:
            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": action_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": average_loss,
                },
                checkpoint_path,
            )
            logger.info("saved %s", checkpoint_path)
            prune_checkpoints()
        if stop:
            break

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "config": str(Path(args.config).resolve()),
                "norm_stats": str(norm_stats_path),
                "cameras": [str(c) for c in config.dataset.camera_names],
                "epochs_completed": epochs if not stop else epoch + 1,
                "global_step": global_step,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    logger.info("training complete")


if __name__ == "__main__":
    main()
