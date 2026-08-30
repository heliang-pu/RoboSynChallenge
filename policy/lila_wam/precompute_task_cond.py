#!/usr/bin/env python
"""Precompute LiLa-WAM Visual Transition Tokens (VTT) from a LeRobot v2.1 dataset.

    python policy/lila_wam/precompute_task_cond.py --config <config.yaml>

Upstream's task condition is a language-free task descriptor: the frozen DINOv3
CLS feature of an episode's last frame minus that of its first frame, averaged
over all episodes of a task. This is the LeRobot-sourced equivalent of
``utils/precompute_task_cond.py``; it reads the frames from the JPEG cache built
by ``build_frame_cache.py`` and writes ``<out>/<task>/task_cond.npy``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
import torch

from _bootstrap import add_repo_root, resolve_path

add_repo_root()

from omegaconf import OmegaConf  # noqa: E402

from policy.lila_wam.lerobot_v21 import FrameCache, cache_path_for, feature_key_to_camera  # noqa: E402
from policy.lila_wam.lila_dataset import (  # noqa: E402
    ensure_frame_cache,
    normalize_image,
    resolve_cache_dir,
)
from policy.lila_wam.lila_model import load_vision_encoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="defaults to dataset.task_cond_dir from the config",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    output_dir = resolve_path(args.output_dir or config.dataset.task_cond_dir)
    image_size = tuple(int(v) for v in config.dataset.image_size)

    cache_file = cache_path_for(resolve_cache_dir(config), image_size)
    if not cache_file.exists():
        logger.info("frame cache missing, building it first ...")
        cache_file = ensure_frame_cache(config)
    cache = FrameCache(cache_file)

    primary_camera = feature_key_to_camera(str(config.dataset.camera_names[0]))
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device != "cpu" and torch.cuda.is_bf16_supported()) else torch.float32

    encoder, hidden_size, _registers, _patch = load_vision_encoder(
        str(resolve_path(config.model.vision_encoder.checkpoint_path)), dtype, device
    )

    diffs: dict[str, list[np.ndarray]] = defaultdict(list)
    for ep in range(cache.num_episodes):
        start = int(cache.episode_start[ep])
        end = int(cache.episode_end[ep])
        task = cache.tasks[int(cache.episode_task[ep])]

        first = normalize_image(cache.read_jpeg(primary_camera, start))
        last = normalize_image(cache.read_jpeg(primary_camera, end - 1))
        pixels = torch.from_numpy(np.stack([first, last], axis=0)).to(device, dtype)

        cls = encoder(pixel_values=pixels, return_dict=True).last_hidden_state[:, 0, :]
        cls = cls.float().cpu().numpy()
        diffs[task].append((cls[1] - cls[0]).astype(np.float32))

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "lerobot_v2.1",
        "vision_encoder": str(config.model.vision_encoder.checkpoint_path),
        "hidden_size": int(hidden_size),
        "image_size": list(image_size),
        "camera": primary_camera,
        "feature": "last_layer_cls_diff (last_frame - first_frame), averaged per task",
        "tasks": {},
    }
    for task, vectors in sorted(diffs.items()):
        stacked = np.stack(vectors, axis=0)
        task_cond = stacked.mean(axis=0).astype(np.float32)
        (output_dir / task).mkdir(parents=True, exist_ok=True)
        np.save(output_dir / task / "task_cond.npy", task_cond)
        meta["tasks"][task] = {
            "num_episodes": int(stacked.shape[0]),
            "dim": int(task_cond.shape[0]),
            "norm": float(np.linalg.norm(task_cond)),
        }
        logger.info(
            "%s: %d episodes, dim=%d, |cond|=%.3f",
            task,
            stacked.shape[0],
            task_cond.shape[0],
            np.linalg.norm(task_cond),
        )

    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("wrote task condition vectors to %s", output_dir)


if __name__ == "__main__":
    main()
