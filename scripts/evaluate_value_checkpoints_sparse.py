#!/usr/bin/env python
"""Fast checkpoint comparison on a fixed sparse set of trajectory frames.

The full value-inference command is intended to annotate every frame for VLA
training.  Quality control only needs a deterministic set of phases from each
episode, so this tool evaluates evenly spaced frames without modifying datasets.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_value_infer import _resolve_pretrained_model_dir
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config


def mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    name, item = value.split("=", 1)
    if not name or not item:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    return name, item


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    delta = positive[:, None] - negative[None, :]
    return float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order].astype(np.int64)
    precision = np.cumsum(sorted_labels) / np.arange(1, labels.size + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, seed: int) -> list[float]:
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size < 2 or negative.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values = np.empty(2000, dtype=np.float64)
    for i in range(values.size):
        p = positive[rng.integers(0, positive.size, positive.size)]
        n = negative[rng.integers(0, negative.size, negative.size)]
        values[i] = pairwise_auc(p, n)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def classification(labels: np.ndarray, scores: np.ndarray, seed: int) -> dict:
    positive = scores[labels]
    negative = scores[~labels]
    return {
        "roc_auc": pairwise_auc(positive, negative),
        "roc_auc_ci95": bootstrap_auc(labels, scores, seed),
        "pr_auc": average_precision(labels, scores),
        "success_mean": float(positive.mean()) if positive.size else float("nan"),
        "failure_mean": float(negative.mean()) if negative.size else float("nan"),
        "gap": float(positive.mean() - negative.mean())
        if positive.size and negative.size
        else float("nan"),
    }


def summarize(rows: list[dict], seed: int) -> dict:
    value = np.asarray([row["value"] for row in rows], dtype=np.float32)
    target = np.asarray([row["target"] for row in rows], dtype=np.float32)
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    error = value - target

    by_episode: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[(row["source"], row["episode_index"])].append(row)
    episode_rows = []
    for key, items in sorted(by_episode.items()):
        items.sort(key=lambda item: item["phase"])
        episode_rows.append(
            {
                "key": key,
                "success": bool(items[0]["success"]),
                "first": float(items[0]["value"]),
                "mean": float(np.mean([item["value"] for item in items])),
                "last": float(items[-1]["value"]),
                "delta": float(items[-1]["value"] - items[0]["value"]),
            }
        )
    episode_labels = np.asarray([row["success"] for row in episode_rows], dtype=bool)
    first = np.asarray([row["first"] for row in episode_rows], dtype=np.float32)
    mean = np.asarray([row["mean"] for row in episode_rows], dtype=np.float32)
    last = np.asarray([row["last"] for row in episode_rows], dtype=np.float32)
    delta = np.asarray([row["delta"] for row in episode_rows], dtype=np.float32)
    return {
        "frames": len(rows),
        "episodes": len(episode_rows),
        "success_episodes": int(episode_labels.sum()),
        "failure_episodes": int((~episode_labels).sum()),
        "frame_regression": {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "bias": float(np.mean(error)),
            "success_mae": float(np.mean(np.abs(error[success]))) if success.any() else float("nan"),
            "failure_mae": float(np.mean(np.abs(error[~success]))) if (~success).any() else float("nan"),
            "pearson": float(np.corrcoef(value, target)[0, 1]),
        },
        "first_frame": classification(episode_labels, first, seed),
        "episode_mean": classification(episode_labels, mean, seed + 1),
        "last_frame": classification(episode_labels, last, seed + 2),
        "trajectory": {
            "success_delta_mean": float(delta[episode_labels].mean()) if episode_labels.any() else float("nan"),
            "failure_delta_mean": float(delta[~episode_labels].mean()) if (~episode_labels).any() else float("nan"),
        },
        "prediction": {
            "mean": float(value.mean()),
            "std": float(value.std(ddof=1)),
            "min": float(value.min()),
            "max": float(value.max()),
        },
    }


def labels_for_dataset(dataset: LeRobotDataset) -> dict[int, bool]:
    episodes = dataset.meta.episodes.with_format(None)
    result: dict[int, bool] = {}
    if "episode_success" in episodes.column_names:
        values = episodes[:]
        for index, label in zip(values["episode_index"], values["episode_success"], strict=True):
            normalized = str(label).lower()
            if normalized not in {"success", "failure"}:
                raise ValueError(f"bad episode_success={label!r} in {dataset.root}")
            result[int(index)] = normalized == "success"
    else:
        # Fresh rollout datasets keep outcomes in a sidecar until they are
        # merged into a labeled training pool.
        sidecar = Path(dataset.root) / "episode_success.json"
        if not sidecar.is_file():
            raise KeyError(f"{dataset.root} has neither episode_success metadata nor sidecar")
        records = json.loads(sidecar.read_text()).get("episodes", [])
        result = {int(record["episode_index"]): bool(record["success"]) for record in records}
    return result


def selected_rows(dataset: LeRobotDataset, phases: int) -> tuple[list[int], list[dict]]:
    frames = dataset.hf_dataset.with_format(None)
    episode = np.asarray(frames["episode_index"], dtype=np.int64)
    frame = np.asarray(frames["frame_index"], dtype=np.int64)
    labels = labels_for_dataset(dataset)
    selected: list[int] = []
    metadata: list[dict] = []
    for episode_index in np.unique(episode):
        positions = np.flatnonzero(episode == episode_index)
        positions = positions[np.argsort(frame[positions])]
        offsets = np.unique(np.rint(np.linspace(0, len(positions) - 1, phases)).astype(np.int64))
        for offset in offsets:
            relative_index = int(positions[offset])
            selected.append(relative_index)
            metadata.append(
                {
                    "episode_index": int(episode_index),
                    "frame_index": int(frame[relative_index]),
                    "episode_length": int(len(positions)),
                    "phase": float(offset / max(1, len(positions) - 1)),
                    "success": bool(labels[int(episode_index)]),
                }
            )
    return selected, metadata


def load_model(checkpoint: Path, dataset: LeRobotDataset, device: torch.device):
    pretrained = _resolve_pretrained_model_dir(str(checkpoint), "last")
    config = PreTrainedConfig.from_pretrained(pretrained)
    if not isinstance(config, Pistar06Config):
        raise TypeError(f"expected Pistar06Config, got {type(config)}")
    config.pretrained_path = pretrained
    config.device = device.type
    model = make_policy(cfg=config, ds_meta=dataset.meta, rename_map={})
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=pretrained,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    model.eval()
    return model, preprocessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, type=mapping)
    parser.add_argument("--episodes", action="append", default=[], type=mapping)
    parser.add_argument("--group", action="append", required=True, type=mapping)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        type=mapping,
        help="Optional per-source canonical task text override for malformed rollout task metadata.",
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--phases", type=int, default=7)
    parser.add_argument("--task-max-length", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.phases < 2:
        raise ValueError("--phases must be at least 2")
    episode_map = {name: json.loads(value) for name, value in args.episodes}
    group_map = dict(args.group)
    task_map = dict(args.task)
    sources = []
    for name, root in args.dataset:
        if name not in group_map:
            raise KeyError(f"missing --group for {name}")
        dataset = LeRobotDataset(
            repo_id=f"local/{Path(root).name}",
            root=root,
            episodes=episode_map.get(name),
            download_videos=True,
        )
        indices, metadata = selected_rows(dataset, args.phases)
        sources.append(
            {
                "name": name,
                "group": group_map[name],
                "root": root,
                "dataset": dataset,
                "indices": indices,
                "metadata": metadata,
                "task": task_map.get(name),
            }
        )

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    output = {
        "config": {
            "phases": args.phases,
            "task_max_length": args.task_max_length,
            "batch_size": args.batch_size,
            "sources": [
                {
                    "name": source["name"],
                    "group": source["group"],
                    "root": source["root"],
                    "episodes": len(set(row["episode_index"] for row in source["metadata"])),
                    "frames": len(source["metadata"]),
                }
                for source in sources
            ],
        },
        "checkpoints": [],
    }

    for step in args.steps:
        checkpoint = args.checkpoint_root / f"{step:06d}"
        print(f"evaluating checkpoint {checkpoint}", flush=True)
        model, preprocessor = load_model(checkpoint, sources[0]["dataset"], device)
        all_rows: list[dict] = []
        with torch.no_grad():
            for source in sources:
                loader = DataLoader(
                    Subset(source["dataset"], source["indices"]),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                )
                predictions: list[np.ndarray] = []
                for raw_batch in loader:
                    if source["task"] is not None:
                        batch_count = len(raw_batch["episode_index"])
                        raw_batch["task"] = [source["task"]] * batch_count
                    processed = preprocessor(raw_batch)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        values = model.predict_value(processed)
                    predictions.append(values.detach().cpu().float().numpy().reshape(-1))
                predicted = np.concatenate(predictions)
                if predicted.size != len(source["metadata"]):
                    raise RuntimeError("prediction count mismatch")
                for meta, value in zip(source["metadata"], predicted, strict=True):
                    remaining = meta["episode_length"] - meta["frame_index"] - 1
                    raw_target = -float(remaining)
                    if not meta["success"]:
                        raw_target -= float(args.task_max_length)
                    target = float(np.clip(raw_target / (2.0 * args.task_max_length), -1.0, 0.0))
                    all_rows.append(
                        {
                            **meta,
                            "source": source["name"],
                            "group": source["group"],
                            "value": float(value),
                            "target": target,
                        }
                    )

        by_group = {}
        for group in sorted(set(row["group"] for row in all_rows)):
            by_group[group] = summarize(
                [row for row in all_rows if row["group"] == group], args.seed + step
            )
        by_source = {}
        for source in sorted(set(row["source"] for row in all_rows)):
            by_source[source] = summarize(
                [row for row in all_rows if row["source"] == source], args.seed + step
            )
        output["checkpoints"].append(
            {"step": step, "groups": by_group, "sources": by_source}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n")
        del model, preprocessor
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(output, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
