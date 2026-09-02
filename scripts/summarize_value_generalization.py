#!/usr/bin/env python
"""Summarize value-model predictions on one or more derived QC datasets.

Rows without predictions (NaN) are ignored. Episode identities are namespaced by
the dataset label so results from multiple, disjoint datasets can be combined.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass
class Episode:
    source: str
    index: int
    task: str
    length: int
    success: bool


def parse_mapping(value: str, separator: str = "=") -> tuple[str, str]:
    if separator not in value:
        raise argparse.ArgumentTypeError(f"expected NAME{separator}VALUE, got {value!r}")
    key, item = value.split(separator, 1)
    if not key or not item:
        raise argparse.ArgumentTypeError(f"expected NAME{separator}VALUE, got {value!r}")
    return key, item


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


def bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray, seed: int = 20260826) -> list[float]:
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size < 2 or negative.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    aucs = np.empty(2000, dtype=np.float64)
    for i in range(aucs.size):
        p = positive[rng.integers(0, positive.size, positive.size)]
        n = negative[rng.integers(0, negative.size, negative.size)]
        aucs[i] = pairwise_auc(p, n)
    return [float(x) for x in np.quantile(aucs, [0.025, 0.975])]


def read_episode_metadata(source: str, root: Path) -> dict[tuple[str, int], Episode]:
    records: dict[tuple[str, int], Episode] = {}
    files = sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root}")
    for parquet_file in files:
        table = pq.read_table(
            parquet_file,
            columns=["episode_index", "tasks", "length", "episode_success"],
        )
        for row in table.to_pylist():
            tasks = row["tasks"]
            task = tasks[0] if isinstance(tasks, list) else str(tasks)
            label = str(row["episode_success"]).lower()
            if label not in {"success", "failure"}:
                raise ValueError(f"bad episode_success={label!r} in {parquet_file}")
            episode = Episode(
                source=source,
                index=int(row["episode_index"]),
                task=task,
                length=int(row["length"]),
                success=label == "success",
            )
            records[(source, episode.index)] = episode
    return records


def load_predictions(
    datasets: list[tuple[str, str]], tag: str
) -> tuple[dict[tuple[str, int], Episode], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value_col = f"complementary_info.value_{tag}"
    advantage_col = f"complementary_info.advantage_{tag}"
    indicator_col = f"complementary_info.acp_indicator_{tag}"
    metadata: dict[tuple[str, int], Episode] = {}
    keys: list[tuple[str, int]] = []
    frame_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    advantage_parts: list[np.ndarray] = []
    indicator_parts: list[np.ndarray] = []

    for source, root_string in datasets:
        root = Path(root_string)
        metadata.update(read_episode_metadata(source, root))
        files = sorted(glob.glob(str(root / "data/chunk-*/file-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no data parquet under {root}")
        for parquet_file in files:
            schema = pq.read_schema(parquet_file)
            if value_col not in schema.names:
                continue
            columns = ["episode_index", "frame_index", value_col]
            has_advantage = advantage_col in schema.names
            has_indicator = indicator_col in schema.names
            if has_advantage:
                columns.append(advantage_col)
            if has_indicator:
                columns.append(indicator_col)
            table = pq.read_table(parquet_file, columns=columns)
            episode_index = np.asarray(table["episode_index"], dtype=np.int64)
            frame = np.asarray(table["frame_index"], dtype=np.int64)
            value = np.asarray(table[value_col], dtype=np.float32)
            keep = np.isfinite(value)
            keys.extend((source, int(index)) for index in episode_index[keep])
            frame_parts.append(frame[keep])
            value_parts.append(value[keep])
            advantage_parts.append(
                np.asarray(table[advantage_col], dtype=np.float32)[keep]
                if has_advantage
                else np.full(int(keep.sum()), np.nan, dtype=np.float32)
            )
            indicator_parts.append(
                np.asarray(table[indicator_col], dtype=np.int64)[keep]
                if has_indicator
                else np.zeros(int(keep.sum()), dtype=np.int64)
            )

    if not value_parts or not any(part.size for part in value_parts):
        raise ValueError(f"no finite predictions found for tag {tag!r}")
    return (
        metadata,
        np.asarray(keys, dtype=object),
        np.concatenate(frame_parts),
        np.concatenate(value_parts),
        np.concatenate(advantage_parts),
        np.concatenate(indicator_parts),
    )


def compute_metrics(datasets: list[tuple[str, str]], tag: str, step: int) -> dict:
    metadata, key_array, frame, value, advantage, indicator = load_predictions(datasets, tag)
    keys = [tuple(item) for item in key_array.tolist()]
    episodes = [metadata[key] for key in keys]
    success = np.asarray([episode.success for episode in episodes], dtype=bool)

    task_max: dict[str, int] = {}
    for episode in metadata.values():
        task_max[episode.task] = max(task_max.get(episode.task, 0), episode.length)
    target = np.empty(value.size, dtype=np.float32)
    for i, episode in enumerate(episodes):
        maximum = task_max[episode.task]
        remaining = episode.length - int(frame[i]) - 1
        raw = -float(remaining)
        if not episode.success:
            raw -= float(maximum)
        target[i] = np.clip(raw / (2.0 * maximum), -1.0, 0.0)

    unique_keys = sorted(set(keys))
    episode_label = np.asarray([metadata[key].success for key in unique_keys], dtype=bool)
    first_scores = np.empty(len(unique_keys), dtype=np.float32)
    mean_scores = np.empty(len(unique_keys), dtype=np.float32)
    last_scores = np.empty(len(unique_keys), dtype=np.float32)
    for i, key in enumerate(unique_keys):
        mask = np.asarray([row_key == key for row_key in keys], dtype=bool)
        local_frame = frame[mask]
        local_value = value[mask]
        first_scores[i] = local_value[np.argmin(local_frame)]
        last_scores[i] = local_value[np.argmax(local_frame)]
        mean_scores[i] = float(local_value.mean())

    def classification(scores: np.ndarray) -> dict:
        positive = scores[episode_label]
        negative = scores[~episode_label]
        return {
            "roc_auc": pairwise_auc(positive, negative),
            "roc_auc_ci95": bootstrap_auc_ci(episode_label, scores),
            "pr_auc": average_precision(episode_label, scores),
            "success_mean": float(positive.mean()) if positive.size else float("nan"),
            "failure_mean": float(negative.mean()) if negative.size else float("nan"),
            "gap": float(positive.mean() - negative.mean())
            if positive.size and negative.size
            else float("nan"),
        }

    error = value - target
    finite_advantage = np.isfinite(advantage)
    return {
        "step": step,
        "tag": tag,
        "frames": int(value.size),
        "episodes": len(unique_keys),
        "success_episodes": int(episode_label.sum()),
        "failure_episodes": int((~episode_label).sum()),
        "frame_regression": {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "bias": float(np.mean(error)),
            "success_mae": float(np.mean(np.abs(error[success]))) if success.any() else float("nan"),
            "failure_mae": float(np.mean(np.abs(error[~success]))) if (~success).any() else float("nan"),
            "pearson": float(np.corrcoef(value, target)[0, 1]),
        },
        "first_frame": classification(first_scores),
        "episode_mean": classification(mean_scores),
        "last_frame": classification(last_scores),
        "prediction": {
            "mean": float(value.mean()),
            "std": float(value.std(ddof=1)),
            "min": float(value.min()),
            "max": float(value.max()),
        },
        "advantage": {
            "std": float(np.std(advantage[finite_advantage], ddof=1)),
            "abs_lt_001_pct": float(np.mean(np.abs(advantage[finite_advantage]) < 0.01) * 100),
            "indicator_pct": float(indicator.mean() * 100),
        },
    }


def parse_training_log(path: Path) -> dict:
    pattern = re.compile(
        r"step:(?P<step>\d+|\d+K)\s+.*?loss:(?P<loss>[0-9.eE+-]+).*?grdn:(?P<grad>[0-9.eE+-]+).*?lr:(?P<lr>[0-9.eE+-]+)"
    )
    points = []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        raw_step = match.group("step")
        step = int(raw_step[:-1]) * 1000 if raw_step.endswith("K") else int(raw_step)
        points.append(
            {
                "step": step,
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grad")),
                "lr": float(match.group("lr")),
            }
        )
    return {"points": points, "last": points[-1] if points else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, type=parse_mapping)
    parser.add_argument("--tag", action="append", required=True, type=parse_mapping)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for step_string, tag in args.tag:
        results.append(compute_metrics(args.dataset, tag=tag, step=int(step_string)))
    output = {
        "datasets": [{"name": name, "path": path} for name, path in args.dataset],
        "checkpoints": results,
    }
    if args.train_log:
        output["training"] = parse_training_log(args.train_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=True) + "\n")
    print(json.dumps(output, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
