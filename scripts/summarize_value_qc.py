#!/usr/bin/env python

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    scores = positive[:, None] - negative[None, :]
    return float(np.mean(scores > 0) + 0.5 * np.mean(scores == 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.dataset)
    value_col = f"complementary_info.value_{args.tag}"
    advantage_col = f"complementary_info.advantage_{args.tag}"
    indicator_col = f"complementary_info.acp_indicator_{args.tag}"
    columns = ["episode_index", "frame_index", value_col, advantage_col, indicator_col]

    arrays: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    for parquet_file in sorted(glob.glob(str(root / "data/chunk-*/file-*.parquet"))):
        table = pq.read_table(parquet_file, columns=columns)
        for column in columns:
            arrays[column].append(np.asarray(table[column].to_numpy()))

    values = {column: np.concatenate(parts) for column, parts in arrays.items()}
    keep = ~np.isnan(values[value_col].astype(np.float32))
    episode = values["episode_index"].astype(np.int64)[keep]
    frame = values["frame_index"].astype(np.int64)[keep]
    value = values[value_col].astype(np.float32)[keep]
    advantage = values[advantage_col].astype(np.float32)[keep]
    indicator = values[indicator_col].astype(np.int64)[keep]

    label_records = json.loads((root / "episode_success.json").read_text())["episodes"]
    labels = {int(record["episode_index"]): bool(record["success"]) for record in label_records}
    success = np.asarray([labels[int(index)] for index in episode], dtype=bool)
    rollout = episode >= 806

    groups = {
        "clean_success": episode < 756,
        "syn_success": (episode >= 756) & (episode < 806),
        "rollout_success": rollout & success,
        "rollout_failure": rollout & ~success,
    }

    def group_metrics(mask: np.ndarray) -> dict:
        selected_episodes = np.unique(episode[mask])
        first_mask = mask & (frame == 0)
        last_indices = []
        for ep in selected_episodes:
            ep_indices = np.flatnonzero(mask & (episode == ep))
            last_indices.append(ep_indices[np.argmax(frame[ep_indices])])
        last_indices_np = np.asarray(last_indices, dtype=np.int64)
        return {
            "episodes": int(selected_episodes.size),
            "frames": int(np.sum(mask)),
            "first_value_mean": float(np.mean(value[first_mask])),
            "first_value_std": float(np.std(value[first_mask], ddof=1)) if np.sum(first_mask) > 1 else 0.0,
            "last_value_mean": float(np.mean(value[last_indices_np])),
            "trajectory_delta_mean": float(np.mean(value[last_indices_np]) - np.mean(value[first_mask])),
            "value_mean": float(np.mean(value[mask])),
            "advantage_std": float(np.std(advantage[mask], ddof=1)),
            "abs_advantage_lt_001_pct": float(np.mean(np.abs(advantage[mask]) < 0.01) * 100),
            "indicator_pct": float(np.mean(indicator[mask]) * 100),
        }

    first = frame == 0
    rollout_success_first = value[first & rollout & success]
    rollout_failure_first = value[first & rollout & ~success]

    episode_means = {}
    for ep in np.unique(episode):
        episode_means[int(ep)] = float(np.mean(value[episode == ep]))
    rollout_success_means = np.asarray(
        [episode_means[ep] for ep in episode_means if ep >= 806 and labels[ep]], dtype=np.float32
    )
    rollout_failure_means = np.asarray(
        [episode_means[ep] for ep in episode_means if ep >= 806 and not labels[ep]], dtype=np.float32
    )

    result = {
        "step": args.step,
        "tag": args.tag,
        "frames": int(value.size),
        "episodes": int(np.unique(episode).size),
        "overall": {
            "value_mean": float(np.mean(value)),
            "value_std": float(np.std(value, ddof=1)),
            "advantage_mean": float(np.mean(advantage)),
            "advantage_std": float(np.std(advantage, ddof=1)),
            "abs_advantage_lt_001_pct": float(np.mean(np.abs(advantage) < 0.01) * 100),
            "indicator_pct": float(np.mean(indicator) * 100),
        },
        "groups": {
            name: group_metrics(mask) for name, mask in groups.items() if np.any(mask)
        },
        "rollout": {
            "first_value_gap": float(np.mean(rollout_success_first) - np.mean(rollout_failure_first)),
            "first_value_auc": pairwise_auc(rollout_success_first, rollout_failure_first),
            "episode_mean_value_gap": float(
                np.mean(rollout_success_means) - np.mean(rollout_failure_means)
            ),
            "episode_mean_value_auc": pairwise_auc(
                rollout_success_means, rollout_failure_means
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
