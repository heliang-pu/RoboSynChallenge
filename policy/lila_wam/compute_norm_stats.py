#!/usr/bin/env python
"""Compute LiLa-WAM min/max normalization stats from LeRobot v2.1 datasets.

    python policy/lila_wam/compute_norm_stats.py --config <config.yaml> --output <norm_stats.json>

Mirrors upstream ``utils/calc_stat_remove_outlier.py``: per-dimension min/max
over every frame, with episodes whose non-gripper joint commands leave [-pi, pi]
excluded so a single broken demo cannot stretch the normalization range.

The output keeps upstream's ``robotwin2`` top-level key because
``VLAWrapper.load_norm_stats`` reads that key by name.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from _bootstrap import add_repo_root

add_repo_root()

from omegaconf import OmegaConf  # noqa: E402

from policy.lila_wam.lerobot_v21 import (  # noqa: E402
    DEFAULT_ACTION_KEY,
    DEFAULT_STATE_KEY,
    read_episode_table,
)
from policy.lila_wam.lila_dataset import load_metas  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STATS_KEY = "robotwin2"   # read verbatim by upstream VLAWrapper.load_norm_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="destination json")
    parser.add_argument(
        "--gripper-dims",
        type=int,
        nargs="*",
        default=[6, 13],
        help="action dims exempt from the [-pi, pi] outlier check (default: the two grippers)",
    )
    parser.add_argument(
        "--no-outlier-filter",
        action="store_true",
        help="keep every episode, including ones with out-of-range joint commands",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    state_key = str(config.dataset.get("state_key", DEFAULT_STATE_KEY))
    action_key = str(config.dataset.get("action_key", DEFAULT_ACTION_KEY))

    metas = load_metas(config)
    action_min = action_max = state_min = state_max = None
    kept = skipped = 0
    outliers: list[str] = []

    for meta in metas:
        for episode in meta.episodes:
            table = read_episode_table(
                episode.parquet_path, [state_key, action_key], episode.length
            )
            action = table[action_key]
            state = table[state_key]

            if not args.no_outlier_filter:
                joint_dims = [d for d in range(action.shape[1]) if d not in set(args.gripper_dims)]
                bad = np.abs(action[:, joint_dims]) > np.pi
                if bad.any():
                    skipped += 1
                    outliers.append(str(episode.parquet_path))
                    continue

            kept += 1
            a_min, a_max = action.min(axis=0), action.max(axis=0)
            s_min, s_max = state.min(axis=0), state.max(axis=0)
            if action_min is None:
                action_min, action_max, state_min, state_max = a_min, a_max, s_min, s_max
            else:
                action_min = np.minimum(action_min, a_min)
                action_max = np.maximum(action_max, a_max)
                state_min = np.minimum(state_min, s_min)
                state_max = np.maximum(state_max, s_max)

    if action_min is None:
        raise SystemExit(
            "every episode was rejected by the outlier filter; "
            "re-run with --no-outlier-filter to inspect the data"
        )

    payload = {
        STATS_KEY: {
            "meta": {
                "source": "lerobot_v2.1",
                "roots": [str(meta.root) for meta in metas],
                "tasks": [meta.episodes[0].task for meta in metas],
                "state_key": state_key,
                "action_key": action_key,
                "episodes_used": kept,
                "episodes_skipped_outlier": skipped,
                "outlier_filter": not args.no_outlier_filter,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "action": {
                "min": action_min.astype(float).tolist(),
                "max": action_max.astype(float).tolist(),
                "dim": int(action_min.shape[0]),
            },
            "state": {
                "min": state_min.astype(float).tolist(),
                "max": state_max.astype(float).tolist(),
                "dim": int(state_min.shape[0]),
            },
        }
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    logger.info(
        "wrote %s (action dim=%d, state dim=%d, %d episodes used, %d skipped as outliers)",
        output,
        action_min.shape[0],
        state_min.shape[0],
        kept,
        skipped,
    )
    for path in outliers[:10]:
        logger.warning("outlier episode excluded from stats: %s", path)
    if len(outliers) > 10:
        logger.warning("... and %d more", len(outliers) - 10)


if __name__ == "__main__":
    main()
