#!/usr/bin/env python3
"""Evaluate 20 evenly spaced expert episodes and render one video per run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:7891")
    parser.add_argument("--num-evals", type=int, default=20)
    parser.add_argument("--replan-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    episodes = sorted(args.jsonl_dir.glob("episode_*.jsonl"))
    if len(episodes) < args.num_evals:
        raise ValueError(
            f"Requested {args.num_evals} evals but only found {len(episodes)} episodes"
        )
    indices = np.linspace(0, len(episodes) - 1, args.num_evals, dtype=int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = Path(__file__).with_name("render_checkpoint_eval_video.py")
    results = []
    for eval_index, episode_index in enumerate(indices, start=1):
        episode = episodes[int(episode_index)]
        stem = f"eval-{eval_index:02d}_{episode.stem}"
        video = args.output_dir / f"{stem}.mp4"
        metrics = args.output_dir / f"{stem}.json"
        command = [
            sys.executable,
            str(renderer),
            "--episode-jsonl",
            str(episode),
            "--image-dir",
            str(args.image_dir),
            "--server-url",
            args.server_url,
            "--replan-steps",
            str(args.replan_steps),
            "--seed",
            str(args.seed + eval_index * 1000),
            "--output",
            str(video),
            "--metrics-output",
            str(metrics),
        ]
        print(
            f"EVAL_START {eval_index}/{args.num_evals} {episode.stem}",
            flush=True,
        )
        subprocess.run(command, check=True)
        result = json.loads(metrics.read_text())
        result["eval_index"] = eval_index
        result["video"] = video.name
        result["metrics"] = metrics.name
        results.append(result)
        print(
            f"EVAL_DONE {eval_index}/{args.num_evals} "
            f"mae={result['action_mae']:.6f} video={video.name}",
            flush=True,
        )

    maes = np.asarray([item["action_mae"] for item in results])
    p95s = np.asarray([item["action_mae_p95"] for item in results])
    warm_latencies = np.asarray([item["warm_latency_ms"] for item in results])
    summary = {
        "checkpoint": "checkpoint-400",
        "evaluation_type": "expert_trajectory_policy_replay",
        "num_evals": len(results),
        "all_finite": all(item["all_finite"] for item in results),
        "mean_action_mae": float(maes.mean()),
        "median_action_mae": float(np.median(maes)),
        "worst_episode_action_mae": float(maes.max()),
        "mean_action_mae_p95": float(p95s.mean()),
        "mean_warm_latency_ms": float(warm_latencies.mean()),
        "episodes": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("EVAL20_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
