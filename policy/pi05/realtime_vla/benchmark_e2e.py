"""Benchmark realtime-vla with RoboSyn preprocessing and output transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from policy.pi05.realtime_vla.accelerated_policy import RealtimeVlaPi05Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--realtime-vla-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="click the bell")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    observation = {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(14, dtype=np.float32),
        "prompt": args.prompt,
    }
    noise = rng.standard_normal((50, 32), dtype=np.float32)
    policy = RealtimeVlaPi05Policy(
        converted_checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
        tokenizer_path=args.tokenizer,
        realtime_vla_dir=args.realtime_vla_dir,
        prompt_for_allocation=args.prompt,
    )
    for _ in range(args.warmup):
        policy.infer(observation, noise=noise)

    latencies_ms: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        policy.infer(observation, noise=noise)
        latencies_ms.append((time.perf_counter() - started) * 1000)
    result = {
        "backend": "realtime-vla-triton-e2e",
        "iterations": args.iterations,
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "hz_from_mean": 1000.0 / statistics.fmean(latencies_ms),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

