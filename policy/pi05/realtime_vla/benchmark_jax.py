"""Measure the existing OpenPI/JAX policy latency on the same checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import jax
import numpy as np

from openpi.policies import policy_config
from openpi.training import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_base_robosynchallenge_full")
    parser.add_argument("--prompt", default="click the bell")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = policy_config.create_trained_policy(config.get_config(args.config_name), args.checkpoint)
    rng = np.random.default_rng(args.seed)
    observation = {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(14, dtype=np.float32),
        "prompt": args.prompt,
    }
    noise = rng.standard_normal((50, 32), dtype=np.float32)
    for _ in range(args.warmup):
        policy.infer(observation, noise=noise)
    jax.block_until_ready(policy.infer(observation, noise=noise)["actions"])

    latencies_ms: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        actions = policy.infer(observation, noise=noise)["actions"]
        jax.block_until_ready(actions)
        latencies_ms.append((time.perf_counter() - started) * 1000)
    result = {
        "backend": "openpi-jax",
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

