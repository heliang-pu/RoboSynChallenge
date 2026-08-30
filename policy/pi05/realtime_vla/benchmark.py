"""Benchmark a converted Pi0.5 checkpoint with realtime-vla Triton kernels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import time

import numpy as np
import torch

from policy.pi05.realtime_vla.tokenizer_adapter import SentencePieceAutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REALTIME_VLA = REPO_ROOT.parent / "realtime-vla"
DEFAULT_TOKENIZER = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="click the bell")
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(os.environ.get("OPENPI_TOKENIZER_PATH", DEFAULT_TOKENIZER)),
    )
    parser.add_argument(
        "--realtime-vla-dir",
        type=Path,
        default=Path(os.environ.get("REALTIME_VLA_DIR", DEFAULT_REALTIME_VLA)),
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    args = parse_args()
    realtime_vla_dir = args.realtime_vla_dir.expanduser().resolve()
    if not (realtime_vla_dir / "pi05_infer.py").is_file():
        raise FileNotFoundError(f"pi05_infer.py not found below {realtime_vla_dir}")
    sys.path.insert(0, str(realtime_vla_dir))
    import pi05_infer  # noqa: PLC0415

    pi05_infer.AutoTokenizer = SentencePieceAutoTokenizer
    with args.checkpoint.open("rb") as stream:
        checkpoint = pickle.load(stream)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_started = time.perf_counter()
    infer = pi05_infer.Pi05Inference(
        checkpoint=checkpoint,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        tokenizer_path=str(args.tokenizer_path.expanduser().resolve()),
        discrete_state_input=True,
        max_prompt_text=args.prompt,
        state_dim_for_max_prompt=args.state_dim,
    )
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_started

    images = torch.rand(args.num_views, 224, 224, 3, device="cuda", dtype=torch.bfloat16) * 2 - 1
    noise = torch.randn(args.chunk_size, 32, device="cuda", dtype=torch.bfloat16)
    state_tokens = np.full(args.state_dim, 127, dtype=np.int32)

    for _ in range(args.warmup):
        infer.forward(images, noise, args.prompt, state_tokens)
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        infer.forward(images, noise, args.prompt, state_tokens)
        end.record()
        end.synchronize()
        latencies_ms.append(float(start.elapsed_time(end)))

    result = {
        "backend": "realtime-vla-triton",
        "gpu": torch.cuda.get_device_name(),
        "num_views": args.num_views,
        "chunk_size": args.chunk_size,
        "prompt": args.prompt,
        "iterations": args.iterations,
        "initialization_seconds": init_seconds,
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p90_ms": percentile(latencies_ms, 90),
        "p99_ms": percentile(latencies_ms, 99),
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

