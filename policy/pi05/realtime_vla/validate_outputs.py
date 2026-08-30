"""Generate deterministic backend outputs and compare Triton with OpenPI/JAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_input = subparsers.add_parser("make-input")
    make_input.add_argument("--output", type=Path, required=True)
    make_input.add_argument("--seed", type=int, default=7)
    make_input.add_argument("--prompt", default="click the bell")

    jax_parser = subparsers.add_parser("jax")
    jax_parser.add_argument("--input", type=Path, required=True)
    jax_parser.add_argument("--output", type=Path, required=True)
    jax_parser.add_argument("--checkpoint", type=Path, required=True)
    jax_parser.add_argument("--config-name", default="pi05_base_robosynchallenge_full")

    triton_parser = subparsers.add_parser("triton")
    triton_parser.add_argument("--input", type=Path, required=True)
    triton_parser.add_argument("--output", type=Path, required=True)
    triton_parser.add_argument("--checkpoint", type=Path, required=True)
    triton_parser.add_argument("--norm-stats", type=Path, required=True)
    triton_parser.add_argument("--tokenizer", type=Path, required=True)
    triton_parser.add_argument("--realtime-vla-dir", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--jax-output", type=Path, required=True)
    compare.add_argument("--triton-output", type=Path, required=True)
    compare.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_input(path: Path) -> tuple[dict[str, np.ndarray | str], np.ndarray]:
    data = np.load(path)
    observation = {
        "observation/image": data["base_image"],
        "observation/left_wrist_image": data["left_wrist_image"],
        "observation/right_wrist_image": data["right_wrist_image"],
        "observation/state": data["state"],
        "prompt": str(data["prompt"]),
    }
    return observation, data["noise"]


def main() -> None:
    args = parse_args()
    if args.command == "make-input":
        rng = np.random.default_rng(args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            base_image=rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            left_wrist_image=rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            right_wrist_image=rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            state=np.zeros(14, dtype=np.float32),
            noise=rng.standard_normal((50, 32), dtype=np.float32),
            prompt=np.asarray(args.prompt),
        )
        return

    if args.command == "jax":
        from openpi.policies import policy_config  # noqa: PLC0415
        from openpi.training import config  # noqa: PLC0415

        observation, noise = load_input(args.input)
        policy = policy_config.create_trained_policy(config.get_config(args.config_name), args.checkpoint)
        actions = policy.infer(observation, noise=noise)["actions"]
        jax.block_until_ready(actions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, actions)
        return

    if args.command == "triton":
        from policy.pi05.realtime_vla.accelerated_policy import RealtimeVlaPi05Policy  # noqa: PLC0415

        observation, noise = load_input(args.input)
        policy = RealtimeVlaPi05Policy(
            converted_checkpoint=args.checkpoint,
            norm_stats_path=args.norm_stats,
            tokenizer_path=args.tokenizer,
            realtime_vla_dir=args.realtime_vla_dir,
            prompt_for_allocation=str(observation["prompt"]),
        )
        actions = policy.infer(observation, noise=noise)["actions"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, actions)
        return

    jax_actions = np.load(args.jax_output)
    triton_actions = np.load(args.triton_output)
    absolute_error = np.abs(jax_actions - triton_actions)
    result = {
        "shape": list(jax_actions.shape),
        "mae": float(absolute_error.mean()),
        "max_absolute_error": float(absolute_error.max()),
        "per_dimension_mae": absolute_error.mean(axis=0).tolist(),
        "jax_range": [float(jax_actions.min()), float(jax_actions.max())],
        "triton_range": [float(triton_actions.min()), float(triton_actions.max())],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

