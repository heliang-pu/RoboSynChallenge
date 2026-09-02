"""Organizer-style isolated inference benchmark through the deployment adapter.

Mirrors the RoboSynChallenge released-checkpoint benchmark (evaluation_results/
released_checkpoint_results.json -> inference_benchmark): synthetic
deployment-shaped raw observations (three RGBA cameras [1, 480, 640, 4] uint8
and a [1, 14] state on the CPU), one policy job per GPU, and a synchronized
timing boundary covering observation lookup/preprocessing, CPU->GPU transfer,
policy inference, action validation and the action transfer to the env device.
``env.step`` is not involved. Default 20 warmup + 1000 measured calls (the DP
protocol).

Run from the repository root with the same PYTHONPATH as scripts/eval_policy.py::

    python -m policy.pi05.realtime_vla.benchmark_deploy --inference-backend jax \
        --train-config pi05_click_bell --model-name click_bell_29999 --checkpoint-id 29999 \
        --checkpoint-root /data/realtime_vla/ckpt_root --prompt "Click the bell" --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference-backend", choices=["jax", "realtime_vla"], required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--checkpoint-id", type=int, required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--converted-checkpoint", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--realtime-vla-dir", default=os.environ.get("REALTIME_VLA_DIR", str(REPO_ROOT.parent / "realtime-vla")))
    p.add_argument("--tokenizer-path", default=str(Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--calls", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-dim", type=int, default=14)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "policy"))
    import policy.pi05.deploy_policy as dp  # noqa: PLC0415
    from policy.inference_timing import finish_inference, start_inference  # noqa: PLC0415

    device = "cuda"
    usr_args = {
        "train_config_name": args.train_config, "model_name": args.model_name, "checkpoint_id": args.checkpoint_id,
        "checkpoint_root": args.checkpoint_root, "inference_backend": args.inference_backend,
        "converted_checkpoint": args.converted_checkpoint, "realtime_vla_dir": args.realtime_vla_dir,
        "tokenizer_path": args.tokenizer_path, "prompt_for_allocation": args.prompt,
        "pytorch_device": device, "pi0_step": 10,
    }
    model = dp.get_model(usr_args)
    model.set_language(args.prompt)

    # Deployment-shaped raw observation exactly as the env hands it to the adapter (CPU tensors).
    rng = np.random.default_rng(args.seed)
    def cam(): return torch.from_numpy(rng.integers(0, 256, (1, 480, 640, 4), dtype=np.uint8))
    obs = {"sensor": {"cam_high": {"color": cam()}, "cam_left_wrist": {"color": cam()}, "cam_right_wrist": {"color": cam()}},
           "robot": {"qpos": torch.from_numpy(rng.uniform(-1, 1, (1, args.action_dim)).astype(np.float32))}}
    fake_env = SimpleNamespace(unwrapped=SimpleNamespace(single_action_space=SimpleNamespace(shape=(args.action_dim,)), device=device))

    def one_call():
        t0 = start_inference(device)
        img_arr, state = dp.encode_obs(obs)              # observation lookup + preprocessing
        model.update_observation_window(img_arr, state)
        actions = model.get_action()[: model.pi0_step]   # CPU->GPU transfer + policy inference
        tensors = [dp._format_env_action(a, fake_env) for a in actions]  # action validation + transfer to env device
        finish_inference(t0, samples, device)
        return tensors

    samples: list[float] = []
    for _ in range(args.warmup):
        one_call()
    samples.clear()
    wall0 = time.perf_counter()
    for _ in range(args.calls):
        one_call()
    wall = time.perf_counter() - wall0
    s = sorted(samples)
    result = {
        "protocol": "organizer isolated benchmark: synthetic raw obs 3x[1,480,640,4] uint8 + state [1,14] on CPU; boundary = obs lookup/preprocess + H2D + inference + action validation + transfer to env device; env.step excluded",
        "inference_backend": args.inference_backend, "train_config": args.train_config, "model_name": args.model_name, "checkpoint_id": args.checkpoint_id,
        "prompt": args.prompt, "warmup_calls": args.warmup, "measured_calls": args.calls, "action_horizon": int(model.action_horizon), "executed_steps_per_call": int(model.pi0_step),
        "average_inference_time_seconds": statistics.fmean(samples), "median_inference_time_seconds": statistics.median(samples),
        "p95_inference_time_seconds": s[int(0.95 * len(s)) - 1], "p99_inference_time_seconds": s[int(0.99 * len(s)) - 1],
        "min_inference_time_seconds": s[0], "max_inference_time_seconds": s[-1], "wall_seconds": wall,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cpu_affinity_cores": len(os.sched_getaffinity(0)),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in result.items() if k != "protocol"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
