"""Paired open-loop precision study: JAX drives the simulator, Triton shadows it.

At every inference the JAX policy and the realtime-vla policy see the *same*
observation window and the *same* diffusion noise, so the difference between the
two 50-step chunks is purely the accelerator's numerical error. Only the JAX
chunk is executed, so the trajectory is the reference one and the comparison is
never contaminated by closed-loop divergence.

Run from the repository root with the same PYTHONPATH as scripts/eval_policy_parallel.py::

    python -m policy.pi05.realtime_vla.paired_precision --task click_bell \
        --train-config pi05_base_robosynchallenge_full --model-name click_bell_19999 \
        --checkpoint-id 19999 --checkpoint-root /data/realtime_vla/ckpt_root \
        --converted-checkpoint /data/realtime_vla/pi05_click_bell_19999.pkl \
        --prompt "Click the bell" --gpu-id 3 --episodes 5 --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--setting", default="random")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-id", type=int, required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--converted-checkpoint", required=True)
    parser.add_argument("--prompt", required=True, help="prompt_for_allocation for the Triton backend")
    parser.add_argument("--realtime-vla-dir", default=os.environ.get("REALTIME_VLA_DIR", str(REPO_ROOT.parent / "realtime-vla")))
    parser.add_argument("--tokenizer-path", default=str(Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="episode-seed rng, drawn like scripts/eval_policy.py")
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--pi0-step", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sys.argv = [
        "eval_policy.py", "--config", "policy/pi05/deploy_policy.yml", "--overrides",
        "--task_name", args.task, "--setting", args.setting, "--model_name", args.model_name,
        "--train_config_name", args.train_config, "--checkpoint_id", str(args.checkpoint_id),
        "--checkpoint_root", args.checkpoint_root, "--gpu_id", str(args.gpu_id), "--pytorch_device", "cuda",
        "--headless", "True", "--eval_video_log", "False", "--pi0_step", str(args.pi0_step),
    ]
    import eval_policy_parallel as ep  # noqa: PLC0415  (multi-GPU JAX pinning lives in the parallel evaluator)

    config = ep.parse_args_and_config()
    ep.select_cuda_device(config)
    pkg = ep.load_policy_adapter("pi05")
    import policy.pi05.deploy_policy as dp  # noqa: PLC0415

    gym_cfg = ep.find_gym_config(config)
    act_cfg = ep.find_action_config(config)
    max_env_steps, _, _ = ep.resolve_episode_max_steps(config, gym_cfg)
    env, gym_cfg = ep.make_env_from_configs(config, gym_cfg, act_cfg)
    instruction = ep.extract_instruction_from_gym_config(gym_cfg)

    model_jax = pkg.get_model({**config, "inference_backend": "jax", "pytorch_device": "cpu"})
    model_triton = pkg.get_model({
        **config, "inference_backend": "realtime_vla", "converted_checkpoint": args.converted_checkpoint,
        "realtime_vla_dir": args.realtime_vla_dir, "tokenizer_path": args.tokenizer_path,
        "prompt_for_allocation": args.prompt,
    })
    horizon = int(model_jax.action_horizon)
    action_dim = int(np.prod(env.unwrapped.single_action_space.shape))

    episode_rng = np.random.RandomState(args.seed)
    noise_rng = np.random.default_rng(args.noise_seed)
    records = []
    episodes = []
    for episode in range(args.episodes):
        seed = int(episode_rng.randint(0, 2**31 - 1))
        obs, info = env.reset(seed=seed)
        for model in (model_jax, model_triton):
            model.reset_obsrvationwindows()
            model.set_language(instruction)
        steps, success, truncated, n_infer = 0, False, False, 0
        t0 = time.time()
        while steps < max_env_steps:
            img_arr, state = dp.encode_obs(obs)
            model_jax.update_observation_window(img_arr, state)
            model_triton.update_observation_window(img_arr, state)
            noise = noise_rng.standard_normal((horizon, 32), dtype=np.float32)
            a_jax = np.asarray(model_jax.policy.infer(model_jax.observation_window, noise=noise)["actions"], dtype=np.float32)[:, :action_dim]
            a_tri = np.asarray(model_triton.policy.infer(model_triton.observation_window, noise=noise)["actions"], dtype=np.float32)[:, :action_dim]
            diff = np.abs(a_jax - a_tri)
            records.append({
                "episode": episode, "seed": seed, "env_step": steps,
                "mae": float(diff.mean()), "max": float(diff.max()),
                "mae_exec": float(diff[: args.pi0_step].mean()), "max_exec": float(diff[: args.pi0_step].max()),
                "per_dim_mae": diff.mean(axis=0).tolist(),
                "jax_std_per_dim": a_jax.std(axis=0).tolist(),
                "jax_range_per_dim": (a_jax.max(axis=0) - a_jax.min(axis=0)).tolist(),
            })
            n_infer += 1
            for action in a_jax[: args.pi0_step]:
                obs, _, _, trunc, info = env.step(dp._format_env_action(action, env))
                steps += 1
                if env.get_wrapper_attr("is_task_success")():
                    success = True
                    break
                if dp._any_true(trunc):
                    truncated = True
                    break
            if success or truncated:
                break
        episodes.append({"episode": episode, "seed": seed, "success": bool(success and not truncated), "steps": steps, "inferences": n_infer, "seconds": round(time.time() - t0, 1)})
        print(f"[{episode + 1}/{args.episodes}] seed {seed} {'success' if episodes[-1]['success'] else 'fail'} steps {steps} inferences {n_infer}", flush=True)

    per_dim = np.array([r["per_dim_mae"] for r in records])
    jax_std = np.array([r["jax_std_per_dim"] for r in records])
    maes = np.array([r["mae"] for r in records])
    maxes = np.array([r["max"] for r in records])
    summary = {
        "task": args.task, "setting": args.setting, "checkpoint_id": args.checkpoint_id,
        "episodes": len(episodes), "inferences": len(records), "horizon": horizon, "pi0_step": args.pi0_step,
        "mae_mean": float(maes.mean()), "mae_p95": float(np.percentile(maes, 95)), "mae_max": float(maes.max()),
        "max_abs_mean": float(maxes.mean()), "max_abs_p95": float(np.percentile(maxes, 95)), "max_abs_max": float(maxes.max()),
        "mae_exec_mean": float(np.mean([r["mae_exec"] for r in records])),
        "max_exec_max": float(np.max([r["max_exec"] for r in records])),
        "per_dim_mae": per_dim.mean(axis=0).tolist(),
        "jax_std_per_dim": jax_std.mean(axis=0).tolist(),
        "relative_mae_per_dim": (per_dim.mean(axis=0) / np.maximum(jax_std.mean(axis=0), 1e-6)).tolist(),
        "episode_results": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_dim_mae", "jax_std_per_dim", "relative_mae_per_dim", "episode_results")}, indent=1))


if __name__ == "__main__":
    main()
