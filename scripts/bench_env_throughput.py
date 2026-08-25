#!/usr/bin/env python
"""测 RoboSynChallenge 环境在 VLA 负载下的并行吞吐,用来定 total_num_envs。

为什么必须实测:RLinf 的 EmbodiChain 例子是无渲染的 CartPole,LIBERO 是轻量渲染 +
200 步 episode,两者都不能外推到这里——三路 640x480 相机 + 500 步 episode +
每 10 步随机光照。``total_num_envs`` 定不下来,global_batch_size / micro_batch_size
全是猜的。

跑的是真实训练负载:走 ``RoboSynChallengeVLAEnv``,所以包含 RGBA 切 RGB、
缩放到 224、以及每步调用官方 ``is_task_success()`` 的开销。动作用随机采样——
这里量的是环境侧吞吐,不是策略质量。

用法(在 RLinf 的 venv 里)::

    python scripts/bench_env_throughput.py --task mixer_operating --envs 1,4,8,16,32,64

输出每档的 env-steps/s、每步耗时和显存峰值。挑显存有余量(给 actor 和 rollout 留位置)
且 FPS 还在近似线性增长的那一档。
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch


def build_env(task: str, setting: str, num_envs: int, image_size: int, device: str):
    from robosynchallenge.rlinf_env import RoboSynChallengeVLAEnv

    repo_root = os.environ.get("ROBOSYN_PATH") or os.getcwd()
    cfg = {
        "gym_config_path": f"{repo_root}/configs/{task}/{setting}/gym_config.json",
        "headless": True,
        "sim_device": device,
        "seed": 0,
        "auto_reset": True,
        "ignore_terminations": False,
        # 只测吞吐,让 episode 跑满,不要因为成功提前终止而缩短样本
        "terminate_on_success": False,
        "max_episode_steps": 10_000,
        "image_size": image_size,
        "main_camera": "cam_high",
        "wrist_cameras": ["cam_left_wrist", "cam_right_wrist"],
    }
    return RoboSynChallengeVLAEnv(
        cfg=cfg, num_envs=num_envs, seed_offset=0, total_num_processes=1, worker_info=None
    )


def bench_one(task: str, setting: str, num_envs: int, steps: int, warmup: int,
              image_size: int, device: str) -> dict:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    env = build_env(task, setting, num_envs, image_size, device)
    try:
        env.reset(seed=0)
        action_dim = int(env.action_space.shape[-1])

        def random_actions():
            low = env.action_low.reshape(1, -1).expand(num_envs, -1)
            high = env.action_high.reshape(1, -1).expand(num_envs, -1)
            return low + (high - low) * torch.rand(
                (num_envs, action_dim), device=env.device
            )

        for _ in range(warmup):
            env.step(random_actions())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            env.step(random_actions())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        peak_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        )
        return {
            "num_envs": num_envs,
            "env_steps_per_s": num_envs * steps / elapsed,
            "ms_per_step": 1000.0 * elapsed / steps,
            "peak_gb": peak_gb,
        }
    finally:
        env.close()
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mixer_operating")
    ap.add_argument("--setting", default="random")
    ap.add_argument("--envs", default="1,4,8,16,32",
                    help="逗号分隔的 num_envs 档位")
    ap.add_argument("--steps", type=int, default=50, help="每档计时的步数")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import robosynchallenge  # noqa: F401  注册 task

    print(f"task={args.task}/{args.setting}  image_size={args.image_size}  device={args.device}")
    print(f"{'num_envs':>9}{'env-steps/s':>14}{'ms/step':>10}{'peak GB':>10}{'相对 1 env':>12}")
    print("-" * 55)

    baseline = None
    for token in args.envs.split(","):
        n = int(token.strip())
        if not n:
            continue
        try:
            r = bench_one(args.task, args.setting, n, args.steps, args.warmup,
                          args.image_size, args.device)
        except Exception as exc:  # OOM 或引擎拒绝,记下来继续测更小的档
            print(f"{n:>9}  失败: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        if baseline is None:
            baseline = r["env_steps_per_s"]
        speedup = r["env_steps_per_s"] / baseline if baseline else 1.0
        print(f"{r['num_envs']:>9}{r['env_steps_per_s']:>14.1f}"
              f"{r['ms_per_step']:>10.1f}{r['peak_gb']:>10.2f}{speedup:>11.1f}x")

    print("\n选档参考:取显存仍有余量(actor 和 rollout 也要占卡)且加速比还接近线性的那一档。")
    print("加速比明显走平说明渲染已经成为瓶颈,再加环境只会拖长每步耗时。")


if __name__ == "__main__":
    main()
