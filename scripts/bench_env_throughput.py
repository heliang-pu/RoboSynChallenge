#!/usr/bin/env python
"""测 RoboSynChallenge 环境在 VLA 负载下的并行吞吐,用来定 total_num_envs。

为什么必须实测:RLinf 的 EmbodiChain 例子是无渲染的 CartPole,LIBERO 是轻量渲染 +
200 步 episode,两者都不能外推到这里——三路 640x480 相机 + 500 步 episode +
每 10 步随机光照。``total_num_envs`` 定不下来,global_batch_size / micro_batch_size
全是猜的。

跑的是真实训练负载:走 ``RoboSynChallengeVLAEnv``,所以包含 RGBA 切 RGB、
缩放到 224、以及每步调用官方 ``is_task_success()`` 的开销。动作用随机采样——
这里量的是环境侧吞吐,不是策略质量。

**一次只测一档**。dexsim 引擎在 ``env.close()`` 时会让整个进程退出(现象是没有
traceback、退出码 0、后面的代码全不执行),所以同一进程里连测多档,第一次关闭就断了。
用 ``launch/rlinf_bench_envs.sh`` 循环调用本脚本,每档一个独立进程。

结果在关闭环境**之前**打印,原因同上。

用法::

    python scripts/bench_env_throughput.py --task mixer_operating --num-envs 8
"""

from __future__ import annotations

import argparse
import os
import time

import torch

# 让外层脚本能稳定解析,不受引擎日志刷屏干扰
RESULT_PREFIX = "BENCH_RESULT"


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="mixer_operating")
    ap.add_argument("--setting", default="random")
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--steps", type=int, default=30, help="计时步数")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import robosynchallenge  # noqa: F401  注册 task

    n = args.num_envs
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    env = build_env(args.task, args.setting, n, args.image_size, args.device)
    env.reset(seed=0)
    action_dim = int(env.action_space.shape[-1])

    def random_actions():
        low = env.action_low.reshape(1, -1).expand(n, -1)
        high = env.action_high.reshape(1, -1).expand(n, -1)
        return low + (high - low) * torch.rand((n, action_dim), device=env.device)

    for _ in range(args.warmup):
        env.step(random_actions())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        env.step(random_actions())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    fps = n * args.steps / elapsed
    ms = 1000.0 * elapsed / args.steps

    # 必须在 env.close() 之前打印:关闭时引擎会终止进程,之后的代码不会执行。
    print(f"{RESULT_PREFIX} num_envs={n} fps={fps:.2f} ms_per_step={ms:.2f} peak_gb={peak_gb:.3f}",
          flush=True)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
