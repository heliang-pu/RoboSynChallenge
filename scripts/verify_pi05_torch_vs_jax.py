#!/usr/bin/env python
"""验证转换后的 PyTorch pi0.5 与原 JAX checkpoint 数值一致。

给两个后端喂**完全相同**的 observation、prompt、flow-matching 初始噪声和归一化统计量，
比较输出的 action chunk。转换脚本靠路径字符串判断 pi05 分支(见
``scripts/convert_pi05_jax_to_torch.py``)，判断错了会静默产出一个结构合法但权重错位的
模型 —— 只有这一步能发现。

两个后端分开进程跑，避免 JAX 预分配显存和 PyTorch 抢卡::

    V=/home/phl/workspace/RoboSynChallenge/policy/pi05/.venv/bin/python
    $V scripts/verify_pi05_torch_vs_jax.py --backend jax \\
        --checkpoint-dir <orbax 的 28000 目录> --config-name pi05_base_robosynchallenge_full \\
        --norm-stats-dir <28000/assets/RoboSynChallenge/cobotmagic_Sim_mixer_operating> \\
        --out /tmp/jax.npz
    $V scripts/verify_pi05_torch_vs_jax.py --backend torch \\
        --checkpoint-dir <转换输出目录> --config-name pi05_base_robosynchallenge_full \\
        --norm-stats-dir <同上> --out /tmp/torch.npz
    $V scripts/verify_pi05_torch_vs_jax.py --compare /tmp/jax.npz /tmp/torch.npz

``--norm-stats-dir`` 是必要的: ``create_trained_policy`` 默认按 config 的 repo_id 找
assets 子目录，而 base config 的 repo_id 是 click_bell，多任务基座下各 task 的 checkpoint
assets 目录名却是各自的 task 名，对不上。显式传入也顺带保证两边归一化完全同源。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# action_dim 是模型内部的 padding 维度(Pi0Config.action_dim=32)，不是机器人的 14 维。
# 噪声要按模型维度给。
MODEL_ACTION_DIM = 32
DEFAULT_PROMPT = "Operate the mixer"


def build_obs(seed: int, state_dim: int, img_h: int, img_w: int) -> dict:
    """构造确定性的假 observation。键名同 policy/pi05/pi_model.py 的 observation_window。"""
    rng = np.random.default_rng(seed)

    def image() -> np.ndarray:
        return rng.integers(0, 256, size=(img_h, img_w, 3), dtype=np.uint8)

    return {
        "observation/image": image(),
        "observation/left_wrist_image": image(),
        "observation/right_wrist_image": image(),
        "observation/state": rng.standard_normal(state_dim).astype(np.float32),
        "prompt": DEFAULT_PROMPT,
    }


def run_backend(args: argparse.Namespace) -> None:
    import dataclasses

    from openpi.policies import policy_config as _policy_config
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    train_config = _config.get_config(args.config_name)
    if args.no_compile:
        # torch.compile 的 max-autotune 会启用 TF32 并挑选不同 triton kernel，
        # 本身就是一个数值差异源。做精度判别时必须关掉，否则分不清是权重错还是 kernel 差异。
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
        )
    norm_stats = _normalize.load(pathlib.Path(args.norm_stats_dir))

    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        norm_stats=norm_stats,
        pytorch_device=args.device if args.backend == "torch" else None,
    )

    obs = build_obs(args.seed, args.state_dim, args.img_h, args.img_w)
    noise = np.random.default_rng(args.noise_seed).standard_normal(
        (train_config.model.action_horizon, MODEL_ACTION_DIM)
    ).astype(np.float32)

    result = policy.infer(obs, noise=noise)
    actions = np.asarray(result["actions"], dtype=np.float32)

    # 输出 transform 会消费掉 state，只有 actions 一定在
    np.savez(args.out, actions=actions)
    print(f"[{args.backend}] actions shape={actions.shape} "
          f"min={actions.min():.6f} max={actions.max():.6f} mean={actions.mean():.6f}")
    print(f"[{args.backend}] 已写入 {args.out}")


def compare(path_a: str, path_b: str, atol: float) -> int:
    a = np.load(path_a)["actions"].astype(np.float64)
    b = np.load(path_b)["actions"].astype(np.float64)
    if a.shape != b.shape:
        print(f"形状不一致: {a.shape} vs {b.shape}")
        return 1

    diff = np.abs(a - b)
    scale = np.abs(a).mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float((a.ravel() @ b.ravel()) / denom) if denom > 0 else float("nan")

    print(f"shape         : {a.shape}")
    print(f"|a| 平均幅度  : {scale:.6f}")
    print(f"最大绝对误差  : {diff.max():.6e}")
    print(f"平均绝对误差  : {diff.mean():.6e}")
    print(f"相对误差(max/幅度): {diff.max() / scale:.4%}" if scale > 0 else "")
    print(f"余弦相似度    : {cos:.8f}")

    ok = diff.max() <= atol
    print(f"\n判定(atol={atol}): {'通过' if ok else '不通过'}")
    if not ok:
        # 逐时间步给出误差，便于判断是整体错位还是尾部发散
        per_step = diff.reshape(diff.shape[0], -1).max(axis=1)
        worst = np.argsort(per_step)[-5:][::-1]
        print("误差最大的 5 个时间步:", [(int(i), float(per_step[i])) for i in worst])
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"), help="对比两次运行的结果")
    ap.add_argument("--atol", type=float, default=2e-2, help="最大绝对误差容忍度(动作是关节空间量纲)")
    ap.add_argument("--backend", choices=["jax", "torch"])
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--config-name")
    ap.add_argument("--norm-stats-dir", help="含 norm_stats.json 的目录")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true",
                    help="关掉 torch.compile(max-autotune)——它会开 TF32，是独立的数值差异源")
    ap.add_argument("--seed", type=int, default=0, help="observation 随机种子")
    ap.add_argument("--noise-seed", type=int, default=1234, help="flow matching 初始噪声种子")
    ap.add_argument("--state-dim", type=int, default=14)
    ap.add_argument("--img-h", type=int, default=480)
    ap.add_argument("--img-w", type=int, default=640)
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(args.compare[0], args.compare[1], args.atol))

    missing = [n for n in ("backend", "checkpoint_dir", "config_name", "norm_stats_dir", "out")
               if getattr(args, n) is None]
    if missing:
        ap.error("缺少参数: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    run_backend(args)


if __name__ == "__main__":
    main()
