"""定位 4.29% IK 失败到底由解码链路的哪一步引入。

已知：直接拿 JSON 原始末端位姿做 IK -> 14880 次零失败。
所以问题在 encode(log) -> decode(exp) 这个往返里。

对比三种 rotm2aa 实现（其余完全相同）:
  A 简化版      —— 我重放脚本里用的，**没有 theta≈pi 分支**
  B mibot 版    —— 生产链路真正用的（convert 和 xr1_model 都走它），带 _axis_from_pi
同时统计相对旋转角 theta 的分布和解码后旋转矩阵的正交性误差。
"""

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xr1_fk import TCP

from embodichain.lab.sim.solvers import OPWSolverCfg
from embodichain.lab.sim.solvers.opw_solver import OPWSolver

HERE = os.path.dirname(os.path.abspath(__file__))
SIDES = (("left", slice(0, 6)), ("right", slice(7, 13)))
STRIDE, HORIZON = 30, 30

urdf = os.path.expanduser("~/.cache/embodichain_data/extract/CobotMagicArm/CobotMagicNoGripper.urdf")
cfg = OPWSolverCfg(urdf_path=urdf, end_link_name="link6", root_link_name="base_link", tcp=TCP.copy())
solver = OPWSolver(cfg, device="cpu")
solver.set_tcp(TCP.copy())


def aa2rotm(vector):
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    axis = vector / (angle + 1e-10)
    x, y, z = axis
    hat = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * hat + (1 - np.cos(angle)) * hat @ hat


def rotm2aa_simple(rotm):
    """重放脚本里的简化版：没有 theta≈pi 分支。"""
    theta = np.arccos(np.clip((np.trace(rotm) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([rotm[2, 1] - rotm[1, 2], rotm[0, 2] - rotm[2, 0], rotm[1, 0] - rotm[0, 1]])
    return axis / (np.linalg.norm(axis) + 1e-12) * theta


def rotm2aa_mibot(rotm):
    """生产链路用的版本（mibot.utils.io.rotm2aa_batch 的单条等价实现）。"""
    rotm32 = np.asarray(rotm, dtype=np.float32)
    theta = np.arccos(np.clip((np.trace(rotm32) - 1.0) / 2.0, -1.0, 1.0))
    if theta <= 1e-6:
        return np.zeros(3)
    if abs(theta - np.pi) <= 1e-6:
        r00, r11, r22 = rotm32[0, 0], rotm32[1, 1], rotm32[2, 2]
        if r00 >= r11 and r00 >= r22:
            vx = np.sqrt(max((r00 + 1) / 2, 0))
            vy, vz = (rotm32[0, 1] / (2 * vx), rotm32[0, 2] / (2 * vx)) if vx > 1e-8 else (0.0, 0.0)
            axis = np.array([vx, vy, vz])
        elif r11 >= r22:
            vy = np.sqrt(max((r11 + 1) / 2, 0))
            vx, vz = (rotm32[0, 1] / (2 * vy), rotm32[1, 2] / (2 * vy)) if vy > 1e-8 else (0.0, 0.0)
            axis = np.array([vx, vy, vz])
        else:
            vz = np.sqrt(max((r22 + 1) / 2, 0))
            vx, vy = (rotm32[0, 2] / (2 * vz), rotm32[1, 2] / (2 * vz)) if vz > 1e-8 else (0.0, 0.0)
            axis = np.array([vx, vy, vz])
        norm = np.linalg.norm(axis)
        axis = np.array([1.0, 0.0, 0.0]) if norm < 1e-12 else axis / norm
        return axis * theta
    axis = np.array([rotm32[2, 1] - rotm32[1, 2], rotm32[0, 2] - rotm32[2, 0], rotm32[1, 0] - rotm32[0, 1]])
    return axis / (np.linalg.norm(axis) + 1e-12) * theta


truth = np.load(os.path.join(HERE, "xr1_replay_truth.npz"))
files = sorted(glob.glob(os.path.join(HERE, "data", "episode_*.json")))

results = {}
thetas = []
ortho_errors = {"simple": [], "mibot": []}

for label, rotm2aa in (("simple", rotm2aa_simple), ("mibot", rotm2aa_mibot)):
    failures = 0
    calls = 0
    for path in files:
        key = os.path.basename(path).replace("episode_", "").replace(".json", "")
        with open(path) as handle:
            traj = json.load(handle)
        qpos_truth = truth[f"{key}_qpos"].astype(np.float64)
        num_frames = int(traj["num_frames"])

        for anchor in range(0, num_frames, STRIDE):
            steps = min(HORIZON, num_frames - anchor)
            for side, arm_slice in SIDES:
                anchor_position = np.asarray(traj["proprios"][f"{side}_ee_pos"][anchor], dtype=np.float64)
                anchor_rotation = np.asarray(traj["proprios"][f"{side}_ee_rotm"][anchor], dtype=np.float64).reshape(3, 3)
                target_positions = np.asarray(traj["actions"][f"{side}_ee_pos"][anchor:anchor + steps], dtype=np.float64)
                target_rotations = np.asarray(traj["actions"][f"{side}_ee_rotm"][anchor:anchor + steps], dtype=np.float64).reshape(-1, 3, 3)

                delta_position = (anchor_rotation.T @ (target_positions - anchor_position).T).T
                seed = torch.as_tensor(qpos_truth[anchor, arm_slice], dtype=torch.float32)[None]

                for step in range(steps):
                    relative = anchor_rotation.T @ target_rotations[step]
                    delta = rotm2aa(relative)
                    if label == "simple":
                        thetas.append(float(np.linalg.norm(delta)))
                    decoded = anchor_rotation @ aa2rotm(delta)
                    ortho_errors[label].append(
                        float(np.abs(decoded.T @ decoded - np.eye(3)).max())
                    )

                    target = np.eye(4, dtype=np.float32)
                    target[:3, :3] = decoded
                    target[:3, 3] = anchor_position + delta_position[step] @ anchor_rotation.T
                    code, solution = solver.get_ik(
                        target_xpos=torch.as_tensor(target, dtype=torch.float32)[None], qpos_seed=seed
                    )
                    calls += 1
                    if int(np.asarray(code.detach().cpu()).reshape(-1)[0]) == 1:
                        seed = solution.reshape(1, -1)[:, :6]
                    else:
                        failures += 1
    results[label] = (failures, calls)
    print(f"{label:<8} rotm2aa -> IK 失败 {failures}/{calls} = {100.0 * failures / calls:.4f}%")

thetas = np.asarray(thetas)
print(f"\n相对旋转角 theta 分布 (rad): median {np.median(thetas):.4f}  "
      f"p95 {np.percentile(thetas, 95):.4f}  max {thetas.max():.4f}")
print(f"  接近 pi (>3.0) 的占比: {100.0 * float(np.mean(thetas > 3.0)):.3f}%")
print(f"  接近 pi (>3.13) 的占比: {100.0 * float(np.mean(thetas > 3.13)):.3f}%")
for label in ("simple", "mibot"):
    errors = np.asarray(ortho_errors[label])
    print(f"  {label:<8} 解码后正交性误差 max: {errors.max():.3e}")
