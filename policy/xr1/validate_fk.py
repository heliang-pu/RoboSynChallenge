"""FK 交叉验证：pytorch_kinematics 串链 vs EmbodiChain 的 OPW 解析解算器。

两套完全独立的实现（URDF 链式乘法 vs 解析 OPW 公式），都在臂基座系、都含 TCP。
取 sample_loading 真实帧，位置误差必须 < 1mm，否则坐标系/链定义有问题。
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASET = "/home/phl/workspace/datasets/cobotmagic_Sim_sample_loading"
NUM_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200

from xr1_fk import TCP, CobotMagicFK

# ---- 取真实关节角
import pandas as pd

files = sorted(glob.glob(os.path.join(DATASET, "data", "**", "*.parquet"), recursive=True))
print(f"数据集 parquet 文件数: {len(files)}")

frames = []
for path in files[:20]:
    table = pd.read_parquet(path, columns=["observation.state", "action"])
    frames.append(table)
    if sum(len(f) for f in frames) >= NUM_FRAMES * 3:
        break
table = pd.concat(frames, ignore_index=True)

rng = np.random.RandomState(0)
picks = rng.choice(len(table), size=min(NUM_FRAMES, len(table)), replace=False)
state = np.stack(table["observation.state"].values[picks]).astype(np.float64)
action = np.stack(table["action"].values[picks]).astype(np.float64)
print(f"抽样帧数: {len(state)}  (state 与 action 各一份)")

# 14 维 = [左臂6, 左夹爪1, 右臂6, 右夹爪1]
configs = {
    "left/state": state[:, 0:6],
    "right/state": state[:, 7:13],
    "left/action": action[:, 0:6],
    "right/action": action[:, 7:13],
}

# ---- A: pytorch_kinematics
fk = CobotMagicFK()
print(f"PK 串链: {fk.urdf_path}")
print(f"  关节顺序: {fk.joint_names()}")

# ---- B: EmbodiChain OPW 解析解算器
import torch

from embodichain.lab.sim.solvers import OPWSolverCfg
from embodichain.lab.sim.solvers.opw_solver import OPWSolver

# 左右臂几何完全相同（EmbodiChain 也是两边共用同一条 PK 串链），
# 单臂 URDF 里链名是通用的 base_link/link6，所以构造一个解算器即可。
cfg = OPWSolverCfg(
    urdf_path=fk.urdf_path,
    end_link_name="link6",
    root_link_name="base_link",
    tcp=TCP.copy(),
)
opw = OPWSolver(cfg, device="cpu")
# OPWSolver.__init__ 把 TCP 初始化成单位阵，cfg 里的 tcp 只有走 cfg 的工厂方法
# (OPWSolverCfg.build -> solver.set_tcp(...)) 才会生效。直接构造必须手动补这一步，
# 否则两边差整整一个 TCP（表现为恒定 143mm + 180deg 偏差）。
opw.set_tcp(TCP.copy())
opw_by_side = {"left": opw, "right": opw}
print(f"OPW 解算器构造完成 (device=cpu, urdf={os.path.basename(fk.urdf_path)})")

# ---- 比对
print()
print(f"{'配置':<16} {'位置误差 max/mean (mm)':<28} {'旋转误差 max/mean (deg)'}")
print("-" * 74)

worst_position = 0.0
worst_rotation = 0.0

for label, qpos in configs.items():
    side = label.split("/")[0]
    pk_matrix = fk.fk(qpos)
    opw_matrix = opw_by_side[side].get_fk(
        torch.as_tensor(qpos, dtype=torch.float32)
    )
    opw_matrix = np.asarray(
        opw_matrix.detach().cpu().numpy() if hasattr(opw_matrix, "detach") else opw_matrix,
        dtype=np.float64,
    ).reshape(-1, 4, 4)

    position_error = np.linalg.norm(pk_matrix[:, :3, 3] - opw_matrix[:, :3, 3], axis=1) * 1000.0

    relative = np.einsum("nji,njk->nik", pk_matrix[:, :3, :3], opw_matrix[:, :3, :3])
    trace = np.clip((np.einsum("nii->n", relative) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = np.degrees(np.arccos(trace))

    worst_position = max(worst_position, float(position_error.max()))
    worst_rotation = max(worst_rotation, float(rotation_error.max()))
    print(
        f"{label:<16} {position_error.max():8.4f} / {position_error.mean():<14.4f} "
        f"{rotation_error.max():8.4f} / {rotation_error.mean():.4f}"
    )

print()
print(f"位置误差上界: {worst_position:.4f} mm")
print(f"旋转误差上界: {worst_rotation:.4f} deg")
ok = worst_position < 1.0 and worst_rotation < 0.1
print("结论: " + ("两套独立实现一致，FK 可信" if ok else "**超阈值，坐标系/链定义有问题，停下排查**"))

# ---- 顺带确认「增量对左乘常量变换不变」这条性质（EEF 编码的理论基础）
print()
print("=== 增量的坐标系不变性自检 ===")
left = fk.fk(configs["left/state"][:50])
right = fk.fk(configs["left/action"][:50])
transform = np.eye(4)
angle = 0.7
transform[:3, :3] = np.array(
    [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
)
transform[:3, 3] = [1.23, -4.56, 7.89]

def relative_encode(anchor, target):
    rotation_anchor = anchor[:, :3, :3]
    position = np.einsum("nji,nj->ni", rotation_anchor, target[:, :3, 3] - anchor[:, :3, 3])
    rotation = np.einsum("nji,njk->nik", rotation_anchor, target[:, :3, :3])
    return position, rotation

base_position, base_rotation = relative_encode(left, right)
moved_position, moved_rotation = relative_encode(transform @ left, transform @ right)
print(f"  位置增量差异: {np.abs(base_position - moved_position).max():.3e}")
print(f"  旋转增量差异: {np.abs(base_rotation - moved_rotation).max():.3e}")
print("  -> 臂基座系 vs arena 系对训练数据等价" if
      np.abs(base_position - moved_position).max() < 1e-9 else "  -> 不变性不成立!")

sys.exit(0 if ok else 1)
