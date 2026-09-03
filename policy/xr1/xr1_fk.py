# ----------------------------------------------------------------------------
# CobotMagic 正向运动学 —— EEF 编码模式的基础设施
#
# 关键事实（都从 EmbodiChain 源码里核实过，不要凭直觉改）:
#   * EmbodiChain 给 CobotMagic 配的解算器是 OPWSolverCfg，
#     end_link_name = "{side}_link6"，root_link_name = "{side}_arm_base"
#     （不是 gripper_base，也不是整机 base_link）
#   * 它还带一个 TCP 变换: 绕 Z 转 180° + Z 向偏移 0.143 m，
#     get_fk() 返回的是**已经乘过 TCP** 的位姿，我们必须同样施加
#   * 用于 FK/IK 串链的 URDF 是单臂版 CobotMagicNoGripper.urdf，
#     里面链名是通用的 base_link -> link6，左右臂共用同一条链
#     （左右差异只是基座变换，见下面的坐标系说明）
#
# 坐标系（这是最容易翻车的地方）:
#   robot.compute_fk() 返回的是 **arena 系**（base_pose @ solver.get_fk()），
#   而本模块返回的是 **臂基座系**。两者差一个常量刚体变换 T。
#   但 XR-1 的动作编码是「相对锚点帧的增量」:
#       p_rel = R_aᵀ (p_t − p_a)      R_rel = R_aᵀ R_t
#   在整条 T 左乘下这两个量都是不变量（证明见 README EEF 章节），
#   所以训练数据用臂基座系算完全等价，无需知道 arena→base 的外参。
#   部署侧则全程用 EmbodiChain 的 arena 系（compute_fk → 加增量 → compute_ik），
#   自洽且同样不需要本模块。
# ----------------------------------------------------------------------------

from __future__ import annotations

import os

import numpy as np

DEFAULT_URDF = os.path.expanduser(
    "~/.cache/embodichain_data/extract/CobotMagicArm/CobotMagicNoGripper.urdf"
)

# EmbodiChain robots/cobotmagic.py 里 OPWSolverCfg 的 tcp，逐值照抄
TCP = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.143],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

ROOT_LINK = "base_link"
END_LINK = "link6"


class CobotMagicFK:
    """单臂 6 自由度 FK，输出臂基座系下的 4x4（已含 TCP）。"""

    def __init__(self, urdf_path=None, device="cpu", apply_tcp=True):
        import torch

        self.urdf_path = os.path.expanduser(urdf_path or DEFAULT_URDF)
        if not os.path.isfile(self.urdf_path):
            raise FileNotFoundError(f"找不到 URDF: {self.urdf_path}")

        self.device = device
        self.apply_tcp = apply_tcp
        self._torch = torch

        import pytorch_kinematics as pk

        with open(self.urdf_path, "rb") as handle:
            self.chain = pk.build_serial_chain_from_urdf(
                handle.read(), END_LINK, ROOT_LINK
            )
        # FK 是成百上千个小张量算子，torch 默认按核数开线程（112 核机器 = 112 线程），
        # 多进程并行转换时严重超订：实测一集(430 帧×4 次 FK) 默认线程 6.9 s，1 线程 0.02 s，差 350 倍。
        # 只影响本进程的 CPU intra-op 线程数；GPU 与其他进程不受影响。
        if device == "cpu":
            torch.set_num_threads(1)
        self.chain = self.chain.to(dtype=torch.float64, device=device)

        self.dof = len(self.chain.get_joint_parameter_names())
        if self.dof != 6:
            raise ValueError(f"期望 6 自由度串链，实际 {self.dof}: "
                             f"{self.chain.get_joint_parameter_names()}")
        self._tcp = torch.as_tensor(TCP, dtype=torch.float64, device=device)

    def joint_names(self):
        return list(self.chain.get_joint_parameter_names())

    def fk(self, qpos):
        """(N, 6) 关节角 -> (N, 4, 4) 齐次变换（臂基座系，含 TCP）。"""
        torch = self._torch
        array = np.asarray(qpos, dtype=np.float64).reshape(-1, 6)
        tensor = torch.as_tensor(array, dtype=torch.float64, device=self.device)
        matrix = self.chain.forward_kinematics(tensor).get_matrix()
        if self.apply_tcp:
            matrix = matrix @ self._tcp
        return matrix.detach().cpu().numpy()

    def fk_pos_rotm(self, qpos):
        """返回 (位置 (N,3), 旋转 (N,3,3))。"""
        matrix = self.fk(qpos)
        return matrix[:, :3, 3].copy(), matrix[:, :3, :3].copy()
