"""EEF 转换产物的**解码级重放验证**。

链路（与部署实现 xr1_model.decode_action_eef_ik 逐步对齐）:
    转换后 JSON 的末端位姿  --上游同款相对化-->  EEF 增量
    --decode 数学-->  绝对末端目标序列
    --EmbodiChain OPW get_ik（种子链式）-->  重建 qpos
    --逐帧比对-->  原始数据集 action qpos

坐标系说明：全程在**臂基座系**做。robot.compute_ik 内部是
`solver.get_ik(inverse(base_pose) @ arena_pose)`，即先把 arena 位姿转回臂基座系
再交给同一个 OPW 解算器；base_pose 是常量刚体变换，在「锚点相对增量」下严格抵消
（validate_fk.py 实测 1.5e-15）。所以这里直接用 solver.get_ik，走的是与部署
完全相同的求解器代码路径，只是省掉了那层常量变换和仿真器。

锚点步长默认 30（整段 horizon），每帧只被重建一次，是最难的情形；
部署实际每 xr1_step=10 步就重新锚定一次，真实误差只会更小。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEFT_ARM = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_ARM = slice(7, 13)
RIGHT_GRIPPER = 13

SIDES = (
    ("left", LEFT_ARM, LEFT_GRIPPER, slice(0, 3), slice(3, 6), 6),
    ("right", RIGHT_ARM, RIGHT_GRIPPER, slice(8, 11), slice(11, 14), 14),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--truth", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "xr1_replay_truth.npz"))
    parser.add_argument("--stats", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "xr1_stats.json"))
    parser.add_argument("--stride", type=int, default=30, help="锚点步长，默认 30 = 整段 horizon（最难）")
    parser.add_argument("--action_length", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    from xr1_fk import TCP
    import torch
    from embodichain.lab.sim.solvers import OPWSolverCfg
    from embodichain.lab.sim.solvers.opw_solver import OPWSolver

    urdf = os.path.expanduser("~/.cache/embodichain_data/extract/CobotMagicArm/CobotMagicNoGripper.urdf")
    cfg = OPWSolverCfg(urdf_path=urdf, end_link_name="link6", root_link_name="base_link", tcp=TCP.copy())
    solver = OPWSolver(cfg, device="cpu")
    # 直接构造不会应用 cfg.tcp（只有走工厂方法才会），必须手动补，否则差一整个 TCP
    solver.set_tcp(TCP.copy())
    print(f"OPW 解算器就绪 (cpu), TCP 已施加, urdf={os.path.basename(urdf)}")

    with open(args.stats) as handle:
        stats = json.load(handle)
    print(f"stats: encoding={stats['encoding']} rot_scale={stats['rot_scale']} "
          f"gripper_range={[round(x, 4) for x in stats['gripper_range']]}")
    assert stats["encoding"] == "eef"

    truth = np.load(args.truth)
    files = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.json")))
    print(f"重放 {len(files)} 集，锚点步长 {args.stride}\n")

    def aa2rotm(vector):
        vector = np.asarray(vector, dtype=np.float64)
        angle = float(np.linalg.norm(vector))
        axis = vector / (angle + 1e-10)
        x, y, z = axis
        hat = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        return np.eye(3) + np.sin(angle) * hat + (1 - np.cos(angle)) * hat @ hat

    def rotm2aa(rotm):
        """与生产链路一致：mibot.utils.io.rotm2aa_batch 的单条等价实现。

        注意它**全程 float32**——这正是 IK 失败的来源（见 diag_decode_path.py）：
        本数据集的相对旋转角中位数只有 0.0012 rad，arccos 在 1.0 附近病态，
        float32 会把误差放大到约 1e-4 rad，解码后目标位姿偏差约 1e-5 m。
        不能为了好看换成 float64——那样重放就不代表真实链路了。
        """
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

    per_episode = []
    all_errors = []
    total_ik = 0
    total_fail = 0

    for path in files:
        key = os.path.basename(path).replace("episode_", "").replace(".json", "")
        with open(path) as handle:
            traj = json.load(handle)

        qpos_truth = truth[f"{key}_qpos"].astype(np.float64)
        action_truth = truth[f"{key}_action"].astype(np.float64)
        num_frames = int(traj["num_frames"])
        assert num_frames == len(qpos_truth), (num_frames, len(qpos_truth))

        rebuilt = np.array(action_truth, dtype=np.float64, copy=True)
        episode_ik = 0
        episode_fail = 0

        for anchor in range(0, num_frames, args.stride):
            steps = min(args.action_length, num_frames - anchor)

            for side, arm_slice, gripper_index, pos_slot, aa_slot, grip_slot in SIDES:
                # --- 上游同款相对化：从 JSON 的末端位姿算 EEF 增量
                anchor_position = np.asarray(traj["proprios"][f"{side}_ee_pos"][anchor], dtype=np.float64)
                anchor_rotation = np.asarray(traj["proprios"][f"{side}_ee_rotm"][anchor], dtype=np.float64).reshape(3, 3)
                target_positions = np.asarray(traj["actions"][f"{side}_ee_pos"][anchor:anchor + steps], dtype=np.float64)
                target_rotations = np.asarray(traj["actions"][f"{side}_ee_rotm"][anchor:anchor + steps], dtype=np.float64).reshape(-1, 3, 3)

                delta_position = (anchor_rotation.T @ (target_positions - anchor_position).T).T
                delta_rotation = np.stack(
                    [rotm2aa(anchor_rotation.T @ R) for R in target_rotations], axis=0
                )

                # --- decode 数学（与 xr1_model.decode_action_eef_ik 一致）
                decoded_positions = anchor_position[None] + delta_position @ anchor_rotation.T
                decoded_rotations = np.stack(
                    [anchor_rotation @ aa2rotm(d) for d in delta_rotation], axis=0
                )

                # --- IK 逐步解，种子链式
                seed = torch.as_tensor(qpos_truth[anchor, arm_slice], dtype=torch.float32)[None]
                for step in range(steps):
                    target = np.eye(4, dtype=np.float32)
                    target[:3, :3] = decoded_rotations[step]
                    target[:3, 3] = decoded_positions[step]
                    code, solution = solver.get_ik(
                        target_xpos=torch.as_tensor(target, dtype=torch.float32)[None],
                        qpos_seed=seed,
                    )
                    episode_ik += 1
                    accepted = int(np.asarray(code.detach().cpu()).reshape(-1)[0]) == 1
                    if accepted:
                        seed = solution.reshape(1, -1)[:, :6]
                    else:
                        episode_fail += 1
                    rebuilt[anchor + step, arm_slice] = seed[0].detach().cpu().numpy()

        # 夹爪是直通的（增量加回锚点），不经 IK
        error = np.abs(rebuilt - action_truth)
        arm_error = np.concatenate([error[:, LEFT_ARM], error[:, RIGHT_ARM]], axis=1)
        all_errors.append(arm_error)
        total_ik += episode_ik
        total_fail += episode_fail
        per_episode.append({
            "episode": key,
            "frames": num_frames,
            "median": float(np.median(arm_error)),
            "p95": float(np.percentile(arm_error, 95)),
            "max": float(arm_error.max()),
            "ik": episode_ik,
            "fail": episode_fail,
        })
        print(f"  ep{key}  帧{num_frames:4d}  median {per_episode[-1]['median']:.2e}  "
              f"p95 {per_episode[-1]['p95']:.2e}  max {per_episode[-1]['max']:.2e}  "
              f"IK {episode_fail}/{episode_ik}")

    errors = np.concatenate(all_errors, axis=0)
    print("\n" + "=" * 66)
    print("① 每关节重建误差 (rad)")
    print(f"{'关节':<14} {'median':>10} {'p95':>10} {'max':>10}")
    names = [f"left_j{i+1}" for i in range(6)] + [f"right_j{i+1}" for i in range(6)]
    for index, name in enumerate(names):
        column = errors[:, index]
        print(f"{name:<14} {np.median(column):>10.2e} {np.percentile(column, 95):>10.2e} {column.max():>10.2e}")
    print(f"{'全部':<14} {np.median(errors):>10.2e} {np.percentile(errors, 95):>10.2e} {errors.max():>10.2e}")

    print(f"\n② IK 失败率: {total_fail}/{total_ik} = {100.0 * total_fail / max(total_ik, 1):.4f}%")

    print("\n③ 最差 3 集")
    for row in sorted(per_episode, key=lambda r: -r["max"])[:3]:
        print(f"   ep{row['episode']}  max {row['max']:.3e}  p95 {row['p95']:.3e}  "
              f"IK 失败 {row['fail']}/{row['ik']}")

    passed = errors.max() < 0.01 and total_fail == 0
    print(f"\n④ 判定: {'PASS' if passed else 'FAIL'}  "
          f"(阈值 max<0.01 rad 且 IK 失败=0；实测 max={errors.max():.3e}, 失败={total_fail})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
