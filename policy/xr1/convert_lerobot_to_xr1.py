#!/usr/bin/env python
# ----------------------------------------------------------------------------
# LeRobot -> XR-1 后训练数据格式转换器
#
#   python policy/xr1/convert_lerobot_to_xr1.py \
#       --repo_dir lerobot_dataset/click_bell_aug_base/cobotmagic_Sim_click_the_bell_000 \
#       --out_dir  policy/xr1/training_data/click_bell \
#       --instruction "Click the bell"
#
# 产出:
#   <out_dir>/data/episode_XXXXXX.json   每集一个 XR-1 元数据 JSON
#   <out_dir>/videos/                    仅 --video_mode transcode 时生成
#   <out_dir>/xr1_stats.json             部署侧反归一化用的统计量
#   <out_dir>/xr1_data.yaml              可直接喂给 xr1/tools/train.py 的 hydra data 配置
#
# ---------------------------------------------------------------------------
# 动作打包方案（关键，务必先读 README_INTEGRATION.md）
#
# XR-1 的 60 维动作是末端位姿增量，本赛题只有 14 维绝对关节位置、没有末端位姿。
# 我们把每条手臂的 6 个关节塞进 XR-1 原本给 (ee_pos, ee_axis-angle) 的 3+3 个槽位，
# 并用指数映射保证严格可逆：
#
#   proprios.{arm}_ee_pos  [f] = q123[f]                      (关节 1..3 原样)
#   proprios.{arm}_ee_rotm [f] = exp(s * q456[f])             (关节 4..6 -> SO(3))
#   actions .{arm}_ee_pos  [f] = a123[f]
#   actions .{arm}_ee_rotm [f] = exp(s * a456[f])
#
# 于是上游 JsonDataset._arm_action 算出来的相对动作天然是:
#   dims[0:3] = R_a^T (a123[t] - q123[f])          R_a = exp(s * q456[f])
#   dims[3:6] = log(R_a^T exp(s * a456[t]))
#   dims[6]   = 夹爪增量
# 解码（xr1_model.XR1.decode_action）是它的严格逆运算，无任何近似。
#
# s = rot_scale 由数据自动定，保证 ||s * q456|| < pi，指数映射才是单射。
# ----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

# EmbodiChain 14 维 qpos: [左臂6, 左夹爪1, 右臂6, 右夹爪1]
LEFT_ARM = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_ARM = slice(7, 13)
RIGHT_GRIPPER = 13

DEFAULT_ACTION_LENGTH = 30
ACTION_DIM = 60
STATE_DIM = 60

# 指数映射的单射范围是 ||v|| < pi；留安全余量避免 log 在 pi 附近数值退化
ROT_NORM_BUDGET = 2.8

VIDEO_KEYS = {
    "ego": "observation.images.cam_high",
    "wrist_left": "observation.images.cam_left_wrist",
    "wrist_right": "observation.images.cam_right_wrist",
}

STATE_FEATURE_CANDIDATES = ("observation.state", "observation.qpos")
ACTION_FEATURE = "action"


# --------------------------------------------------------------------------
# 旋转工具：与 mibot.utils.io 的实现严格等价（见 --self_test）
# --------------------------------------------------------------------------


def aa2rotm_batch(axis_angle):
    """(N, 3) 轴角 -> (N, 3, 3) 旋转矩阵，Rodrigues 公式。"""
    axis_angle = np.asarray(axis_angle, dtype=np.float64).reshape(-1, 3)
    angle = np.linalg.norm(axis_angle, axis=1)
    axis = axis_angle / (angle[:, None] + 1e-10)

    zero = np.zeros_like(angle)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    hat = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=1
    ).reshape(-1, 3, 3)

    eye = np.broadcast_to(np.identity(3), (len(axis_angle), 3, 3))
    sin = np.sin(angle)[:, None, None]
    cos = np.cos(angle)[:, None, None]
    return eye + sin * hat + (1.0 - cos) * (hat @ hat)


def rotm2aa_batch(rotms):
    """(N, 3, 3) -> (N, 3)。逐行照抄 mibot.utils.io.rotm2aa_batch 的分支逻辑。"""
    rotms = np.asarray(rotms, dtype=np.float32)
    theta = np.arccos(np.clip((np.einsum("nii->n", rotms) - 1.0) / 2.0, -1.0, 1.0))

    axis_angle = np.zeros((rotms.shape[0], 3))
    near_zero = theta <= 1e-6
    near_pi = np.abs(theta - np.pi) <= 1e-6
    normal = ~(near_zero | near_pi)

    if np.any(normal):
        axis = np.stack(
            [
                rotms[:, 2, 1] - rotms[:, 1, 2],
                rotms[:, 0, 2] - rotms[:, 2, 0],
                rotms[:, 1, 0] - rotms[:, 0, 1],
            ],
            axis=1,
        )
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        axis_angle[normal] = axis[normal] * theta[normal, None]

    if np.any(near_pi):
        from_pi = _axis_from_pi_batch(rotms[near_pi])
        axis_angle[near_pi] = from_pi * theta[near_pi, None]

    return axis_angle


def _axis_from_pi_batch(rotms):
    out = np.zeros((len(rotms), 3))
    for index, rotm in enumerate(rotms):
        rot00, rot11, rot22 = rotm[0, 0], rotm[1, 1], rotm[2, 2]
        if rot00 >= rot11 and rot00 >= rot22:
            vx = np.sqrt(max((rot00 + 1) / 2, 0))
            vy, vz = (rotm[0, 1] / (2 * vx), rotm[0, 2] / (2 * vx)) if vx > 1e-8 else (0.0, 0.0)
            axis = np.array([vx, vy, vz])
        elif rot11 >= rot22:
            vy = np.sqrt(max((rot11 + 1) / 2, 0))
            vx, vz = (rotm[0, 1] / (2 * vy), rotm[1, 2] / (2 * vy)) if vy > 1e-8 else (0.0, 0.0)
            axis = np.array([vx, vy, vz])
        else:
            vz = np.sqrt(max((rot22 + 1) / 2, 0))
            vx, vy = (rotm[0, 2] / (2 * vz), rotm[1, 2] / (2 * vz)) if vz > 1e-8 else (0.0, 0.0)
            axis = np.array([vx, vy, vz])
        norm = np.linalg.norm(axis)
        out[index] = np.array([1.0, 0.0, 0.0]) if norm < 1e-12 else axis / norm
    return out


# --------------------------------------------------------------------------
# 打包 / 统计
# --------------------------------------------------------------------------


def compose_state_batch(qpos):
    """(N, 14) qpos -> (N, 60) XR-1 打包状态。等价于 mibot.utils.io.compose_state。"""
    qpos = np.asarray(qpos, dtype=np.float32)
    state = np.zeros((len(qpos), STATE_DIM), dtype=np.float32)
    state[:, 0:6] = qpos[:, LEFT_ARM]        # 左臂 6 关节（第 7 关节位留 0）
    state[:, 7] = qpos[:, LEFT_GRIPPER]
    state[:, 8:14] = qpos[:, RIGHT_ARM]      # 右臂 6 关节
    state[:, 15] = qpos[:, RIGHT_GRIPPER]
    return state


def episode_ee_frames(qpos, action, encoding, rot_scale, fk=None):
    """两种编码的统一出口：每帧的「末端」位姿（锚点侧来自 state，目标侧来自 action）。

    slot 模式（槽位复用）：把关节 1-3 当位置、关节 4-6 经指数映射当旋转，是纯粹的
        维度搬运，与真实几何无关。
    eef 模式（真实末端）：走 CobotMagicFK，得到臂基座系下的真实末端位姿（含 TCP），
        语义与 XR-1 预训练时一致。

    返回 dict: side -> (锚点位置 (N,3), 锚点旋转 (N,3,3), 目标位置, 目标旋转)
    """
    frames = {}
    for side, arm_slice in (("left", LEFT_ARM), ("right", RIGHT_ARM)):
        arm_qpos = np.asarray(qpos, dtype=np.float64)[:, arm_slice]
        arm_action = np.asarray(action, dtype=np.float64)[:, arm_slice]
        if encoding == "eef":
            if fk is None:
                raise ValueError("eef 编码需要传入 CobotMagicFK 实例")
            anchor_position, anchor_rotation = fk.fk_pos_rotm(arm_qpos)
            target_position, target_rotation = fk.fk_pos_rotm(arm_action)
        else:
            anchor_position = arm_qpos[:, 0:3]
            anchor_rotation = aa2rotm_batch(rot_scale * arm_qpos[:, 3:6])
            target_position = arm_action[:, 0:3]
            target_rotation = aa2rotm_batch(rot_scale * arm_action[:, 3:6])
        frames[side] = (anchor_position, anchor_rotation, target_position, target_rotation)
    return frames


def pack_relative_actions(qpos, action, rot_scale, action_length=DEFAULT_ACTION_LENGTH,
                          encoding="slot", fk=None):
    """向量化复刻 JsonDataset._arm_action + compose_action。

    返回 (N, action_length, 60) 的未归一化打包相对动作，其中第 f 行对应
    “以第 f 帧为锚点、未来 action_length 步”的监督目标（尾部按上游语义重复末帧）。
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    num_frames = len(qpos)

    # 上游 _future + _pad 的效果：越界索引一律钳到最后一帧
    horizon = np.minimum(
        np.arange(num_frames)[:, None] + np.arange(action_length)[None, :],
        num_frames - 1,
    )

    packed = np.zeros((num_frames, action_length, ACTION_DIM), dtype=np.float32)

    sides = (
        ("left", LEFT_ARM, LEFT_GRIPPER, slice(0, 3), slice(3, 6), 6),
        ("right", RIGHT_ARM, RIGHT_GRIPPER, slice(8, 11), slice(11, 14), 14),
    )
    ee_frames = episode_ee_frames(qpos, action, encoding, rot_scale, fk)

    for side, arm_slice, gripper_index, pos_slot, aa_slot, grip_slot in sides:
        anchor_position, anchor_rotation, target_position, target_rotation = ee_frames[side]

        # 位置槽: R_a^T (p_t[t] - p_a[f])
        delta = target_position[horizon] - anchor_position[:, None, :]
        packed[:, :, pos_slot] = np.einsum(
            "fji,ftj->fti", anchor_rotation, delta
        ).astype(np.float32)

        # 旋转槽: log(R_a^T R_t)
        relative = np.einsum(
            "fji,ftjk->ftik", anchor_rotation, target_rotation[horizon]
        )
        packed[:, :, aa_slot] = rotm2aa_batch(
            relative.reshape(-1, 3, 3)
        ).reshape(num_frames, action_length, 3).astype(np.float32)

        # 夹爪槽: 绝对增量
        packed[:, :, grip_slot] = (
            action[:, gripper_index][horizon] - qpos[:, gripper_index][:, None]
        )

    # waist(16) / base(17:20) 本机器人没有，保持 0
    return packed


class StatsAccumulator:
    """按步累计打包动作的 mean/std，并收集打包状态用于算 q01/q99。

    注意只统计“真实存在”的步：上游 JsonDataset 对超出集尾的步做重复末帧填充，
    但 action_mask 把它们排除在损失之外，所以归一化统计也必须排除，否则
    短集的末帧会被重复计入、把 mean/std 拉偏。
    """

    def __init__(self, action_length=DEFAULT_ACTION_LENGTH, max_state_rows=2_000_000):
        self.action_length = action_length
        self.count = np.zeros(action_length, dtype=np.int64)
        self.total = np.zeros((action_length, ACTION_DIM), dtype=np.float64)
        self.total_sq = np.zeros((action_length, ACTION_DIM), dtype=np.float64)
        self.states = []
        self.state_rows = 0
        self.max_state_rows = max_state_rows

    @property
    def num_samples(self):
        return int(self.count[0])

    def add(self, packed, states):
        num_frames = len(packed)
        # valid[f, t] == (t < min(action_length, num_frames - f))
        valid = (
            np.arange(self.action_length)[None, :]
            < (num_frames - np.arange(num_frames))[:, None]
        )
        weights = valid.astype(np.float64)[:, :, None]

        self.count += valid.sum(axis=0)
        self.total += (packed * weights).sum(axis=0)
        self.total_sq += (np.square(packed, dtype=np.float64) * weights).sum(axis=0)

        if self.state_rows < self.max_state_rows:
            self.states.append(np.asarray(states, dtype=np.float32))
            self.state_rows += len(states)

    def finalize(self, std_floor=1e-4):
        if self.count[0] == 0:
            raise ValueError("没有累计到任何样本")
        if np.any(self.count == 0):
            raise ValueError(
                f"动作 horizon={self.action_length} 超过了所有轨迹的长度，"
                f"最短步位只有 {int(self.count.min())} 个样本"
            )

        denominator = self.count[:, None].astype(np.float64)
        mean64 = self.total / denominator
        mean = mean64.astype(np.float32)
        variance = np.maximum(self.total_sq / denominator - np.square(mean64), 0.0)
        std = np.sqrt(variance).astype(np.float32)

        # 有监督的槽位里，std 太小会让归一化目标炸掉；恒为 0 的维度保持 0（例如夹爪从不动）
        supervised = np.zeros(ACTION_DIM, dtype=bool)
        for slot in (slice(0, 7), slice(8, 15), slice(16, 20)):
            supervised[slot] = True
        tiny = (std < std_floor) & (std > 0.0) & supervised[None, :]
        std[tiny] = std_floor

        states = np.concatenate(self.states, axis=0)
        q01 = np.percentile(states, 1, axis=0).astype(np.float32).reshape(1, STATE_DIM)
        q99 = np.percentile(states, 99, axis=0).astype(np.float32).reshape(1, STATE_DIM)

        # validate_quantiles 要求：非零填充维必须 q99 > q01。
        # 常量维（例如夹爪全程 0）会触发这条断言，统一压成 0 让它走 padding 分支。
        degenerate = (q99 - q01) < 1e-8
        q01[degenerate] = 0.0
        q99[degenerate] = 0.0
        if np.any(degenerate):
            indices = np.where(degenerate[0])[0]
            informative = [int(i) for i in indices if i in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15)]
            if informative:
                print(f"[stats] 提示: 状态维 {informative} 在数据集中恒定，已按 padding 处理（归一化为 0）")

        return mean, std, q01, q99


# --------------------------------------------------------------------------
# LeRobot 读取（v3.0 与 v2.1 两种布局）
# --------------------------------------------------------------------------


def _read_parquet(path):
    import pandas as pd

    return pd.read_parquet(path)


def _stack(series):
    return np.stack([np.asarray(value, dtype=np.float32) for value in series], axis=0)


def find_dataset_root(repo_dir):
    """允许直接给数据集根，也允许给它的父目录（自动找唯一的子数据集）。"""
    if os.path.isfile(os.path.join(repo_dir, "meta", "info.json")):
        return repo_dir
    candidates = sorted(glob.glob(os.path.join(repo_dir, "*", "meta", "info.json")))
    if len(candidates) == 1:
        return os.path.dirname(os.path.dirname(candidates[0]))
    if not candidates:
        raise FileNotFoundError(f"{repo_dir} 下找不到 meta/info.json")
    raise ValueError(
        f"{repo_dir} 下有多个数据集，请明确指定其中一个:\n  "
        + "\n  ".join(os.path.dirname(os.path.dirname(c)) for c in candidates)
    )


class LeRobotReader:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "meta", "info.json"), "r") as handle:
            self.info = json.load(handle)
        self.fps = float(self.info.get("fps", 25))
        self.version = str(self.info.get("codebase_version", "v3.0"))
        self.is_v3 = not self.version.startswith("v2")

        features = self.info.get("features", {})
        self.state_key = next(
            (key for key in STATE_FEATURE_CANDIDATES if key in features), None
        )
        if self.state_key is None:
            raise KeyError(
                f"数据集缺少状态特征，已尝试 {STATE_FEATURE_CANDIDATES}；实际有 {sorted(features)}"
            )
        if ACTION_FEATURE not in features:
            raise KeyError(f"数据集缺少 '{ACTION_FEATURE}' 特征")

        state_dim = int(features[self.state_key]["shape"][0])
        action_dim = int(features[ACTION_FEATURE]["shape"][0])
        if state_dim != 14 or action_dim != 14:
            raise ValueError(
                f"本转换器只处理 14 维双臂数据，实际 state={state_dim} action={action_dim}"
            )

        self.episodes = self._read_episode_index()

    # ---- episode 索引

    def _read_episode_index(self):
        if self.is_v3:
            files = sorted(
                glob.glob(os.path.join(self.root, "meta", "episodes", "**", "*.parquet"), recursive=True)
            )
            if not files:
                raise FileNotFoundError("v3 数据集缺少 meta/episodes/**/*.parquet")
            import pandas as pd

            return pd.concat([_read_parquet(path) for path in files], ignore_index=True)

        # v2.1: meta/episodes.jsonl
        path = os.path.join(self.root, "meta", "episodes.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"v2 数据集缺少 {path}")
        import pandas as pd

        with open(path, "r") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return pd.DataFrame(rows)

    def __len__(self):
        return len(self.episodes)

    # ---- 单集读取

    def episode(self, index):
        row = self.episodes.iloc[index]
        episode_index = int(row["episode_index"])

        if self.is_v3:
            data_path = os.path.join(
                self.root,
                self.info["data_path"].format(
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                ),
            )
            frame = _read_parquet(data_path)
            start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            # 同一个 parquet 装了多集，按全局行号切片
            if "index" in frame.columns:
                frame = frame[(frame["index"] >= start) & (frame["index"] < stop)]
            else:
                frame = frame.iloc[start:stop]
        else:
            data_path = os.path.join(
                self.root,
                self.info["data_path"].format(
                    episode_chunk=episode_index // int(self.info.get("chunks_size", 1000)),
                    episode_index=episode_index,
                ),
            )
            frame = _read_parquet(data_path)

        frame = frame.sort_values("frame_index") if "frame_index" in frame.columns else frame
        qpos = _stack(frame[self.state_key].values)
        action = _stack(frame[ACTION_FEATURE].values)

        tasks = row.get("tasks")
        if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks):
            task = str(tasks[0])
        else:
            task = str(tasks) if tasks is not None else None

        return {
            "episode_index": episode_index,
            "qpos": qpos,
            "action": action,
            "task": task,
            "videos": self._video_sources(row, len(qpos)),
        }

    def _video_sources(self, row, num_frames):
        sources = {}
        for name, key in VIDEO_KEYS.items():
            if self.is_v3:
                path = os.path.join(
                    self.root,
                    self.info["video_path"].format(
                        video_key=key,
                        chunk_index=int(row[f"videos/{key}/chunk_index"]),
                        file_index=int(row[f"videos/{key}/file_index"]),
                    ),
                )
                from_timestamp = float(row[f"videos/{key}/from_timestamp"])
                to_timestamp = float(row[f"videos/{key}/to_timestamp"])
                start_frame = int(round(from_timestamp * self.fps))
            else:
                episode_index = int(row["episode_index"])
                path = os.path.join(
                    self.root,
                    self.info["video_path"].format(
                        episode_chunk=episode_index // int(self.info.get("chunks_size", 1000)),
                        video_key=key,
                        episode_index=episode_index,
                    ),
                )
                from_timestamp = 0.0
                to_timestamp = num_frames / self.fps
                start_frame = 0

            if not os.path.isfile(path):
                raise FileNotFoundError(f"找不到视频文件: {path}")
            sources[name] = {
                "path": os.path.abspath(path),
                "start": start_frame,
                "from_timestamp": from_timestamp,
                "to_timestamp": to_timestamp,
            }
        return sources


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


def count_video_frames(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {path}\n{result.stderr}")
    return int(result.stdout.strip().split(",")[0])


def transcode_video(source, destination, expected_frames=None):
    """整段转码成 h264，帧序号 1:1 保留。

    本赛题的 LeRobot 视频是 AV1，decord 0.6.0 打不开（'cannot find video stream'）。
    这里刻意不按集裁剪：v3 数据把整个 chunk 的所有集拼在一个文件里，整段转一次
    既避免了按时间戳 seek 的对齐风险，也比每集切一刀快得多；各集仍然用
    start 帧偏移去索引。
    """
    if os.path.isfile(destination):
        return destination

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".tmp.mp4"
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source,
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
        temporary,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {' '.join(command)}\n{result.stderr}")

    if expected_frames is not None:
        actual = count_video_frames(temporary)
        if actual != expected_frames:
            os.remove(temporary)
            raise RuntimeError(
                f"转码后帧数不符: {source} 期望 {expected_frames} 帧，实际 {actual} 帧。"
                "帧号与动作序列会错位，已中止。"
            )

    os.replace(temporary, destination)
    return destination


def build_episode_json(episode, instruction, rot_scale, video_entries, encoding="slot", fk=None):
    qpos = episode["qpos"]
    action = episode["action"]
    num_frames = len(qpos)

    payload = {
        "trajectory_type": "success",
        "time": f"episode_{episode['episode_index']:06d}",
        "num_frames": int(num_frames),
        "instruction": {
            "general": [
                {
                    "images": [
                        "observations.ego",
                        "observations.wrist_left",
                        "observations.wrist_right",
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "The following observations are captured from multiple views.\n"
                                "# Ego View\n<image>\n# Left-Wrist View\n<image>\n"
                                "# Right-Wrist View\n<image>\n"
                                f"Generate robot actions for the task:\n{instruction}"
                            ),
                        },
                        {"from": "gpt", "value": ""},
                    ],
                }
            ]
        },
        "observations": {name: [video_entries[name]] for name in ("ego", "wrist_left", "wrist_right")},
        "proprios": {},
        "actions": {},
    }

    zeros_1 = np.zeros((num_frames, 1), dtype=np.float32)
    ee_frames = episode_ee_frames(qpos, action, encoding, rot_scale, fk)
    for side, arm_slice, gripper_index in (
        ("left", LEFT_ARM, LEFT_GRIPPER),
        ("right", RIGHT_ARM, RIGHT_GRIPPER),
    ):
        arm_qpos = qpos[:, arm_slice]
        anchor_position, anchor_rotation, target_position, target_rotation = ee_frames[side]

        payload["proprios"][f"{side}_ee_pos"] = anchor_position.tolist()
        payload["proprios"][f"{side}_ee_rotm"] = anchor_rotation.reshape(num_frames, 9).tolist()
        # 状态侧永远是纯关节打包（compose_state），与 encoding 无关
        payload["proprios"][f"{side}_arm_joint"] = arm_qpos.tolist()
        payload["proprios"][f"{side}_gripper_pos"] = qpos[:, gripper_index : gripper_index + 1].tolist()

        payload["actions"][f"{side}_ee_pos"] = target_position.tolist()
        payload["actions"][f"{side}_ee_rotm"] = target_rotation.reshape(num_frames, 9).tolist()
        payload["actions"][f"{side}_gripper_pos"] = action[
            :, gripper_index : gripper_index + 1
        ].tolist()

    payload["proprios"]["waist_pos"] = zeros_1.tolist()
    payload["actions"]["waist_pos"] = zeros_1.tolist()
    payload["actions"]["base_vel"] = np.zeros((num_frames, 3), dtype=np.float32).tolist()
    return payload


def _format_rows(rows, indent="      "):
    lines = []
    for row in rows:
        values = ", ".join(f"{float(value):.6g}" for value in row)
        lines.append(f"{indent}- [{values}]")
    return "\n".join(lines)


def write_data_yaml(path, data_dir, mean, std, q01, q99, rot_scale, action_length, batch_size,
                    encoding="eef", gripper_range=None):
    gripper_range = list(gripper_range) if gripper_range else [0.0, 1.0]
    content = f"""# @package _global_
# 由 convert_lerobot_to_xr1.py 自动生成，勿手改。
# rot_scale 是本适配层的自定义字段（上游 JsonDataset 会忽略未知键），
# 部署时 xr1_model.py 依赖它把旋转槽位还原成关节角。

data:
  type: BaseDataModule
  params:
    type: json
    max_steps: ${{trainer.max_steps}}
    train_datasets:
      batch_size: {batch_size}
      action_length: {action_length}
      encoding: {encoding}
      gripper_range: {gripper_range}
      rot_scale: {rot_scale:.10g}
      paths:
      - {data_dir}
      mean:
{_format_rows(mean)}
      std:
{_format_rows(std)}
      q01:
{_format_rows(q01)}
      q99:
{_format_rows(q99)}
"""
    with open(path, "w") as handle:
        handle.write(content)


# --------------------------------------------------------------------------


def self_test():
    """验证本文件的旋转工具与 mibot 一致，且编解码严格互逆。"""
    rng = np.random.RandomState(0)
    ok = True

    vectors = rng.uniform(-1.0, 1.0, size=(64, 3))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True) * rng.uniform(
        0.0, 2.8, size=(64, 1)
    )

    # 1) exp/log 互逆
    recovered = rotm2aa_batch(aa2rotm_batch(vectors))
    error = np.abs(recovered - vectors).max()
    print(f"[self_test] exp/log 往返最大误差: {error:.3e}")
    ok &= error < 1e-5

    # 2) 与 mibot 的实现一致
    try:
        from mibot.utils.io import aa2rotm as mibot_aa2rotm
        from mibot.utils.io import rotm2aa_batch as mibot_rotm2aa

        mine = aa2rotm_batch(vectors)
        theirs = np.stack([mibot_aa2rotm(v) for v in vectors], axis=0)
        error = np.abs(mine - theirs).max()
        print(f"[self_test] aa2rotm 与 mibot 差异: {error:.3e}")
        ok &= error < 1e-5

        error = np.abs(rotm2aa_batch(mine) - mibot_rotm2aa(mine)).max()
        print(f"[self_test] rotm2aa 与 mibot 差异: {error:.3e}")
        ok &= error < 1e-5
    except ImportError:
        print("[self_test] 跳过 mibot 对比（当前解释器里没有 mibot）")

    # 3) 打包 -> 解码 往返还原绝对 qpos
    num_frames = 40
    qpos = rng.uniform(-0.6, 0.6, size=(num_frames, 14)).astype(np.float32)
    action = (qpos + rng.uniform(-0.05, 0.05, size=(num_frames, 14))).astype(np.float32)
    scale = 1.0
    packed = pack_relative_actions(qpos, action, scale, action_length=DEFAULT_ACTION_LENGTH)

    decoded = decode_packed(packed[0], qpos[0], scale)
    horizon = np.minimum(np.arange(DEFAULT_ACTION_LENGTH), num_frames - 1)
    error = np.abs(decoded - action[horizon]).max()
    print(f"[self_test] 打包/解码往返最大误差: {error:.3e}")
    ok &= error < 1e-4

    print("[self_test] " + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


def decode_packed(packed, anchor_qpos, rot_scale):
    """xr1_model.XR1.decode_action 的独立副本，供 self_test 用（不依赖 torch）。"""
    packed = np.asarray(packed, dtype=np.float32)
    anchor = np.asarray(anchor_qpos, dtype=np.float32).reshape(-1)
    steps = len(packed)
    targets = np.tile(anchor[None], (steps, 1)).astype(np.float32)

    for arm_slice, gripper_index, pos_slot, aa_slot, grip_slot in (
        (LEFT_ARM, LEFT_GRIPPER, slice(0, 3), slice(3, 6), 6),
        (RIGHT_ARM, RIGHT_GRIPPER, slice(8, 11), slice(11, 14), 14),
    ):
        arm_anchor = anchor[arm_slice]
        rotation_anchor = aa2rotm_batch(rot_scale * arm_anchor[None, 3:6])[0]

        targets[:, arm_slice.start : arm_slice.start + 3] = (
            arm_anchor[None, 0:3] + packed[:, pos_slot] @ rotation_anchor.T
        )
        rotations = rotation_anchor[None] @ aa2rotm_batch(packed[:, aa_slot])
        targets[:, arm_slice.start + 3 : arm_slice.start + 6] = rotm2aa_batch(rotations) / rot_scale
        targets[:, gripper_index] = anchor[gripper_index] + packed[:, grip_slot]

    return targets


def parse_args():
    parser = argparse.ArgumentParser(
        description="LeRobot -> XR-1 后训练数据转换器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_dir", help="LeRobot 数据集根目录（含 meta/info.json）")
    parser.add_argument("--out_dir", help="输出目录")
    parser.add_argument("--instruction", default=None, help="语言指令，默认取数据集里的 task")
    parser.add_argument("--action_length", type=int, default=DEFAULT_ACTION_LENGTH)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "写进 data yaml 的 batch_size。实测单样本约 484 token，默认 MAX_LENGTH=4096 "
            "只装得下 8 条；CustomCollate 会把超出的样本直接丢掉且不报错，所以别照抄官方 demo 的 48"
        ),
    )
    parser.add_argument(
        "--video_mode",
        choices=("link", "transcode"),
        default="transcode",
        help=(
            "transcode=按集切成 h264（默认；本赛题数据是 AV1，decord 0.6.0 读不了，实测报 "
            "'cannot find video stream'）；link=直接引用原视频+start 帧偏移，只在数据本身是 "
            "h264/h265 时可用"
        ),
    )
    parser.add_argument("--max_episodes", type=int, default=None, help="只转前 N 集，调试用")
    parser.add_argument("--video_workers", type=int, default=12, help="并行转码的 ffmpeg 进程数")
    parser.add_argument(
        "--rot_scale",
        type=float,
        default=None,
        help="旋转槽缩放；默认按数据自动取，保证 ||s*q456|| < pi",
    )
    parser.add_argument(
        "--encoding",
        choices=("eef", "slot"),
        default="eef",
        help=(
            "eef=真实末端位姿（默认，语义与 XR-1 预训练一致，需要 FK）；"
            "slot=槽位复用（把关节塞进末端槽位，零样本语义不对但不依赖 URDF），两种模式可 A/B"
        ),
    )
    parser.add_argument("--urdf_path", default=None, help="eef 模式用的单臂 URDF，默认走 EmbodiChain 的 CobotMagicNoGripper")
    parser.add_argument("--self_test", action="store_true", help="只跑数值自检后退出")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        return self_test()

    if not args.repo_dir or not args.out_dir:
        print("错误: --repo_dir 和 --out_dir 必填（或用 --self_test）", file=sys.stderr)
        return 2

    root = find_dataset_root(os.path.abspath(os.path.expanduser(args.repo_dir)))
    reader = LeRobotReader(root)
    total = len(reader) if args.max_episodes is None else min(len(reader), args.max_episodes)

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    data_dir = os.path.join(out_dir, "data")
    video_dir = os.path.join(out_dir, "videos")
    os.makedirs(data_dir, exist_ok=True)

    print(f"数据集      : {root}")
    print(f"LeRobot 版本: {reader.version}  fps={reader.fps:g}  episodes={len(reader)} (转 {total} 集)")
    print(f"状态特征    : {reader.state_key}")
    print(f"视频模式    : {args.video_mode}")

    # eef 模式要用 FK 把关节换算成真实末端位姿
    fk = None
    if args.encoding == "eef":
        from xr1_fk import CobotMagicFK

        fk = CobotMagicFK(urdf_path=args.urdf_path)
        print(f"FK 串链    : {fk.urdf_path}")
        print(f"             关节顺序 {fk.joint_names()}（含 TCP）")

    # ---- 第一遍：读所有集，定 rot_scale
    print("\n[1/3] 读取轨迹 ...")
    episodes = []
    max_rotation_norm = 0.0
    for index in range(total):
        episode = reader.episode(index)
        episodes.append(episode)
        if args.encoding == "slot":
            for arm_slice in (LEFT_ARM, RIGHT_ARM):
                for array in (episode["qpos"], episode["action"]):
                    norms = np.linalg.norm(array[:, arm_slice][:, 3:6], axis=1)
                    max_rotation_norm = max(max_rotation_norm, float(norms.max()))
        if (index + 1) % 100 == 0:
            print(f"    {index + 1}/{total}")

    if args.rot_scale is not None:
        rot_scale = float(args.rot_scale)
    elif args.encoding == "eef":
        # 真实旋转，不做任何缩放；解码侧直接 log/exp，无需还原
        rot_scale = 1.0
    else:
        rot_scale = 1.0 if max_rotation_norm <= ROT_NORM_BUDGET else ROT_NORM_BUDGET / max_rotation_norm

    if args.encoding == "eef":
        print(f"    编码 = eef（真实末端位姿），rot_scale = {rot_scale:.6f}")
    else:
        print(f"    关节 4-6 三元组模长上限 = {max_rotation_norm:.4f} -> rot_scale = {rot_scale:.6f}")
        if rot_scale * max_rotation_norm >= math.pi:
            raise ValueError(
                f"rot_scale={rot_scale} 会让 ||s*q456||={rot_scale * max_rotation_norm:.4f} >= pi，指数映射不再单射"
            )

    # ---- 第二遍：写 JSON + 视频 + 累计统计
    print("\n[2/3] 写出 JSON / 视频并累计统计量 ...")
    accumulator = StatsAccumulator(action_length=args.action_length)
    instruction_used = None

    # 每个源视频文件只转一次；先统计它应有的总帧数用于校验
    expected_frames = {}
    for episode in episodes:
        for source in episode["videos"].values():
            expected_frames[source["path"]] = max(
                expected_frames.get(source["path"], 0), source["start"] + len(episode["qpos"])
            )

    transcoded = {}
    if args.video_mode == "transcode":
        # 1000 集 x 3 路 = 3000 个 AV1 视频，串行转码要几小时，这里并行跑。
        # ffmpeg 是子进程，线程池不受 GIL 影响。
        jobs = []
        for path in expected_frames:
            relative = os.path.relpath(path, reader.root).replace(os.sep, "__")
            destination = os.path.join(video_dir, relative)
            transcoded[path] = destination
            if not os.path.isfile(destination):
                jobs.append((path, destination, expected_frames.get(path)))

        if jobs:
            import concurrent.futures

            print(f"    并行转码 {len(jobs)} 个视频 (workers={args.video_workers}) ...")
            done = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.video_workers) as pool:
                futures = {
                    pool.submit(transcode_video, source, target, frames): source
                    for source, target, frames in jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # 有异常就抛出来，不静默吞掉
                    done += 1
                    if done % 100 == 0 or done == len(jobs):
                        print(f"      转码 {done}/{len(jobs)}")
        else:
            print("    转码产物已存在，跳过")

    for order, episode in enumerate(episodes):
        instruction = args.instruction or episode["task"] or "complete the task"
        instruction_used = instruction

        video_entries = {}
        for name, source in episode["videos"].items():
            path = source["path"]
            if args.video_mode == "transcode":
                path = transcoded[path]
            video_entries[name] = {
                "path": path,
                "start": source["start"],
                "crop_bbox": None,
            }

        payload = build_episode_json(
            episode, instruction, rot_scale, video_entries, args.encoding, fk
        )
        json_path = os.path.join(data_dir, f"episode_{episode['episode_index']:06d}.json")
        with open(json_path, "w") as handle:
            json.dump(payload, handle)

        packed = pack_relative_actions(
            episode["qpos"], episode["action"], rot_scale, args.action_length,
            args.encoding, fk,
        )
        accumulator.add(packed, compose_state_batch(episode["qpos"]))

        if (order + 1) % 50 == 0 or order + 1 == total:
            print(f"    {order + 1}/{total}")

    # ---- 第三遍：落统计量与训练配置
    print("\n[3/3] 计算统计量 ...")
    mean, std, q01, q99 = accumulator.finalize()

    # 夹爪量程随数据集而变（sample_loading 是 0~1 归一化，不是 0~0.05 米），
    # 写进 stats 让部署侧自适应，避免把夹爪指令裁掉
    gripper_values = np.concatenate(
        [
            np.stack([e["qpos"][:, LEFT_GRIPPER], e["qpos"][:, RIGHT_GRIPPER]], axis=1)
            for e in episodes
        ]
        + [
            np.stack([e["action"][:, LEFT_GRIPPER], e["action"][:, RIGHT_GRIPPER]], axis=1)
            for e in episodes
        ]
    )
    gripper_range = [float(gripper_values.min()), float(gripper_values.max())]
    print(f"    夹爪取值范围: [{gripper_range[0]:.4f}, {gripper_range[1]:.4f}]")

    stats_path = os.path.join(out_dir, "xr1_stats.json")
    with open(stats_path, "w") as handle:
        json.dump(
            {
                "action_length": args.action_length,
                "action_dim": ACTION_DIM,
                "state_dim": STATE_DIM,
                "encoding": args.encoding,
                "gripper_range": gripper_range,
                "rot_scale": rot_scale,
                "qpos_dim": 14,
                "instruction": instruction_used,
                "source_dataset": root,
                "num_episodes": total,
                "num_samples": accumulator.num_samples,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "q01": q01.tolist(),
                "q99": q99.tolist(),
            },
            handle,
        )

    yaml_path = os.path.join(out_dir, "xr1_data.yaml")
    write_data_yaml(
        yaml_path, data_dir, mean, std, q01, q99, rot_scale, args.action_length, args.batch_size,
        args.encoding, gripper_range,
    )

    print(f"\n完成: {total} 集 / {accumulator.num_samples} 个训练样本")
    print(f"  JSON  : {data_dir}")
    print(f"  统计量: {stats_path}")
    print(f"  训练配置: {yaml_path}")
    print(f"\n下一步: bash policy/xr1/finetune.sh {out_dir} <exp_name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
