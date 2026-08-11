"""EEF 编码往返验证：上游真实 JsonDataset -> 反归一化 -> 解回绝对末端位姿 -> 对比 FK(action)。

不涉及 IK（那是部署侧的事），只验证「编码 + 上游相对化 + 解码」这条链是准确的。
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/phl/workspace/xr1_eef_smoke"
DATASET = "/home/phl/workspace/datasets/cobotmagic_Sim_sample_loading"

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.io import ACTION_EPS, aa2rotm, build_action_mask

from xr1_fk import CobotMagicFK

with open(os.path.join(OUT, "xr1_stats.json")) as handle:
    stats = json.load(handle)
assert stats["encoding"] == "eef", stats["encoding"]

mean = np.asarray(stats["mean"], dtype=np.float32)
std = np.asarray(stats["std"], dtype=np.float32)
action_length = int(stats["action_length"])
mask = build_action_mask(action_length).astype(bool)

dataset = JsonDataset({
    "max_steps": 4000,
    "train_datasets": {
        "batch_size": 1, "action_length": action_length,
        "paths": [os.path.join(OUT, "data")],
        "mean": mean.tolist(), "std": std.tolist(),
        "q01": stats["q01"], "q99": stats["q99"],
    },
})
print(f"上游 JsonDataset: {len(dataset.files)} 个 JSON, {len(dataset.samples)} 个样本")

import pandas as pd

fk = CobotMagicFK()
SIDES = (("left", slice(0, 6), slice(0, 3), slice(3, 6)),
         ("right", slice(7, 13), slice(8, 11), slice(11, 14)))

worst_position = 0.0
worst_rotation = 0.0
checked = 0

for episode_index, probe_frame in [(0, 0), (0, 50), (1, 120), (2, 200), (3, 30)]:
    path = os.path.join(DATASET, "data", "chunk-000", f"episode_{episode_index:06d}.parquet")
    if not os.path.isfile(path):
        continue
    table = pd.read_parquet(path, columns=["observation.state", "action"])
    qpos = np.stack(table["observation.state"].values).astype(np.float64)
    action = np.stack(table["action"].values).astype(np.float64)
    num_frames = len(qpos)
    if probe_frame >= num_frames:
        continue

    sample_index = next(
        i for i, s in enumerate(dataset.samples)
        if s["file"].endswith(f"episode_{episode_index:06d}.json") and s["frame_index"] == probe_frame
    )
    packed = dataset[sample_index]["action"].numpy().astype(np.float32)
    packed = np.where(mask, packed * (std + ACTION_EPS) + mean, 0.0)

    horizon = np.minimum(probe_frame + np.arange(action_length), num_frames - 1)

    for side, arm_slice, pos_slot, aa_slot in SIDES:
        anchor_position, anchor_rotation = fk.fk_pos_rotm(qpos[probe_frame : probe_frame + 1, arm_slice])
        anchor_position, anchor_rotation = anchor_position[0], anchor_rotation[0]

        # 解码：p_t = p_a + R_a · dims[0:3]；R_t = R_a · exp(dims[3:6])
        decoded_position = anchor_position[None] + packed[:, pos_slot] @ anchor_rotation.T
        decoded_rotation = np.stack(
            [anchor_rotation @ aa2rotm(delta) for delta in packed[:, aa_slot]], axis=0
        )

        truth_position, truth_rotation = fk.fk_pos_rotm(action[horizon, arm_slice])

        position_error = np.linalg.norm(decoded_position - truth_position, axis=1).max() * 1000.0
        relative = np.einsum("nji,njk->nik", decoded_rotation, truth_rotation)
        trace = np.clip((np.einsum("nii->n", relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation_error = np.degrees(np.arccos(trace)).max()

        worst_position = max(worst_position, float(position_error))
        worst_rotation = max(worst_rotation, float(rotation_error))
        checked += 1
        print(f"  ep{episode_index} f{probe_frame:3d} {side:<5}: "
              f"位置 {position_error:.5f} mm  旋转 {rotation_error:.5f} deg")

print()
print(f"== EEF 往返（{checked} 组）==")
print(f"   绝对末端位置最大误差: {worst_position:.5f} mm")
print(f"   绝对末端旋转最大误差: {worst_rotation:.5f} deg")
ok = worst_position < 1.0 and worst_rotation < 0.1
print("   " + ("通过" if ok else "失败"))
sys.exit(0 if ok else 1)
