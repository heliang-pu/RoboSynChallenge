"""端到端验证：用上游真实的 JsonDataset 加载转换产物，再解码回绝对 qpos。

覆盖 JSON schema 合法性、decord 读视频、上游相对动作打包，以及
xr1_model.decode_action 的逆运算精度。分三段定位误差来源：
  A. 上游打包 vs 本地 pack_relative_actions  -> 编码是否一致
  B. 反归一化往返                            -> float32 精度损失
  C. 解码 -> 绝对 qpos                       -> 整链路
"""

import json
import os
import sys

import numpy as np

# 用法: python test_roundtrip.py <转换输出目录> <对应的 LeRobot 数据集根>
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "policy/xr1/training_data/click_bell")
LEROBOT = (
    sys.argv[2]
    if len(sys.argv) > 2
    else os.path.join(REPO, "lerobot_dataset/click_bell_aug_base/cobotmagic_Sim_click_the_bell_000")
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.io import ACTION_EPS, build_action_mask

from convert_lerobot_to_xr1 import decode_packed, pack_relative_actions

with open(os.path.join(OUT, "xr1_stats.json")) as handle:
    stats = json.load(handle)

mean = np.asarray(stats["mean"], dtype=np.float32)
std = np.asarray(stats["std"], dtype=np.float32)
q01 = np.asarray(stats["q01"], dtype=np.float32)
q99 = np.asarray(stats["q99"], dtype=np.float32)
rot_scale = float(stats["rot_scale"])
action_length = int(stats["action_length"])

params = {
    "max_steps": 370,
    "train_datasets": {
        "batch_size": 1,
        "action_length": action_length,
        "paths": [os.path.join(OUT, "data")],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    },
}

print("== 构建上游 JsonDataset ==")
dataset = JsonDataset(params)
print(f"   源样本数 = {len(dataset.samples)}  文件数 = {len(dataset.files)}")

import pandas as pd

import glob
frame_table = pd.read_parquet(sorted(glob.glob(os.path.join(LEROBOT, "data/**/*.parquet"), recursive=True))[0])
episodes = pd.read_parquet(sorted(glob.glob(os.path.join(LEROBOT, "meta/episodes/**/*.parquet"), recursive=True))[0])

mask = build_action_mask(action_length).astype(bool)

worst = {"encode": 0.0, "denorm": 0.0, "decode": 0.0, "state": 0.0}
clipped_total = 0
checked = 0

for probe_episode, probe_frame in [(0, 0), (0, 20), (0, 60), (2, 5), (4, 40), (3, 70)]:
    row = episodes.iloc[probe_episode]
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    chunk = frame_table[(frame_table["index"] >= start) & (frame_table["index"] < stop)]
    qpos_truth = np.stack(chunk["observation.state"].values).astype(np.float32)
    action_truth = np.stack(chunk["action"].values).astype(np.float32)
    num_frames = len(qpos_truth)
    if probe_frame >= num_frames:
        continue

    sample_index = next(
        i
        for i, sample in enumerate(dataset.samples)
        if sample["file"].endswith(f"episode_{probe_episode:06d}.json")
        and sample["frame_index"] == probe_frame
    )
    sample = dataset[sample_index]

    # ---- A/B: 上游归一化动作 -> 反归一化 -> 与本地打包对比
    upstream_packed = sample["action"].numpy().astype(np.float32)
    upstream_packed = np.where(mask, upstream_packed * (std + ACTION_EPS) + mean, 0.0)

    local_packed = pack_relative_actions(
        qpos_truth, action_truth, rot_scale, action_length
    )[probe_frame]
    local_packed = np.where(mask, local_packed, 0.0)

    # 本地打包同样走一遍归一化往返，隔离出纯 float32 精度损失
    local_roundtrip = np.where(
        mask, ((local_packed - mean) / (std + ACTION_EPS)) * (std + ACTION_EPS) + mean, 0.0
    )

    encode_error = np.abs(upstream_packed - local_roundtrip).max()
    denorm_error = np.abs(local_roundtrip - local_packed).max()
    worst["encode"] = max(worst["encode"], float(encode_error))
    worst["denorm"] = max(worst["denorm"], float(denorm_error))

    # ---- C: 解码回绝对 qpos
    decoded = decode_packed(upstream_packed, qpos_truth[probe_frame], rot_scale)
    horizon = np.minimum(probe_frame + np.arange(action_length), num_frames - 1)
    decode_error = np.abs(decoded - action_truth[horizon]).max()
    worst["decode"] = max(worst["decode"], float(decode_error))

    # ---- 状态：只比对没有被 q01/q99 截断的维度
    state = sample["state"].numpy().astype(np.float32)
    valid = q99 > q01
    recovered = np.zeros_like(state)
    recovered[valid] = (state[valid] + 1.0) / 2.0 * (q99[valid] - q01[valid] + ACTION_EPS) + q01[valid]

    packed_truth = np.zeros((1, 60), dtype=np.float32)
    packed_truth[0, 0:6] = qpos_truth[probe_frame][0:6]
    packed_truth[0, 7] = qpos_truth[probe_frame][6]
    packed_truth[0, 8:14] = qpos_truth[probe_frame][7:13]
    packed_truth[0, 15] = qpos_truth[probe_frame][13]

    inside = valid & (packed_truth >= q01) & (packed_truth <= q99)
    clipped = int((valid & ~inside).sum())
    clipped_total += clipped
    state_error = (
        np.abs(recovered[inside] - packed_truth[inside]).max() if inside.any() else 0.0
    )
    worst["state"] = max(worst["state"], float(state_error))

    images = [c for m in sample["messages"] for c in m["content"] if c.get("type") == "image"]
    checked += 1
    print(
        f"   ep{probe_episode} frame{probe_frame:3d}: "
        f"编码={encode_error:.2e} 反归一化={denorm_error:.2e} 解码={decode_error:.2e} "
        f"状态={state_error:.2e} (截断{clipped}维) 图像={len(images)}张{images[0]['image'].size}"
    )

print()
print(f"== 结果（{checked} 个采样点）==")
print(f"   A 上游打包 vs 本地打包   : {worst['encode']:.3e}   (float32 舍入，见 diag_precision.py)")
print(f"   B 归一化往返精度损失     : {worst['denorm']:.3e}   (float32 固有)")
print(f"   C 解码回绝对 qpos 目标   : {worst['decode']:.3e}   (弧度，约 0.02 度)")
print(f"   状态还原（未截断维）     : {worst['state']:.3e}")
print(f"   被 q01/q99 截断的状态维  : {clipped_total} 个（属预期，分位数归一化本就会裁剪离群值）")

# A/C 的残差来源已定位：上游 rotm2aa 全程 float32，把本地计算强制成 float32 后
# 与上游逐位相同（diag_precision.py 实测 0.000e+00）。1e-3 弧度 = 0.06 度，
# 远低于机器人控制分辨率，取此为阈值。
ok = worst["encode"] < 1e-3 and worst["decode"] < 1e-3 and worst["state"] < 1e-4
print("   " + ("通过" if ok else "失败"))
sys.exit(0 if ok else 1)
