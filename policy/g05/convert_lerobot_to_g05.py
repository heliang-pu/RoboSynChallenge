#!/usr/bin/env python
# ----------------------------------------------------------------------------
# LeRobot 数据集 -> G0.5 (GalaxeaVLA) 训练格式
#
# G0.5 读数据的方式是"声明式切片"：数据本身保持标准 LeRobot 布局，
# 由 configs/data/<name>.yaml 里的 shape_meta 声明每个部件从打平向量的哪一位开始、
# 取几维（见 GalaxeaVLA/configs/data/robotwin.yaml）。所以绝大多数情况下
# **不需要搬运数据本体**，只需要:
#   1. 确认数据是 G0.5 能读的版本（v2.1 / v3.0 都支持，见
#      src/g05/data/base_lerobot_dataset.py 的 lerobot_ds_version 开关）
#   2. 生成对应的 data yaml + task yaml
#
# 比赛数据（RoboSynChallenge/lerobot_dataset/）实测已经是 LeRobot v3.0，
# 且 key 名（observation.state / action / observation.images.cam_*）和维度（14/14）
# 与 G0.5 的 robotwin embodiment 完全一致，因此默认走"零搬运"路径。
# 只有拿到真正的 v2.1 数据集时才需要 --migrate 做目录迁移。
#
# 用法:
#   # 扫描 + 体检 + 生成 cobotmagic 配置
#   python policy/g05/convert_lerobot_to_g05.py scan lerobot_dataset --emit-config
#
#   # 单个数据集体检
#   python policy/g05/convert_lerobot_to_g05.py inspect <dataset_dir>
#
#   # 真 v2.1 数据集 -> v3.0 目录布局
#   python policy/g05/convert_lerobot_to_g05.py migrate <src_v21> <dst_v30>
#
#   # 自测（造假数据跑全流程，不依赖真实数据集）
#   python policy/g05/convert_lerobot_to_g05.py selftest
# ----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 14 维双臂布局：与比赛 env.step 的 (1,14) 绝对关节位置逐位对应
#   [0:6] 左臂 | [6] 左夹爪 | [7:13] 右臂 | [13] 右夹爪
# ---------------------------------------------------------------------------
PART_LAYOUT: List[Tuple[str, int, int]] = [
    # (part_key, start_index, dim)
    ("left_arm", 0, 6),
    ("left_gripper", 6, 1),
    ("right_arm", 7, 6),
    ("right_gripper", 13, 1),
]
FLAT_DIM = 14

# 相机 key -> G0.5 的 camera_type
CAMERA_TYPES = {
    "cam_high": "exterior",
    "cam_left_wrist": "wrist_left",
    "cam_right_wrist": "wrist_right",
}

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
IMAGE_FEATURE_PREFIX = "observation.images."

# G0.5 训练时把原图 resize 到这个尺寸（configs/data/robotwin.yaml 的 shape 字段）
TARGET_IMAGE_HW = (256, 256)


class DatasetProblem(Exception):
    pass


# ---------------------------------------------------------------------------
# 读取 / 体检
# ---------------------------------------------------------------------------

def load_info(dataset_dir: Path) -> Dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise DatasetProblem(f"不是 LeRobot 数据集（缺 meta/info.json）: {dataset_dir}")
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_version(info: Dict[str, Any]) -> str:
    """返回 '2.1' 或 '3.0'（对应 G0.5 的 lerobot_ds_version 取值）。"""
    raw = str(info.get("codebase_version", "")).strip().lstrip("v")
    if raw.startswith("3"):
        return "3.0"
    if raw.startswith("2"):
        return "2.1"
    # 没写版本就按路径模板猜：v3.0 用 file-{file_index}，v2.1 用 episode_{episode_index}
    data_path = str(info.get("data_path", ""))
    return "3.0" if "file_index" in data_path else "2.1"


def inspect_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """体检一个数据集，返回结构化结论；有硬伤则抛 DatasetProblem。"""
    info = load_info(dataset_dir)
    version = detect_version(info)
    features = info.get("features") or {}

    problems: List[str] = []

    def _shape_of(name: str) -> Optional[List[int]]:
        feat = features.get(name)
        return list(feat.get("shape", [])) if isinstance(feat, dict) else None

    state_shape = _shape_of(STATE_FEATURE)
    action_shape = _shape_of(ACTION_FEATURE)
    if state_shape is None:
        problems.append(f"缺少特征 `{STATE_FEATURE}`（现有: {sorted(features)}）")
    elif state_shape != [FLAT_DIM]:
        problems.append(f"`{STATE_FEATURE}` 维度是 {state_shape}，期望 [{FLAT_DIM}]")
    if action_shape is None:
        problems.append(f"缺少特征 `{ACTION_FEATURE}`")
    elif action_shape != [FLAT_DIM]:
        problems.append(f"`{ACTION_FEATURE}` 维度是 {action_shape}，期望 [{FLAT_DIM}]")

    cameras: Dict[str, Dict[str, Any]] = {}
    for name, feat in features.items():
        if not name.startswith(IMAGE_FEATURE_PREFIX) or not isinstance(feat, dict):
            continue
        cam = name[len(IMAGE_FEATURE_PREFIX):]
        shape = list(feat.get("shape", []))
        cameras[cam] = {
            "feature": name,
            "shape": shape,  # [H, W, C]
            "dtype": feat.get("dtype"),
            "codec": (feat.get("info") or {}).get("video.codec"),
        }
    for cam in CAMERA_TYPES:
        if cam not in cameras:
            problems.append(f"缺少相机 `{cam}`（现有: {sorted(cameras)}）")

    if problems:
        raise DatasetProblem(
            f"{dataset_dir} 不满足 G0.5 输入要求:\n  - " + "\n  - ".join(problems)
        )

    return {
        "dir": str(dataset_dir),
        "version": version,
        "fps": int(info.get("fps", 0)),
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
        "robot_type": info.get("robot_type"),
        "cameras": cameras,
        "state_shape": state_shape,
        "action_shape": action_shape,
    }


def scan_root(root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """递归找出 root 下所有 LeRobot 数据集（以 meta/info.json 为标志）。"""
    found, failures = [], []
    for info_path in sorted(root.rglob("meta/info.json")):
        ds_dir = info_path.parent.parent
        try:
            found.append(inspect_dataset(ds_dir))
        except DatasetProblem as exc:
            failures.append(str(exc))
    return found, failures


# ---------------------------------------------------------------------------
# 生成 G0.5 配置
# ---------------------------------------------------------------------------

def _shape_meta_block(indent: str) -> str:
    """action / state / images 三段 shape_meta，14 维切片声明。"""
    lines: List[str] = []

    for group, lerobot_key in (("action", ACTION_FEATURE), ("state", STATE_FEATURE)):
        lines.append(f"{indent}{group}:")
        for key, start, dim in PART_LAYOUT:
            lines += [
                f"{indent}- key: {key}",
                f"{indent}  lerobot_key: {lerobot_key}",
                f"{indent}  start_index: {start}",
                f"{indent}  raw_shape: {dim}",
                f"{indent}  shape: {dim}",
                f"{indent}  time_offset: 0",
            ]

    lines.append(f"{indent}images:")
    for cam, cam_type in CAMERA_TYPES.items():
        th, tw = TARGET_IMAGE_HW
        lines += [
            f"{indent}- key: {cam}",
            f"{indent}  camera_type: {cam_type}",
            f"{indent}  lerobot_key: {IMAGE_FEATURE_PREFIX}{cam}",
            f"{indent}  start_index: 0",
            f"{indent}  raw_shape:",
            f"{indent}  - 3",
            f"{indent}  - 480",
            f"{indent}  - 640",
            f"{indent}  shape:",
            f"{indent}  - 3",
            f"{indent}  - {th}",
            f"{indent}  - {tw}",
            f"{indent}  time_offset: 0",
        ]
    return "\n".join(lines)


def render_data_yaml(
    embodiment: str,
    dataset_dirs: List[str],
    lerobot_ds_version: str,
    action_size: int = 32,
) -> str:
    """生成 configs/data/<embodiment>.yaml，结构对齐官方 configs/data/robotwin.yaml。"""
    dirs_block = "\n".join(f"      - {d}" for d in dataset_dirs)
    meta_ds = _shape_meta_block("      ")
    meta_proc = _shape_meta_block("      ")
    return f"""# 由 policy/g05/convert_lerobot_to_g05.py 生成，请勿手工编辑
# RoboSynChallenge 双臂 14 维 embodiment: {embodiment}
#   打平布局 [0:6] 左臂 | [6] 左夹爪 | [7:13] 右臂 | [13] 右夹爪
_target_: g05.data.mixture_lerobot_dataset.MixtureLerobotDataset
use_weight_normalization: true
use_weight_for_sampling: false
action_size: {action_size}
past_action_size: 0
obs_size: 1
val_set_proportion: 0.01
is_training_set: true
embodiment_datasets:
  {embodiment}:
    type: g05.data.base_lerobot_datasetV3.BaseLerobotDatasetV3
    embodiment_type: {embodiment}
    lerobot_ds_version: '{lerobot_ds_version}'
    shape_meta:
{meta_ds}
    dataset_groups:
    - weight: 1.0
      dataset_dirs:
{dirs_block}
processors:
  {embodiment}:
    shape_meta:
{meta_proc}
    action_state_transforms: null
    train_transforms:
      cam_high: ${{oc.load:configs/data/_transforms.yaml,train_head}}
      cam_left_wrist: ${{oc.load:configs/data/_transforms.yaml,train_head}}
      cam_right_wrist: ${{oc.load:configs/data/_transforms.yaml,train_head}}
    val_transforms:
      cam_high: ${{oc.load:configs/data/_transforms.yaml,val_exterior}}
      cam_left_wrist: ${{oc.load:configs/data/_transforms.yaml,val_exterior}}
      cam_right_wrist: ${{oc.load:configs/data/_transforms.yaml,val_exterior}}
    norm_default_mode: z-score
    norm_exception_mode: null
    action_filter:
      _target_: g05.data_processor.transforms.action_filter.DummyActionFilter
    action_state_merger:
      _target_: g05.data_processor.transforms.action_state_merger.PaddingActionMerger
      merge: true
      max_action_shape_meta: {FLAT_DIM}
      max_state_shape_meta: {FLAT_DIM}
    num_obs_steps: 1
    use_stepwise_action_norm: false
    drop_high_level_prob: 1.0
    use_zh_instruction: false
"""


def render_task_yaml(embodiment: str) -> str:
    """生成 configs/task/<embodiment>.yaml，结构对齐官方 configs/task/robotwin.yaml。

    关键点: model.processor.num_output_cameras / num_input_cameras 必须是 3，
    否则会继承 g05-base 的 18 路相机，部署时对不上。
    """
    # `# @package _global_` 必须是**文件第一行**，前面连注释都不能有，
    # 否则 hydra 不把它当全局包，`override /model` 会被解释成 `model@task.model`，
    # 报 "Could not override 'model@task.model'"。
    return f"""# @package _global_
# 由 policy/g05/convert_lerobot_to_g05.py 生成，请勿手工编辑
defaults:
- override /model: g05
- override /tokenizer: actioncodec
- override /data: {embodiment}
- _self_
# 留空则由 finetune.py 从数据集现算，并写到产物目录的 dataset_stats.json
datastatics_path: null
model:
  use_torch_compile: false
  use_8bit_optimizer: false
  backbone_lr_multiplier: 1.0
  batch_size: 16
  max_epochs: 2
  max_steps: null
  learning_rate: 1.0e-05
  weight_decay: 0.01
  betas:
  - 0.9
  - 0.95
  warmup_steps: 500
  num_workers: 8
  find_unused_parameters: true
  model_arch:
    position_ids_type: pi0fast
    action_position_offset: null
    ae_vlm_condition_mode: cross_attn_only
    checkpoint_joint: false
    checkpoint_vision: false
    checkpoint_action_decoder: false
    action_dim: 20
    proprio_dim: 20
    discrete_action: false
    continuous_action: true
    fm:
      joint_training: true
      fm_weight: 1.0
      num_flow_samples: 8
    ar:
      ce_weight: 0.0
  processor:
    num_output_cameras: 3
    num_input_cameras: 3
    camera_size_config:
      exterior:
      - 256
      - 256
      wrist_left:
      - 256
      - 256
      wrist_right:
      - 256
      - 256
    norm_default_mode: q01/q99
    vlm_input_action_norm_default_mode: null
    vlm_input_action_norm_exception_mode: null
    tokenizer_params:
      local_files_only: true
tokenizer:
  vq_config:
    block_wise_autoregressive: false
    model_arch:
      codebook_size: 4096
    num_residuals: 2
    rule_based_key_patterns:
    - gripper
    rule_based_binarize_threshold: 0.0
    use_group_markers: true
    group_order_shuffle: false
    absent_key_fill_value: -100.0
    dropout_noop_parts: false
"""


# ---------------------------------------------------------------------------
# v2.1 -> v3.0 目录迁移
# ---------------------------------------------------------------------------

V21_DATA_TMPL = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_TMPL = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V30_DATA_TMPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V30_VIDEO_TMPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def migrate_v21_to_v30(src: Path, dst: Path, link_videos: bool = True) -> Dict[str, Any]:
    """把 v2.1 目录布局迁移成 v3.0。

    v2.1: 每 episode 一个 parquet / 一个 mp4，meta 用 jsonl
    v3.0: 同 chunk 的 episode 合并成一个 parquet 文件，meta 用 parquet

    视频默认软链（数据动辄几十 GB，没必要复制）。
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise DatasetProblem("迁移需要 pyarrow，请在 g05 venv 里跑这个脚本") from exc

    info = load_info(src)
    if detect_version(info) != "2.1":
        raise DatasetProblem(f"{src} 不是 v2.1 数据集，无需迁移")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(exist_ok=True)

    chunks_size = int(info.get("chunks_size", 1000))
    total_episodes = int(info.get("total_episodes", 0))
    video_keys = [k for k in (info.get("features") or {}) if k.startswith(IMAGE_FEATURE_PREFIX)]

    # --- episodes.jsonl -> 每集元信息 ---
    episodes_meta: Dict[int, Dict[str, Any]] = {}
    ep_jsonl = src / "meta" / "episodes.jsonl"
    if ep_jsonl.is_file():
        with open(ep_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    episodes_meta[int(rec["episode_index"])] = rec

    # --- tasks.jsonl -> tasks.parquet ---
    tasks: List[str] = []
    tasks_jsonl = src / "meta" / "tasks.jsonl"
    if tasks_jsonl.is_file():
        with open(tasks_jsonl, "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rows.sort(key=lambda r: int(r.get("task_index", 0)))
        tasks = [str(r["task"]) for r in rows]
    if tasks:
        pq.write_table(
            pa.table({"task_index": list(range(len(tasks))), "task": tasks}),
            dst / "meta" / "tasks.parquet",
        )

    # --- 按 chunk 合并 data parquet ---
    ep_rows: List[Dict[str, Any]] = []
    dataset_cursor = 0
    for chunk_index in range((total_episodes + chunks_size - 1) // chunks_size):
        lo = chunk_index * chunks_size
        hi = min(lo + chunks_size, total_episodes)
        tables, per_ep = [], []
        for ep in range(lo, hi):
            rel = V21_DATA_TMPL.format(episode_chunk=chunk_index, episode_index=ep)
            ep_parquet = src / rel
            if not ep_parquet.is_file():
                raise DatasetProblem(f"缺少 episode parquet: {ep_parquet}")
            table = pq.read_table(ep_parquet)
            tables.append(table)
            per_ep.append((ep, table.num_rows))
        if not tables:
            continue
        merged = pa.concat_tables(tables)
        out_rel = V30_DATA_TMPL.format(chunk_index=chunk_index, file_index=0)
        out_path = dst / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged, out_path)

        for ep, n_rows in per_ep:
            meta = episodes_meta.get(ep, {})
            row: Dict[str, Any] = {
                "episode_index": ep,
                "tasks": list(meta.get("tasks", []) or []),
                "length": int(meta.get("length", n_rows)),
                "data/chunk_index": chunk_index,
                "data/file_index": 0,
                "dataset_from_index": dataset_cursor,
                "dataset_to_index": dataset_cursor + n_rows,
            }
            for vk in video_keys:
                row[f"videos/{vk}/chunk_index"] = chunk_index
                row[f"videos/{vk}/file_index"] = ep
                row[f"videos/{vk}/from_timestamp"] = 0.0
                row[f"videos/{vk}/to_timestamp"] = n_rows / float(info.get("fps", 30))
            ep_rows.append(row)
            dataset_cursor += n_rows

        # --- 视频：v2.1 一集一个文件，v3.0 用 file_index 区分，逐集软链 ---
        for vk in video_keys:
            for ep, _ in per_ep:
                src_v = src / V21_VIDEO_TMPL.format(
                    episode_chunk=chunk_index, video_key=vk, episode_index=ep
                )
                if not src_v.is_file():
                    raise DatasetProblem(f"缺少视频: {src_v}")
                dst_v = dst / V30_VIDEO_TMPL.format(
                    video_key=vk, chunk_index=chunk_index, file_index=ep
                )
                dst_v.parent.mkdir(parents=True, exist_ok=True)
                if dst_v.exists() or dst_v.is_symlink():
                    dst_v.unlink()
                if link_videos:
                    dst_v.symlink_to(src_v.resolve())
                else:
                    shutil.copy2(src_v, dst_v)

    # --- meta/episodes/*.parquet ---
    if ep_rows:
        ep_out = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ep_out.parent.mkdir(parents=True, exist_ok=True)
        columns = {k: [r.get(k) for r in ep_rows] for k in ep_rows[0]}
        pq.write_table(pa.table(columns), ep_out)

    # --- info.json ---
    new_info = dict(info)
    new_info["codebase_version"] = "v3.0"
    new_info["data_path"] = V30_DATA_TMPL
    new_info["video_path"] = V30_VIDEO_TMPL
    new_info.setdefault("data_files_size_in_mb", 100)
    new_info.setdefault("video_files_size_in_mb", 200)
    with open(dst / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(new_info, f, indent=4, ensure_ascii=False)

    return {"episodes": len(ep_rows), "dst": str(dst)}


# ---------------------------------------------------------------------------
# 自测：造一份假数据跑通全流程
# ---------------------------------------------------------------------------

def _fake_features() -> Dict[str, Any]:
    names14 = [f"j{i}" for i in range(FLAT_DIM)]
    features: Dict[str, Any] = {
        STATE_FEATURE: {"dtype": "float32", "shape": [FLAT_DIM], "names": names14},
        ACTION_FEATURE: {"dtype": "float32", "shape": [FLAT_DIM], "names": names14},
    }
    for cam in CAMERA_TYPES:
        features[f"{IMAGE_FEATURE_PREFIX}{cam}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.codec": "av1", "video.fps": 25},
        }
    for extra, dtype in (("timestamp", "float32"), ("frame_index", "int64"),
                         ("episode_index", "int64"), ("index", "int64"),
                         ("task_index", "int64")):
        features[extra] = {"dtype": dtype, "shape": [1], "names": None}
    return features


def _write_fake_v30(root: Path, n_episodes: int = 2, n_frames: int = 4) -> Path:
    ds = root / "fake_v30"
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "cobotmagic",
        "total_episodes": n_episodes,
        "total_frames": n_episodes * n_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 25,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": V30_DATA_TMPL,
        "video_path": V30_VIDEO_TMPL,
        "features": _fake_features(),
    }
    with open(ds / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    return ds


def _write_fake_v21(root: Path, n_episodes: int = 2, n_frames: int = 4) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = root / "fake_v21"
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "cobotmagic",
        "total_episodes": n_episodes,
        "total_frames": n_episodes * n_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 25,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": V21_DATA_TMPL,
        "video_path": V21_VIDEO_TMPL,
        "features": _fake_features(),
    }
    with open(ds / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    with open(ds / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": "fake task"}) + "\n")
    with open(ds / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            f.write(json.dumps({"episode_index": ep, "tasks": ["fake task"], "length": n_frames}) + "\n")

    for ep in range(n_episodes):
        table = pa.table({
            STATE_FEATURE: pa.array([[0.1 * i] * FLAT_DIM for i in range(n_frames)],
                                    type=pa.list_(pa.float32(), FLAT_DIM)),
            ACTION_FEATURE: pa.array([[0.2 * i] * FLAT_DIM for i in range(n_frames)],
                                     type=pa.list_(pa.float32(), FLAT_DIM)),
            "timestamp": pa.array([i / 25.0 for i in range(n_frames)], type=pa.float32()),
            "frame_index": pa.array(list(range(n_frames)), type=pa.int64()),
            "episode_index": pa.array([ep] * n_frames, type=pa.int64()),
            "index": pa.array([ep * n_frames + i for i in range(n_frames)], type=pa.int64()),
            "task_index": pa.array([0] * n_frames, type=pa.int64()),
        })
        out = ds / V21_DATA_TMPL.format(episode_chunk=0, episode_index=ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)

        for cam in CAMERA_TYPES:
            vk = f"{IMAGE_FEATURE_PREFIX}{cam}"
            v = ds / V21_VIDEO_TMPL.format(episode_chunk=0, video_key=vk, episode_index=ep)
            v.parent.mkdir(parents=True, exist_ok=True)
            v.write_bytes(b"\x00fake mp4")
    return ds


def selftest() -> int:
    import pyarrow.parquet as pq

    print("=== selftest: 造假数据跑全流程 ===")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. v3.0 体检
        ds30 = _write_fake_v30(root)
        report = inspect_dataset(ds30)
        assert report["version"] == "3.0", report
        assert report["fps"] == 25, report
        assert sorted(report["cameras"]) == sorted(CAMERA_TYPES), report
        print("  ok  v3.0 体检")

        # 2. 配置生成：切片必须严格是 0/6/7/13
        data_yaml = render_data_yaml("cobotmagic", [str(ds30)], "3.0")
        for key, start, dim in PART_LAYOUT:
            assert f"- key: {key}" in data_yaml, key
            assert f"start_index: {start}" in data_yaml, (key, start)
        assert "raw_shape: 6" in data_yaml and "raw_shape: 1" in data_yaml
        assert str(ds30) in data_yaml
        assert "embodiment_type: cobotmagic" in data_yaml
        task_yaml = render_task_yaml("cobotmagic")
        assert "override /data: cobotmagic" in task_yaml
        assert "num_output_cameras: 3" in task_yaml
        # hydra 要求这行必须是文件第一行，否则 override /model 会失效
        assert task_yaml.splitlines()[0] == "# @package _global_", task_yaml.splitlines()[0]
        print("  ok  data/task yaml 生成")

        # 3. 缺相机要报错
        bad_info = json.loads((ds30 / "meta" / "info.json").read_text())
        del bad_info["features"][f"{IMAGE_FEATURE_PREFIX}cam_high"]
        bad = root / "bad"
        (bad / "meta").mkdir(parents=True)
        (bad / "meta" / "info.json").write_text(json.dumps(bad_info))
        try:
            inspect_dataset(bad)
        except DatasetProblem as exc:
            assert "cam_high" in str(exc)
            print("  ok  缺相机被拦下")
        else:
            raise AssertionError("缺相机居然没报错")

        # 4. 维度不对要报错
        bad2_info = json.loads((ds30 / "meta" / "info.json").read_text())
        bad2_info["features"][ACTION_FEATURE]["shape"] = [16]
        bad2 = root / "bad2"
        (bad2 / "meta").mkdir(parents=True)
        (bad2 / "meta" / "info.json").write_text(json.dumps(bad2_info))
        try:
            inspect_dataset(bad2)
        except DatasetProblem as exc:
            assert "[16]" in str(exc)
            print("  ok  动作维度不符被拦下")
        else:
            raise AssertionError("维度不符居然没报错")

        # 5. v2.1 -> v3.0 迁移
        ds21 = _write_fake_v21(root)
        assert detect_version(load_info(ds21)) == "2.1"
        out30 = root / "migrated"
        result = migrate_v21_to_v30(ds21, out30)
        assert result["episodes"] == 2, result
        migrated = inspect_dataset(out30)
        assert migrated["version"] == "3.0", migrated
        merged = pq.read_table(out30 / "data" / "chunk-000" / "file-000.parquet")
        assert merged.num_rows == 8, merged.num_rows
        eps = pq.read_table(out30 / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        assert eps.num_rows == 2
        cols = set(eps.column_names)
        for required in ("episode_index", "length", "data/chunk_index",
                         "dataset_from_index", "dataset_to_index"):
            assert required in cols, required
        d = eps.to_pydict()
        assert d["dataset_from_index"] == [0, 4] and d["dataset_to_index"] == [4, 8], d
        assert (out30 / "meta" / "tasks.parquet").is_file()
        for cam in CAMERA_TYPES:
            vk = f"{IMAGE_FEATURE_PREFIX}{cam}"
            v = out30 / V30_VIDEO_TMPL.format(video_key=vk, chunk_index=0, file_index=0)
            assert v.is_symlink() and v.resolve().is_file(), v
        print("  ok  v2.1 -> v3.0 迁移（parquet 合并 / episodes 索引 / 视频软链）")

    print("=== selftest 全部通过 ===")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _emit_configs(out_dir: Path, embodiment: str, dataset_dirs: List[str], version: str) -> None:
    data_dir, task_dir = out_dir / "data", out_dir / "task"
    data_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"{embodiment}.yaml"
    task_path = task_dir / f"{embodiment}.yaml"
    data_path.write_text(render_data_yaml(embodiment, dataset_dirs, version), encoding="utf-8")
    task_path.write_text(render_task_yaml(embodiment), encoding="utf-8")
    print(f"\n已生成配置:\n  {data_path}\n  {task_path}")
    print(f"\n下一步:\n  bash policy/g05/finetune.sh <num_gpus> {embodiment}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LeRobot 数据集 -> G0.5 训练格式（体检 / 生成配置 / v2.1 迁移）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="递归扫描目录下所有 LeRobot 数据集并体检")
    p_scan.add_argument("root", type=Path)
    p_scan.add_argument("--emit-config", action="store_true", help="生成 data/task yaml")
    p_scan.add_argument("--embodiment", default="cobotmagic")
    p_scan.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent / "configs",
        help="配置输出目录（默认 policy/g05/configs）",
    )

    p_inspect = sub.add_parser("inspect", help="体检单个数据集")
    p_inspect.add_argument("dataset_dir", type=Path)

    p_mig = sub.add_parser("migrate", help="v2.1 目录布局 -> v3.0")
    p_mig.add_argument("src", type=Path)
    p_mig.add_argument("dst", type=Path)
    p_mig.add_argument("--copy-videos", action="store_true", help="复制视频而不是软链")

    sub.add_parser("selftest", help="造假数据跑通全流程（不需要真实数据集）")

    args = parser.parse_args(argv)

    if args.command == "selftest":
        return selftest()

    if args.command == "inspect":
        report = inspect_dataset(args.dataset_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "migrate":
        result = migrate_v21_to_v30(args.src, args.dst, link_videos=not args.copy_videos)
        print(f"迁移完成: {result['episodes']} 个 episode -> {result['dst']}")
        inspect_dataset(args.dst)
        print("迁移产物体检通过")
        return 0

    # scan
    found, failures = scan_root(args.root)
    if not found:
        print(f"在 {args.root} 下没找到可用的 LeRobot 数据集", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1

    versions = sorted({r["version"] for r in found})
    fps_values = sorted({r["fps"] for r in found})
    codecs = sorted({c["codec"] for r in found for c in r["cameras"].values() if c["codec"]})
    total_eps = sum(r["total_episodes"] for r in found)
    total_frames = sum(r["total_frames"] for r in found)

    print(f"扫描 {args.root}")
    print(f"  可用数据集: {len(found)}  episodes={total_eps}  frames={total_frames}")
    print(f"  LeRobot 版本: {versions}")
    print(f"  fps: {fps_values}")
    print(f"  视频编码: {codecs}")
    for r in found:
        print(f"    - {r['dir']}  ep={r['total_episodes']} frames={r['total_frames']} v{r['version']}")
    if failures:
        print(f"\n  跳过 {len(failures)} 个不合格目录:")
        for msg in failures:
            print(f"    {msg.splitlines()[0]}")

    if len(versions) > 1:
        print(
            f"\n[警告] 扫描到多个 LeRobot 版本 {versions}，"
            "一个 data yaml 只能声明一个 lerobot_ds_version，请分开生成或先 migrate。",
            file=sys.stderr,
        )
        return 1

    if args.emit_config:
        _emit_configs(args.out_dir, args.embodiment, [r["dir"] for r in found], versions[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
