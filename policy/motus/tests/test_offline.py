#!/usr/bin/env python3
"""Offline checks for the Motus adapter — no weights, no GPU, no simulator.

    python policy/motus/tests/test_offline.py

Covers the two things most likely to be silently wrong:
  * the three-view frame geometry the checkpoint was trained on, and
  * the LeRobot v3.0 -> Motus native conversion round-trip.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

POLICY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POLICY_DIR))

import prepare_data  # noqa: E402
from motus_model import MotusPolicy, build_three_view  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
def test_three_view_geometry():
    print("\n[1] three-view stitch geometry")
    high = np.full((480, 640, 3), 10, np.uint8)
    left = np.full((480, 640, 3), 20, np.uint8)
    right = np.full((480, 640, 3), 30, np.uint8)

    out = build_three_view(high, left, right)
    check("challenge 480x640 -> 720x640", out.shape == (720, 640, 3), out.shape)
    check("top band is cam_high", int(out[100, 320, 0]) == 10)
    check("bottom-left is cam_left", int(out[600, 100, 0]) == 20)
    check("bottom-right is cam_right", int(out[600, 500, 0]) == 30)

    # RoboTwin trained at 240x320; both must land on the same padded canvas.
    rt = build_three_view(
        np.zeros((240, 320, 3), np.uint8),
        np.zeros((240, 320, 3), np.uint8),
        np.zeros((240, 320, 3), np.uint8),
    )
    check("robotwin 240x320 -> 360x320", rt.shape == (360, 320, 3), rt.shape)

    sys.path.insert(0, str(POLICY_DIR / "Motus" / "inference" / "robotwin" / "Motus"))
    from utils.image_utils import resize_with_padding

    padded_challenge = resize_with_padding(out, (384, 320))
    padded_robotwin = resize_with_padding(rt, (384, 320))
    check("both resize to 384x320", padded_challenge.shape == padded_robotwin.shape == (384, 320, 3))
    # 720x640 and 360x320 scale to an identical 360x320 image, so the padding
    # (12 black rows top and bottom) must match too.
    check("identical padding rows", bool((padded_challenge[:12] == 0).all() and (padded_robotwin[:12] == 0).all()))

    # prepare_data must stitch exactly like the deploy path.
    same = prepare_data.stitch_three_view(high, left, right)
    check("prepare_data stitch == deploy stitch", bool(np.array_equal(same, out)))

    shrunk = prepare_data.resize_keep_aspect(out, (360, 320))
    check("converter output is 360x320", shrunk.shape == (360, 320, 3), shrunk.shape)


# ---------------------------------------------------------------------------
def test_action_postprocess():
    print("\n[2] action post-processing")
    fake = types.SimpleNamespace(
        gripper_indices=(6, 13),
        gripper_scale=0.05,
        gripper_limits=(0.0, 0.05),
        action_clip=None,
    )
    actions = np.ones((4, 14), dtype=np.float32)
    actions[:, 6] = 1.0    # RoboTwin gripper fully open
    actions[:, 13] = 2.0   # out of range on purpose
    out = MotusPolicy._postprocess(fake, actions)
    check("gripper 1.0 -> 0.05", abs(float(out[0, 6]) - 0.05) < 1e-6, float(out[0, 6]))
    check("gripper clamped to 0.05", abs(float(out[0, 13]) - 0.05) < 1e-6, float(out[0, 13]))
    check("arm joints untouched", abs(float(out[0, 0]) - 1.0) < 1e-6)

    identity = types.SimpleNamespace(
        gripper_indices=(6, 13), gripper_scale=1.0, gripper_limits=None, action_clip=None
    )
    out2 = MotusPolicy._postprocess(identity, actions.copy())
    check("default config is identity", bool(np.array_equal(out2, actions)))


def test_normalization_roundtrip():
    print("\n[3] normalization round-trip")
    fake = types.SimpleNamespace(
        action_normalization="minmax",
        action_min=np.zeros(14, np.float32),
        action_max=np.concatenate([np.full(6, 2.0), [0.05], np.full(6, 2.0), [0.05]]).astype(np.float32),
    )
    raw = np.random.RandomState(0).uniform(0, 0.05, size=(3, 14)).astype(np.float32)
    norm = MotusPolicy._normalize(fake, raw)
    back = MotusPolicy._denormalize(fake, norm)
    check("minmax normalize/denormalize", bool(np.allclose(raw, back, atol=1e-6)))

    none_cfg = types.SimpleNamespace(action_normalization="none")
    check("none mode is a no-op",
          bool(np.array_equal(MotusPolicy._normalize(none_cfg, raw), raw)))


# ---------------------------------------------------------------------------
def make_fake_lerobot_v3(root: Path, n_episodes=2, n_frames=12, fps=25):
    """Build a minimal but structurally faithful LeRobot v3.0 dataset."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    cams = list(prepare_data.CAM_KEYS)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "cobotmagic",
        "total_episodes": n_episodes,
        "total_frames": n_episodes * n_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            **{c: {"dtype": "video", "shape": [480, 640, 3]} for c in cams},
        },
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    rng = np.random.RandomState(7)
    actions, states, ep_col = [], [], []
    for ep in range(n_episodes):
        a = rng.uniform(-1, 1, size=(n_frames, 14)).astype(np.float32)
        a[:, 6] = 0.02
        a[:, 13] = 0.04
        actions.append(a)
        states.append(a + 0.001)
        ep_col.extend([ep] * n_frames)
    actions_all = np.concatenate(actions)
    states_all = np.concatenate(states)

    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "action": pa.array(actions_all.tolist(), type=pa.list_(pa.float32())),
            "observation.state": pa.array(states_all.tolist(), type=pa.list_(pa.float32())),
            "episode_index": pa.array(ep_col, type=pa.int64()),
        }),
        data_file,
    )

    # One mp4 per camera holding every episode back to back (the v3.0 packing).
    for ci, cam in enumerate(cams):
        path = root / "videos" / cam / "chunk-000" / "file-000.mp4"
        with prepare_data.VideoWriter(path, fps, (480, 640)) as w:
            for ep in range(n_episodes):
                for f in range(n_frames):
                    frame = np.zeros((480, 640, 3), np.uint8)
                    frame[..., ci] = 40 + 60 * ep      # channel identifies the camera
                    w.write(frame)

    ep_rows = {
        "episode_index": list(range(n_episodes)),
        "tasks": [["Pick the bottle then place it to the basket."] for _ in range(n_episodes)],
        "length": [n_frames] * n_episodes,
        "data/chunk_index": [0] * n_episodes,
        "data/file_index": [0] * n_episodes,
        "dataset_from_index": [ep * n_frames for ep in range(n_episodes)],
        "dataset_to_index": [(ep + 1) * n_frames for ep in range(n_episodes)],
    }
    for cam in cams:
        ep_rows[f"videos/{cam}/chunk_index"] = [0] * n_episodes
        ep_rows[f"videos/{cam}/file_index"] = [0] * n_episodes
        ep_rows[f"videos/{cam}/from_timestamp"] = [ep * n_frames / fps for ep in range(n_episodes)]
        ep_rows[f"videos/{cam}/to_timestamp"] = [(ep + 1) * n_frames / fps for ep in range(n_episodes)]

    meta_file = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(ep_rows), meta_file)
    return actions_all, n_frames


def test_conversion_roundtrip():
    print("\n[4] LeRobot v3.0 -> Motus native conversion")
    import torch

    tmp = Path(tempfile.mkdtemp(prefix="motus_test_"))
    try:
        n_episodes, n_frames = 2, 12
        src = tmp / "lerobot" / "fake_task" / "cobotmagic_fake_000"
        actions_all, n_frames = make_fake_lerobot_v3(src, n_episodes, n_frames)

        out_root = tmp / "converted"
        roots = prepare_data.find_dataset_roots(tmp / "lerobot", None)
        check("dataset discovered", len(roots) == 1 and roots[0][0] == "fake_task", roots)

        source = prepare_data.LeRobotSource(roots[0][1])
        check("version parsed", source.version == "v3.0")
        check("fps parsed", source.fps == 25)

        written, qpos_arrays = prepare_data.convert_dataset(
            source=source,
            out_task_dir=out_root / "clean" / "fake_task",
            qpos_source="action",
            target_hw=(360, 320),
            start_index=0,
            limit_episodes=None,
            overwrite=True,
        )
        check("episode count", written == n_episodes, written)

        task_dir = out_root / "clean" / "fake_task"
        for i in range(n_episodes):
            check(f"videos/{i}.mp4 exists", (task_dir / "videos" / f"{i}.mp4").exists())
            check(f"qpos/{i}.pt exists", (task_dir / "qpos" / f"{i}.pt").exists())
            check(f"metas/{i}.txt exists", (task_dir / "metas" / f"{i}.txt").exists())

        qpos0 = torch.load(task_dir / "qpos" / "0.pt")
        check("qpos shape [T,14]", tuple(qpos0.shape) == (n_frames, 14), tuple(qpos0.shape))
        check("qpos matches source actions",
              bool(np.allclose(qpos0.numpy(), actions_all[:n_frames], atol=1e-6)))

        meta_text = (task_dir / "metas" / "0.txt").read_text()
        check("meta carries the scene prefix", meta_text.startswith(prepare_data.SCENE_PREFIX))
        check("meta carries the instruction", "Pick the bottle" in meta_text)

        frames = list(prepare_data.iter_video_frames(task_dir / "videos" / "0.mp4"))
        check("video frame count", len(frames) == n_frames, len(frames))
        check("video is 360x320", frames[0].shape == (360, 320, 3), frames[0].shape)

        # Episode 1's frames must come from the second half of the shared mp4.
        f0 = np.asarray(list(prepare_data.iter_video_frames(task_dir / "videos" / "0.mp4"))[0])
        f1 = np.asarray(list(prepare_data.iter_video_frames(task_dir / "videos" / "1.mp4"))[0])
        check("episodes are sliced apart", int(f0[10, 160, 0]) != int(f1[10, 160, 0]),
              (int(f0[10, 160, 0]), int(f1[10, 160, 0])))

        stat_path = tmp / "stat.json"
        prepare_data.write_stats(qpos_arrays, stat_path, "robosyn")
        import json

        stats = json.loads(stat_path.read_text())
        check("stats key written", "robosyn" in stats and len(stats["robosyn"]["min"]) == 14)
        check("gripper stat captured", abs(stats["robosyn"]["min"][6] - 0.02) < 1e-5,
              stats["robosyn"]["min"][6])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def main():
    test_three_view_geometry()
    test_action_postprocess()
    test_normalization_roundtrip()
    test_conversion_roundtrip()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("All offline checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
