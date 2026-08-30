"""Data-path tests for the LiLa-WAM adapter.

They build a tiny synthetic LeRobot v2.1 dataset on disk and push it through the
real reader -> frame cache -> torch Dataset path, so the parts most likely to
break silently (episode/frame alignment, chunk padding masks, camera ordering)
are checked end to end. No GPU, no DINOv3 weights, no network.

    python -m pytest policy/lila_wam/tests/test_lila_wam.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.lila_wam.lerobot_v21 import (  # noqa: E402
    FrameCache,
    LeRobotV21Meta,
    build_frame_cache,
    camera_to_feature_key,
)
from policy.lila_wam.lila_dataset import LeRobotV21LilaDataset  # noqa: E402

STATE_DIM = 14
ACTION_DIM = 14
CAMERAS = ["cam_high", "cam_left_wrist"]
WIDTH, HEIGHT = 64, 48
EPISODE_LENGTHS = [9, 7]


def frame_colour(episode: int, frame: int, camera_index: int) -> tuple[int, int, int]:
    """A colour that identifies (episode, frame, camera) unambiguously."""
    return (20 + 25 * frame, 40 + 60 * episode, 90 + 60 * camera_index)


def _write_video(path: Path, colours, fps: int = 25):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        # Near-lossless so the per-frame colour survives for the alignment check.
        stream.codec_context.qmin = 1
        stream.codec_context.qmax = 1
        for colour in colours:
            image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            image[:, :] = colour
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_dataset(root: Path, codebase_version: str = "v2.1") -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fps = 25
    info = {
        "codebase_version": codebase_version,
        "robot_type": "cobotmagic",
        "total_episodes": len(EPISODE_LENGTHS),
        "total_frames": sum(EPISODE_LENGTHS),
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(EPISODE_LENGTHS)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None},
            **{
                camera_to_feature_key(camera): {
                    "dtype": "video",
                    "shape": [HEIGHT, WIDTH, 3],
                    "names": ["height", "width", "channel"],
                }
                for camera in CAMERAS
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "do the thing"}) + "\n"
    )
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": i, "tasks": ["do the thing"], "length": length}) + "\n"
            for i, length in enumerate(EPISODE_LENGTHS)
        )
    )

    global_index = 0
    for episode, length in enumerate(EPISODE_LENGTHS):
        # state[t, 0] == t and action[t, 0] == -t make misalignment obvious.
        state = np.zeros((length, STATE_DIM), dtype=np.float32)
        action = np.zeros((length, ACTION_DIM), dtype=np.float32)
        state[:, 0] = np.arange(length)
        state[:, 1] = episode
        action[:, 0] = -np.arange(length)
        action[:, 1] = episode

        table = pa.table(
            {
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
                "timestamp": pa.array(np.arange(length) / fps, type=pa.float32()),
                "frame_index": pa.array(np.arange(length), type=pa.int64()),
                "episode_index": pa.array([episode] * length, type=pa.int64()),
                "index": pa.array(np.arange(global_index, global_index + length), type=pa.int64()),
                "task_index": pa.array([0] * length, type=pa.int64()),
            }
        )
        parquet_path = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)

        for camera_index, camera in enumerate(CAMERAS):
            _write_video(
                root / "videos" / "chunk-000" / camera_to_feature_key(camera)
                / f"episode_{episode:06d}.mp4",
                [frame_colour(episode, t, camera_index) for t in range(length)],
                fps=fps,
            )
        global_index += length
    return root


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("lerobot_v21") / "click_bell")


@pytest.fixture(scope="module")
def cache_file(dataset_root, tmp_path_factory) -> Path:
    meta = LeRobotV21Meta.load(dataset_root, task_name="click_bell")
    return build_frame_cache(
        [meta],
        CAMERAS,
        cache_dir=tmp_path_factory.mktemp("cache"),
        image_size=(WIDTH, HEIGHT),
        progress=False,
    )


def test_meta_lists_every_episode(dataset_root):
    meta = LeRobotV21Meta.load(dataset_root)
    assert [ep.length for ep in meta.episodes] == EPISODE_LENGTHS
    assert meta.fps == 25
    assert meta.episodes[0].task == "click_bell"          # defaults to the directory name
    assert meta.episodes[0].prompt == "do the thing"
    assert sorted(meta.camera_keys) == sorted(camera_to_feature_key(c) for c in CAMERAS)
    assert meta.feature_dim("action") == ACTION_DIM


def test_v3_dataset_is_rejected_with_a_conversion_hint(tmp_path):
    root = make_dataset(tmp_path / "v30", codebase_version="v3.0")
    with pytest.raises(ValueError, match="convert_lerobot3.0_to_2.1"):
        LeRobotV21Meta.load(root)


def test_unknown_camera_is_rejected(dataset_root):
    meta = LeRobotV21Meta.load(dataset_root)
    with pytest.raises(KeyError, match="cam_right_wrist"):
        meta.require_cameras(["cam_right_wrist"])


def test_cache_preserves_episode_and_frame_alignment(cache_file):
    cache = FrameCache(cache_file)
    assert cache.total_frames == sum(EPISODE_LENGTHS)
    assert list(cache.episode_length) == EPISODE_LENGTHS
    assert cache.cameras == CAMERAS
    assert cache.tasks == ["click_bell"]

    for episode, length in enumerate(EPISODE_LENGTHS):
        start = int(cache.episode_start[episode])
        # State/action rows must line up with the frames of the same index.
        assert cache.states[start : start + length, 0].tolist() == list(range(length))
        assert cache.actions[start : start + length, 0].tolist() == [-t for t in range(length)]
        assert cache.states[start, 1] == episode

        for camera_index, camera in enumerate(CAMERAS):
            for t in (0, length - 1):
                decoded = cache.read_jpeg(camera, start + t).reshape(-1, 3).mean(axis=0)
                expected = np.array(frame_colour(episode, t, camera_index), dtype=np.float32)
                assert np.abs(decoded - expected).max() < 12, (
                    f"episode {episode} frame {t} camera {camera}: "
                    f"decoded {decoded} != expected {expected}"
                )


def _make_dataset(cache_file, chunk: int = 4, **kwargs) -> LeRobotV21LilaDataset:
    return LeRobotV21LilaDataset(
        cache_path=cache_file,
        indices_config={
            "state_indices": [0],
            "camera_indices": [0],
            "action_indices": list(range(chunk)),
        },
        camera_names=CAMERAS,
        image_size=(WIDTH, HEIGHT),
        **kwargs,
    )


def test_sample_shapes_and_camera_axis(cache_file):
    dataset = _make_dataset(cache_file, chunk=4)
    assert len(dataset) == sum(EPISODE_LENGTHS)

    sample = dataset[0]
    assert sample["state"].shape == (1, STATE_DIM)
    assert sample["action_sequence"].shape == (4, ACTION_DIM)
    assert sample["pixel_values"].shape == (len(CAMERAS), 3, HEIGHT, WIDTH)
    assert sample["action_mask"].tolist() == [True] * 4
    # ImageNet-normalized, so values leave [0, 1].
    assert sample["pixel_values"].dtype.is_floating_point


def test_action_chunk_is_clamped_and_masked_at_the_episode_end(cache_file):
    chunk = 4
    dataset = _make_dataset(cache_file, chunk=chunk)
    length = EPISODE_LENGTHS[0]
    last = length - 1

    sample = dataset[last]
    # Only the anchor step is real; the rest is padding clamped to the last frame.
    assert sample["action_mask"].tolist() == [True, False, False, False]
    assert sample["action_sequence"][:, 0].tolist() == [float(-last)] * chunk

    # A chunk that fits entirely inside the episode is fully valid.
    inside = dataset[0]
    assert inside["action_sequence"][:, 0].tolist() == [0.0, -1.0, -2.0, -3.0]


def test_samples_never_cross_an_episode_boundary(cache_file):
    dataset = _make_dataset(cache_file, chunk=6)
    first_length = EPISODE_LENGTHS[0]
    sample = dataset[first_length - 1]
    # state[:, 1] carries the episode id: everything must stay in episode 0.
    assert float(sample["state"][0, 1]) == 0.0
    assert set(sample["action_sequence"][:, 1].tolist()) == {0.0}

    sample = dataset[first_length]
    assert float(sample["state"][0, 1]) == 1.0


def test_future_frame_is_offset_and_clamped(cache_file):
    offset = 3
    dataset = _make_dataset(cache_file, chunk=4, use_future_feat=True, future_frame_offset=offset)

    sample = dataset[0]
    assert sample["future_pixel_values"].shape == (3, HEIGHT, WIDTH)

    # Rebuild the expected colour through the cache to compare like with like.
    cache = FrameCache(dataset.cache.path)
    expected = cache.read_jpeg(CAMERAS[0], offset).astype(np.float32)
    got = sample["future_pixel_values"].numpy()
    from policy.lila_wam.lila_dataset import normalize_image

    assert np.abs(got - normalize_image(expected.astype(np.uint8))).max() < 1e-5

    # Past the end of the episode the future frame clamps to the last frame.
    last = EPISODE_LENGTHS[0] - 1
    clamped = dataset[last]["future_pixel_values"].numpy()
    end_frame = normalize_image(cache.read_jpeg(CAMERAS[0], last))
    assert np.abs(clamped - end_frame).max() < 1e-5


def test_missing_task_condition_vector_is_reported(cache_file, tmp_path):
    with pytest.raises(FileNotFoundError, match="precompute_task_cond"):
        _make_dataset(cache_file, task_cond_dir=tmp_path / "empty_task_cond")


def test_task_condition_vector_is_attached(cache_file, tmp_path):
    task_cond_dir = tmp_path / "task_cond"
    (task_cond_dir / "click_bell").mkdir(parents=True)
    vector = np.arange(8, dtype=np.float32)
    np.save(task_cond_dir / "click_bell" / "task_cond.npy", vector)

    dataset = _make_dataset(cache_file, task_cond_dir=task_cond_dir)
    assert dataset[0]["task_cond"].tolist() == vector.tolist()


def test_camera_time_offsets_are_rejected(cache_file):
    with pytest.raises(ValueError, match="camera_indices"):
        LeRobotV21LilaDataset(
            cache_path=cache_file,
            indices_config={
                "state_indices": [0],
                "camera_indices": [-1, 0],
                "action_indices": [0, 1],
            },
            camera_names=CAMERAS,
            image_size=(WIDTH, HEIGHT),
        )


def test_image_size_mismatch_is_reported(cache_file):
    with pytest.raises(ValueError, match="rebuild the cache"):
        LeRobotV21LilaDataset(
            cache_path=cache_file,
            indices_config={
                "state_indices": [0],
                "camera_indices": [0],
                "action_indices": [0, 1],
            },
            camera_names=CAMERAS,
            image_size=(WIDTH * 2, HEIGHT),
        )
