"""LeRobot v2.1 reading utilities for the LiLa-WAM adapter.

LiLa-WAM upstream reads RoboTwin 2.0 HDF5 episodes.  RoboSynChallenge ships
LeRobot v2.1 datasets (``meta/`` + ``data/*.parquet`` + ``videos/*.mp4``), so
this module provides the pieces the adapter needs:

* :class:`LeRobotV21Meta`   -- parse ``meta/info.json`` / ``episodes.jsonl`` /
  ``tasks.jsonl`` and resolve per-episode parquet / video paths.
* :func:`read_episode_table` -- pull ``observation.state`` / ``action`` out of a
  parquet shard as dense float32 arrays.
* :func:`build_frame_cache` -- decode every episode video once into a single
  HDF5 file of JPEG buffers at the training resolution.

The frame cache exists because training samples frames in random order: a full
episode of AV1 video decodes in ~0.2 s, which is fine once but ruinous when it
happens for every item of every batch.  After the one-off conversion the loader
does an O(1) HDF5 read plus a JPEG decode per frame, which is the same access
pattern (and cost) as the upstream RoboTwin HDF5 loader.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

CODEBASE_VERSION = "v2.1"

# Feature keys of a RoboSynChallenge LeRobot dataset.
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
IMAGE_KEY_PREFIX = "observation.images."


def camera_to_feature_key(camera: str) -> str:
    """``cam_high`` -> ``observation.images.cam_high`` (idempotent)."""
    if camera.startswith(IMAGE_KEY_PREFIX):
        return camera
    return IMAGE_KEY_PREFIX + camera


def feature_key_to_camera(key: str) -> str:
    """``observation.images.cam_high`` -> ``cam_high`` (idempotent)."""
    return key[len(IMAGE_KEY_PREFIX):] if key.startswith(IMAGE_KEY_PREFIX) else key


@dataclass
class EpisodeRef:
    """One episode of one dataset root."""

    root: Path
    episode_index: int
    length: int
    task: str            # RoboSynChallenge task name (used for the VTT lookup)
    prompt: str          # natural-language instruction from meta/tasks.jsonl
    parquet_path: Path


@dataclass
class LeRobotV21Meta:
    root: Path
    info: dict
    episodes: list[EpisodeRef] = field(default_factory=list)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, root: str | os.PathLike, task_name: str | None = None) -> "LeRobotV21Meta":
        root = Path(root).expanduser().resolve()
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"{root} is not a LeRobot dataset (missing meta/info.json)"
            )
        info = json.loads(info_path.read_text())

        version = str(info.get("codebase_version", "?"))
        if version != CODEBASE_VERSION:
            raise ValueError(
                f"{root} is LeRobot {version}, but the LiLa-WAM adapter reads {CODEBASE_VERSION}. "
                f"Convert it first: python scripts/convert_lerobot3.0_to_2.1.py --input {root} --output <dst>"
            )

        task_name = task_name or root.name
        prompts = {}
        tasks_path = root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            for line in tasks_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    prompts[int(row["task_index"])] = str(row.get("task", ""))
        default_prompt = prompts.get(0, "")

        episodes: list[EpisodeRef] = []
        episodes_path = root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"missing {episodes_path}")
        for line in episodes_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            if length < 2:
                continue
            row_tasks = row.get("tasks") or []
            prompt = str(row_tasks[0]) if row_tasks else default_prompt
            episodes.append(
                EpisodeRef(
                    root=root,
                    episode_index=episode_index,
                    length=length,
                    task=task_name,
                    prompt=prompt,
                    parquet_path=root / cls._format_path(info["data_path"], info, episode_index),
                )
            )
        if not episodes:
            raise ValueError(f"no usable episodes (length >= 2) found in {root}")
        return cls(root=root, info=info, episodes=episodes)

    # ------------------------------------------------------------- properties
    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def features(self) -> dict:
        return self.info["features"]

    @property
    def camera_keys(self) -> list[str]:
        return [
            key
            for key, spec in self.features.items()
            if key.startswith(IMAGE_KEY_PREFIX) and spec.get("dtype") in ("video", "image")
        ]

    def feature_dim(self, key: str) -> int:
        shape = self.features[key]["shape"]
        return int(np.prod(shape))

    # ------------------------------------------------------------------ paths
    @staticmethod
    def _format_path(template: str, info: dict, episode_index: int, **extra: Any) -> str:
        chunks_size = int(info.get("chunks_size", 1000))
        return template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            **extra,
        )

    def video_path(self, episode_index: int, camera: str) -> Path:
        return self.root / self._format_path(
            self.info["video_path"],
            self.info,
            episode_index,
            video_key=camera_to_feature_key(camera),
        )

    def require_cameras(self, cameras: Sequence[str]) -> list[str]:
        """Validate requested cameras and return their LeRobot feature keys."""
        available = set(self.camera_keys)
        keys = []
        for camera in cameras:
            key = camera_to_feature_key(camera)
            if key not in available:
                raise KeyError(
                    f"camera '{camera}' not in {self.root}; available: "
                    + ", ".join(sorted(feature_key_to_camera(k) for k in available))
                )
            keys.append(key)
        return keys


def read_episode_table(
    parquet_path: Path,
    keys: Sequence[str],
    length: int,
) -> dict[str, np.ndarray]:
    """Read ``keys`` from one episode parquet as dense ``(length, dim)`` float32."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=list(keys))
    out: dict[str, np.ndarray] = {}
    for key in keys:
        column = table.column(key).to_pylist()
        array = np.asarray(column, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        array = array.reshape(array.shape[0], -1)
        if array.shape[0] != length:
            raise ValueError(
                f"{parquet_path}: '{key}' has {array.shape[0]} rows but episodes.jsonl says {length}"
            )
        out[key] = np.ascontiguousarray(array)
    return out


# --------------------------------------------------------------------- cache
def cache_fingerprint(
    metas: Sequence[LeRobotV21Meta],
    cameras: Sequence[str],
    image_size: tuple[int, int],
    jpeg_quality: int,
) -> dict:
    return {
        "roots": [str(meta.root) for meta in metas],
        "tasks": [meta.episodes[0].task for meta in metas],
        "episodes": [len(meta.episodes) for meta in metas],
        "frames": [sum(ep.length for ep in meta.episodes) for meta in metas],
        "cameras": [feature_key_to_camera(c) for c in cameras],
        "image_size": list(image_size),
        "jpeg_quality": int(jpeg_quality),
        "format": 1,
    }


def cache_path_for(cache_dir: str | os.PathLike, image_size: tuple[int, int]) -> Path:
    width, height = image_size
    return Path(cache_dir).expanduser().resolve() / f"frames_{width}x{height}.h5"


def _decode_video_rgb(path: Path, expected: int, image_size: tuple[int, int]) -> list[np.ndarray]:
    """Decode a whole episode video to a list of resized RGB frames."""
    import av
    import cv2

    width, height = image_size
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if (rgb.shape[1], rgb.shape[0]) != (width, height):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    if len(frames) < expected:
        raise ValueError(f"{path}: decoded {len(frames)} frames but the episode has {expected}")
    return frames[:expected]


def build_frame_cache(
    metas: Sequence[LeRobotV21Meta],
    cameras: Sequence[str],
    cache_dir: str | os.PathLike,
    image_size: tuple[int, int] = (320, 240),
    jpeg_quality: int = 92,
    state_key: str = DEFAULT_STATE_KEY,
    action_key: str = DEFAULT_ACTION_KEY,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Decode every episode once into ``<cache_dir>/frames_<W>x<H>.h5``.

    Layout::

        /states                     (N, state_dim) float32
        /actions                    (N, action_dim) float32
        /episode_start              (E,) int64      global index of each episode
        /episode_length             (E,) int64
        /episode_task               (E,) int64      index into attrs["tasks"]
        /frames/<camera>            (N,) vlen uint8 JPEG buffers

    Frames are stored already resized to ``image_size`` so the cache stays small
    and the training-time cost is a single JPEG decode.
    """
    import cv2
    import h5py

    camera_keys = [camera_to_feature_key(c) for c in cameras]
    camera_names = [feature_key_to_camera(c) for c in camera_keys]
    fingerprint = cache_fingerprint(metas, camera_keys, image_size, jpeg_quality)

    out_path = cache_path_for(cache_dir, image_size)
    if out_path.exists() and not overwrite:
        with h5py.File(out_path, "r") as handle:
            existing = json.loads(handle.attrs["fingerprint"])
        if existing == fingerprint:
            return out_path
        raise FileExistsError(
            f"{out_path} was built for a different dataset/camera/size combination. "
            f"Re-run with --overwrite to rebuild it."
        )

    all_episodes: list[EpisodeRef] = []
    meta_by_root: dict[Path, LeRobotV21Meta] = {}
    state_dim = metas[0].feature_dim(state_key)
    action_dim = metas[0].feature_dim(action_key)
    for meta in metas:
        meta.require_cameras(camera_names)
        if (meta.feature_dim(state_key), meta.feature_dim(action_key)) != (state_dim, action_dim):
            raise ValueError(
                f"{meta.root} has ({state_key}, {action_key}) dims "
                f"({meta.feature_dim(state_key)}, {meta.feature_dim(action_key)}) but "
                f"{metas[0].root} has ({state_dim}, {action_dim}); datasets trained together "
                f"must share a schema"
            )
        meta_by_root[meta.root] = meta
        all_episodes.extend(meta.episodes)

    total_frames = sum(ep.length for ep in all_episodes)
    tasks = sorted({ep.task for ep in all_episodes})
    task_to_index = {task: i for i, task in enumerate(tasks)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".h5.partial")
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    iterator: Iterable[EpisodeRef] = all_episodes
    if progress:
        from tqdm import tqdm

        iterator = tqdm(all_episodes, desc="caching episodes", unit="ep")

    with h5py.File(tmp_path, "w") as handle:
        states = handle.create_dataset("states", (total_frames, state_dim), dtype="float32")
        actions = handle.create_dataset("actions", (total_frames, action_dim), dtype="float32")
        episode_start = handle.create_dataset("episode_start", (len(all_episodes),), dtype="int64")
        episode_length = handle.create_dataset("episode_length", (len(all_episodes),), dtype="int64")
        episode_task = handle.create_dataset("episode_task", (len(all_episodes),), dtype="int64")

        vlen = h5py.vlen_dtype(np.dtype("uint8"))
        group = handle.create_group("frames")
        frame_sets = {
            name: group.create_dataset(name, (total_frames,), dtype=vlen)
            for name in camera_names
        }

        offset = 0
        for ep_i, episode in enumerate(iterator):
            meta = meta_by_root[episode.root]
            table = read_episode_table(episode.parquet_path, [state_key, action_key], episode.length)
            states[offset : offset + episode.length] = table[state_key]
            actions[offset : offset + episode.length] = table[action_key]
            episode_start[ep_i] = offset
            episode_length[ep_i] = episode.length
            episode_task[ep_i] = task_to_index[episode.task]

            for camera_name, camera_key in zip(camera_names, camera_keys):
                video = meta.video_path(episode.episode_index, camera_key)
                rgb_frames = _decode_video_rgb(video, episode.length, image_size)
                buffers = []
                for rgb in rgb_frames:
                    ok, encoded = cv2.imencode(
                        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_params
                    )
                    if not ok:
                        raise RuntimeError(f"failed to JPEG-encode a frame of {video}")
                    buffers.append(np.frombuffer(encoded.tobytes(), dtype=np.uint8))
                frame_sets[camera_name][offset : offset + episode.length] = buffers

            offset += episode.length

        handle.attrs["fingerprint"] = json.dumps(fingerprint)
        handle.attrs["tasks"] = json.dumps(tasks)
        handle.attrs["cameras"] = json.dumps(camera_names)
        handle.attrs["image_size"] = json.dumps(list(image_size))
        handle.attrs["state_key"] = state_key
        handle.attrs["action_key"] = action_key
        handle.attrs["prompts"] = json.dumps(
            {ep.task: ep.prompt for ep in all_episodes}
        )

    tmp_path.replace(out_path)
    return out_path


class FrameCache:
    """Read-only handle onto a cache built by :func:`build_frame_cache`.

    The HDF5 file is opened lazily so a single instance can be forked into
    DataLoader workers (each worker ends up with its own file handle).
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"frame cache not found: {self.path}\n"
                f"Build it first: python policy/lila_wam/build_frame_cache.py --config <config.yaml>"
            )
        self._handle = None
        import h5py

        with h5py.File(self.path, "r") as handle:
            self.tasks: list[str] = json.loads(handle.attrs["tasks"])
            self.cameras: list[str] = json.loads(handle.attrs["cameras"])
            self.image_size: tuple[int, int] = tuple(json.loads(handle.attrs["image_size"]))
            self.fingerprint: dict = json.loads(handle.attrs["fingerprint"])
            self.prompts: dict = json.loads(handle.attrs.get("prompts", "{}"))
            self.episode_start = handle["episode_start"][:]
            self.episode_length = handle["episode_length"][:]
            self.episode_task = handle["episode_task"][:]
            self.states = handle["states"][:]
            self.actions = handle["actions"][:]
        self.episode_end = self.episode_start + self.episode_length
        self.total_frames = int(self.episode_end[-1]) if len(self.episode_end) else 0

    @property
    def num_episodes(self) -> int:
        return len(self.episode_start)

    def _file(self):
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self.path, "r")
        return self._handle

    def read_jpeg(self, camera: str, index: int) -> np.ndarray:
        """Decode one cached frame to ``(H, W, 3)`` uint8 RGB."""
        import cv2

        buffer = self._file()["frames"][camera][index]
        bgr = cv2.imdecode(np.asarray(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"corrupt JPEG in cache: camera={camera} index={index}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state
