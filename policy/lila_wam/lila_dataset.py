"""LeRobot v2.1 dataset for LiLa-WAM.

Drop-in replacement for upstream ``dataloader/dataset.py``: it emits exactly the
batch contract ``VLAWrapper.forward`` consumes, but sources frames/state/actions
from a RoboSynChallenge LeRobot v2.1 dataset instead of RoboTwin 2.0 HDF5.

Differences from upstream that the rest of the adapter has to know about:

* ``pixel_values`` always carries a camera axis ``(C, 3, H, W)``.  With a single
  camera the wrapper squeezes it away and the model sees exactly what upstream
  feeds it; with several cameras their DINOv3 tokens are concatenated.
* ``state`` is the 14-dim joint vector (``observation.state``), not RoboTwin's
  16-dim endpose, so ``common.state_dim`` must be 14 in the config.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.utils.data as data

from policy.lila_wam._bootstrap import resolve_path
from policy.lila_wam.lerobot_v21 import (
    DEFAULT_ACTION_KEY,
    DEFAULT_STATE_KEY,
    FrameCache,
    LeRobotV21Meta,
    build_frame_cache,
    cache_path_for,
    feature_key_to_camera,
)

logger = logging.getLogger(__name__)

# DINOv3 (like DINOv2) expects ImageNet-normalized input.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` uint8 RGB -> ``(3, H, W)`` float32, ImageNet normalized."""
    out = img.astype(np.float32) / 255.0
    out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(out, (2, 0, 1))


def resolve_dataset_roots(config) -> list[tuple[Path, str]]:
    """``dataset.dataset_dir`` (+ optional ``dataset.task_names``) -> [(root, task)]."""
    raw = config.dataset.dataset_dir
    roots = [raw] if isinstance(raw, str) else list(raw)
    names = config.dataset.get("task_names", None)
    names = list(names) if names else [Path(str(r)).name for r in roots]
    if len(names) != len(roots):
        raise ValueError(
            f"dataset.task_names has {len(names)} entries but dataset.dataset_dir has {len(roots)}"
        )
    return [(resolve_path(root), str(name)) for root, name in zip(roots, names)]


def load_metas(config) -> list[LeRobotV21Meta]:
    return [LeRobotV21Meta.load(root, task_name=task) for root, task in resolve_dataset_roots(config)]


def resolve_cache_dir(config) -> Path:
    cache_dir = config.dataset.get("cache_dir", None)
    if cache_dir:
        return resolve_path(cache_dir)
    roots = resolve_dataset_roots(config)
    stem = "_".join(task for _, task in roots)
    return Path(__file__).resolve().parent / "cache" / stem


def ensure_frame_cache(config, overwrite: bool = False, progress: bool = True) -> Path:
    """Build the JPEG frame cache if it is missing, and return its path."""
    metas = load_metas(config)
    cameras = [str(c) for c in config.dataset.camera_names]
    image_size = tuple(int(v) for v in config.dataset.image_size)
    cache_dir = resolve_cache_dir(config)
    path = cache_path_for(cache_dir, image_size)
    if path.exists() and not overwrite:
        return path
    return build_frame_cache(
        metas,
        cameras,
        cache_dir=cache_dir,
        image_size=image_size,
        jpeg_quality=int(config.dataset.get("jpeg_quality", 92)),
        state_key=str(config.dataset.get("state_key", DEFAULT_STATE_KEY)),
        action_key=str(config.dataset.get("action_key", DEFAULT_ACTION_KEY)),
        overwrite=overwrite,
        progress=progress,
    )


class LeRobotV21LilaDataset(data.Dataset):
    """Flat frame-indexed view over one or more LeRobot v2.1 datasets."""

    def __init__(
        self,
        cache_path: str | Path,
        indices_config: dict[str, Sequence[int]],
        camera_names: Sequence[str],
        image_size: tuple[int, int] = (320, 240),
        task_cond_dir: str | Path | None = None,
        use_future_feat: bool = False,
        future_frame_offset: int | None = None,
        image_aug: bool = False,
        val: bool = False,
    ):
        if "state_indices" not in indices_config or "action_indices" not in indices_config:
            raise ValueError("indices_config needs 'state_indices' and 'action_indices'")

        self.cache = FrameCache(cache_path)
        self.camera_names = [feature_key_to_camera(str(c)) for c in camera_names]
        missing = [c for c in self.camera_names if c not in self.cache.cameras]
        if missing:
            raise KeyError(
                f"frame cache {self.cache.path} has cameras {self.cache.cameras}, "
                f"but the config asks for {missing}. Rebuild the cache with --overwrite."
            )

        self.image_size = tuple(int(v) for v in image_size)
        if self.image_size != tuple(self.cache.image_size):
            raise ValueError(
                f"config image_size {self.image_size} != cache image_size "
                f"{tuple(self.cache.image_size)}; rebuild the cache with --overwrite"
            )

        self.state_offsets = list(int(v) for v in indices_config["state_indices"])
        self.action_offsets = list(int(v) for v in indices_config["action_indices"])
        camera_offsets = list(int(v) for v in indices_config.get("camera_indices", [0]))
        if camera_offsets != [0]:
            raise ValueError(
                f"camera_indices must be [0] for the LeRobot adapter (got {camera_offsets}); "
                f"the extra axis of pixel_values is used for multiple cameras, not for time"
            )
        self.chunk_size = len(self.action_offsets)

        self.use_future_feat = use_future_feat
        self.future_frame_offset = (
            int(future_frame_offset) if future_frame_offset is not None else self.chunk_size
        )

        self.task_cond_dir = resolve_path(task_cond_dir) if task_cond_dir else None
        self.use_task_cond = self.task_cond_dir is not None
        self.task_cond_cache: dict[str, torch.Tensor] = {}
        if self.use_task_cond:
            self._load_task_cond_vectors()

        self.aug_pool = []
        if image_aug and not val:
            import torchvision.transforms as T

            self.aug_pool = [
                T.ColorJitter(brightness=0.05),
                T.ColorJitter(contrast=0.05),
                T.ColorJitter(saturation=0.05),
                T.ColorJitter(hue=0.05),
            ]

        logger.info(
            "LeRobot v2.1 dataset ready: %d frames / %d episodes / %d task(s), cameras=%s, "
            "chunk=%d, future_feat=%s(offset=%d)",
            self.cache.total_frames,
            self.cache.num_episodes,
            len(self.cache.tasks),
            self.camera_names,
            self.chunk_size,
            self.use_future_feat,
            self.future_frame_offset,
        )

    # ------------------------------------------------------------------ setup
    def _load_task_cond_vectors(self):
        for task in self.cache.tasks:
            path = self.task_cond_dir / task / "task_cond.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing VTT vector for task '{task}': {path}\n"
                    f"Generate it first: python policy/lila_wam/precompute_task_cond.py --config <config.yaml>"
                )
            self.task_cond_cache[task] = torch.from_numpy(np.load(path).astype(np.float32))
        dims = {v.shape[0] for v in self.task_cond_cache.values()}
        if len(dims) != 1:
            raise ValueError(f"task condition vectors disagree on dimension: {dims}")
        logger.info(
            "loaded %d task condition vectors (dim=%d) from %s",
            len(self.task_cond_cache),
            next(iter(dims)),
            self.task_cond_dir,
        )

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return int(self.cache.total_frames)

    def _episode_of(self, index: int) -> int:
        return int(np.searchsorted(self.cache.episode_end, index, side="right"))

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if not self.aug_pool:
            return img
        from PIL import Image

        pil = Image.fromarray(img)
        for op in random.sample(self.aug_pool, random.choice([1, 2])):
            pil = op(pil)
        return np.asarray(pil)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep = self._episode_of(idx)
        start = int(self.cache.episode_start[ep])
        end = int(self.cache.episode_end[ep])
        anchor = idx

        action_abs = [anchor + off for off in self.action_offsets]
        action_mask = np.array([i < end for i in action_abs], dtype=bool)
        action_rows = [min(max(i, start), end - 1) for i in action_abs]
        state_rows = [min(max(anchor + off, start), end - 1) for off in self.state_offsets]

        result: dict[str, Any] = {
            "state": torch.from_numpy(self.cache.states[state_rows].copy()),
            "action_sequence": torch.from_numpy(self.cache.actions[action_rows].copy()),
            "state_mask": torch.ones(len(state_rows), dtype=torch.bool),
            "action_mask": torch.from_numpy(action_mask),
        }

        frames = []
        for camera in self.camera_names:
            img = self.cache.read_jpeg(camera, anchor)
            frames.append(normalize_image(self._augment(img)))
        # (C, 3, H, W) — C is the camera axis, squeezed by the wrapper when C == 1
        result["pixel_values"] = torch.from_numpy(np.stack(frames, axis=0)).float()

        if self.use_future_feat:
            future_row = min(max(anchor + self.future_frame_offset, start), end - 1)
            future = self.cache.read_jpeg(self.camera_names[0], future_row)
            result["future_pixel_values"] = torch.from_numpy(normalize_image(future)).float()

        if self.use_task_cond:
            task = self.cache.tasks[int(self.cache.episode_task[ep])]
            result["task_cond"] = self.task_cond_cache[task]

        return result


def create_dataset(config, val: bool = False, cache_path: str | Path | None = None):
    """Factory mirroring upstream ``dataloader.dataset.create_dataset``."""
    from omegaconf import OmegaConf

    indices_config = OmegaConf.to_container(config.dataset.indices_config, resolve=True)
    image_size = tuple(int(v) for v in config.dataset.image_size)

    task_cond_dir = None
    if config.model.get("use_task_cond", False):
        task_cond_dir = config.dataset.get("task_cond_dir", None)
        if task_cond_dir is None:
            raise ValueError("model.use_task_cond=true but dataset.task_cond_dir is not set")

    future_cfg = config.model.get("future_feat", {}) or {}
    if cache_path is None:
        cache_path = cache_path_for(resolve_cache_dir(config), image_size)

    return LeRobotV21LilaDataset(
        cache_path=cache_path,
        indices_config=indices_config,
        camera_names=list(config.dataset.camera_names),
        image_size=image_size,
        task_cond_dir=task_cond_dir,
        use_future_feat=bool(future_cfg.get("enabled", False)),
        future_frame_offset=config.dataset.get("future_frame_offset", None),
        image_aug=bool(config.dataset.get("image_aug", False)),
        val=val,
    )
