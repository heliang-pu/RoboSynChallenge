#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# RoboSynChallenge LeRobot dataset -> Motus native ("robotwin") training format.
#
# Why convert instead of using Motus' LeRobot loader:
#   1. The challenge datasets are LeRobot **v3.0** (data/chunk-XXX/file-XXX.parquet,
#      meta/episodes/*.parquet, one mp4 per camera holding many episodes).
#      Motus pins lerobot==0.3.2, which only reads v2.1.
#   2. Motus' three-camera stitch in data/lerobot/lerobot_dataset.py assumes the
#      wrist videos are already half-size (it reconstructs a frame that was split
#      out of a concatenated view). Our three cameras are all 480x640, so that
#      path would build a 960x640 frame instead of the 720x640 the model expects.
#   3. The native format normalises nothing, which matches how the released
#      Motus_robotwin2 checkpoint was trained (see README_INTEGRATION.md).
#
# Output layout (consumed by Motus/data/robotwin2/robotwin_agilex_dataset.py):
#
#   <output_root>/<split>/<task>/
#       videos/{i}.mp4      T-shaped three-view, 360x320 (HxW), fps preserved
#       qpos/{i}.pt         torch float32 tensor [T, 14]
#       metas/{i}.txt       one instruction per line, scene prefix included
#       umt5_wan/{i}.pt     list[Tensor[S, 4096]], index-aligned with metas lines
#
# Usage:
#   python policy/motus/prepare_data.py \
#       --lerobot-root /home/phl/workspace/RoboSynChallenge/lerobot_dataset \
#       --output-root  /home/phl/workspace/data/motus_robosyn \
#       --emit-stats
#   # then, on a box where the T5 encoder can be loaded:
#   python policy/motus/prepare_data.py --output-root ... --t5-only \
#       --wan-path /home/phl/workspace/models/motus/Wan2.2-TI2V-5B
# ----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

POLICY_DIR = Path(__file__).resolve().parent

SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)

CAM_HIGH = "observation.images.cam_high"
CAM_LEFT = "observation.images.cam_left_wrist"
CAM_RIGHT = "observation.images.cam_right_wrist"
CAM_KEYS = (CAM_HIGH, CAM_LEFT, CAM_RIGHT)

logger = logging.getLogger("prepare_data")


# ---------------------------------------------------------------------------
# Frame stitching (must stay identical to motus_model.build_three_view)
# ---------------------------------------------------------------------------
def stitch_three_view(cam_high: np.ndarray, cam_left: np.ndarray, cam_right: np.ndarray) -> np.ndarray:
    import cv2

    top_h, target_w = cam_high.shape[:2]
    bottom_h = top_h // 2
    split_w = target_w // 2
    right_w = target_w - split_w

    left_resized = cv2.resize(cam_left, (split_w, bottom_h))
    right_resized = cv2.resize(cam_right, (right_w, bottom_h))

    out = np.zeros((top_h + bottom_h, target_w, 3), dtype=np.uint8)
    out[:top_h] = cam_high[..., :3]
    out[top_h:, :split_w] = left_resized[..., :3]
    out[top_h:, split_w:] = right_resized[..., :3]
    return out


def resize_keep_aspect(frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Scale to fit target while preserving aspect ratio (no padding here).

    720x640 -> 360x320 is an exact 0.5 downscale, so nothing is distorted and the
    dataset loader's later resize_with_padding to 384x320 only adds black bars.
    """
    import cv2

    th, tw = target_hw
    h, w = frame.shape[:2]
    if (h, w) == (th, tw):
        return frame
    scale = min(th / h, tw / w)
    return cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))


# ---------------------------------------------------------------------------
# Video IO
# ---------------------------------------------------------------------------
def iter_video_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield RGB frames. PyAV first (handles the AV1 streams the challenge ships)."""
    try:
        import av
    except ImportError:
        av = None

    if av is not None:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")
        return

    import imageio.v3 as iio

    for frame in iio.imiter(str(path)):
        yield np.asarray(frame)[..., :3]


class VideoWriter:
    """H.264 writer; PyAV when available, imageio-ffmpeg otherwise."""

    def __init__(self, path: Path, fps: float, size_hw: Tuple[int, int], crf: int = 18):
        self.path = path
        self.fps = fps
        self.height, self.width = size_hw
        self.crf = crf
        self._av_container = None
        self._av_stream = None
        self._imageio_writer = None
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import av

            self._av_container = av.open(str(path), mode="w")
            stream = self._av_container.add_stream("libx264", rate=int(round(fps)))
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf)}
            self._av_stream = stream
        except Exception as exc:
            logger.debug("PyAV writer unavailable (%s); using imageio", exc)
            import imageio

            self._imageio_writer = imageio.get_writer(
                str(path), fps=fps, codec="libx264", quality=None,
                output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
                macro_block_size=1,
            )

    def write(self, frame_rgb: np.ndarray):
        if self._av_stream is not None:
            import av

            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_rgb), format="rgb24")
            for packet in self._av_stream.encode(vf):
                self._av_container.mux(packet)
        else:
            self._imageio_writer.append_data(frame_rgb)

    def close(self):
        if self._av_stream is not None:
            for packet in self._av_stream.encode():
                self._av_container.mux(packet)
            self._av_container.close()
            self._av_container = None
            self._av_stream = None
        elif self._imageio_writer is not None:
            self._imageio_writer.close()
            self._imageio_writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# LeRobot dataset reading
# ---------------------------------------------------------------------------
class LeRobotSource:
    """Minimal reader for LeRobot v2.1 / v3.0 dataset roots (no lerobot import)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        self.version = str(self.info.get("codebase_version", "v2.1"))
        self.fps = float(self.info.get("fps", 25))
        self.features = self.info.get("features", {})
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path = self.info["data_path"]
        self.video_path = self.info["video_path"]
        self.episodes = self._read_episode_meta()

    # -- metadata ---------------------------------------------------------
    def _read_episode_meta(self) -> List[dict]:
        import pyarrow.parquet as pq

        meta_dir = self.root / "meta"
        episode_files = sorted((meta_dir / "episodes").glob("**/*.parquet")) if (meta_dir / "episodes").is_dir() else []

        if episode_files:  # v3.0
            rows: List[dict] = []
            wanted_prefixes = ("episode_index", "tasks", "length", "data/", "videos/", "meta/episodes/")
            for fp in episode_files:
                table = pq.read_table(fp)
                keep = [c for c in table.column_names if c.startswith(wanted_prefixes)]
                data = table.select(keep).to_pydict()
                for i in range(table.num_rows):
                    rows.append({c: data[c][i] for c in keep})
            rows.sort(key=lambda r: r["episode_index"])
            return rows

        jsonl = meta_dir / "episodes.jsonl"      # v2.1
        if jsonl.exists():
            rows = []
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            rows.sort(key=lambda r: r["episode_index"])
            return rows

        raise FileNotFoundError(f"No meta/episodes(.jsonl) under {self.root}")

    def task_of(self, ep: dict) -> str:
        tasks = ep.get("tasks")
        if isinstance(tasks, (list, tuple)) and tasks:
            return str(tasks[0])
        if isinstance(tasks, str):
            return tasks
        raise KeyError(f"episode {ep.get('episode_index')} has no task string")

    # -- paths -------------------------------------------------------------
    def data_file_for(self, ep: dict) -> Path:
        if self.version.startswith("v3"):
            return self.root / self.data_path.format(
                chunk_index=int(ep["data/chunk_index"]), file_index=int(ep["data/file_index"])
            )
        ep_idx = int(ep["episode_index"])
        return self.root / self.data_path.format(
            episode_chunk=ep_idx // self.chunks_size, episode_index=ep_idx
        )

    def video_file_for(self, ep: dict, video_key: str) -> Path:
        if self.version.startswith("v3"):
            return self.root / self.video_path.format(
                video_key=video_key,
                chunk_index=int(ep[f"videos/{video_key}/chunk_index"]),
                file_index=int(ep[f"videos/{video_key}/file_index"]),
            )
        ep_idx = int(ep["episode_index"])
        return self.root / self.video_path.format(
            video_key=video_key, episode_chunk=ep_idx // self.chunks_size, episode_index=ep_idx
        )

    def video_frame_range(self, ep: dict, video_key: str) -> Tuple[int, int]:
        """Frame span of this episode inside its (possibly shared) mp4."""
        if self.version.startswith("v3"):
            start = float(ep[f"videos/{video_key}/from_timestamp"])
            end = float(ep[f"videos/{video_key}/to_timestamp"])
            return int(round(start * self.fps)), int(round(end * self.fps))
        return 0, int(ep.get("length", 0))

    # -- frames ------------------------------------------------------------
    def read_qpos(self, ep: dict, column: str) -> np.ndarray:
        import pyarrow.parquet as pq

        table = pq.read_table(self.data_file_for(ep), columns=[column, "episode_index"])
        col = table.column(column).to_pylist()
        ep_col = table.column("episode_index").to_pylist()
        ep_idx = int(ep["episode_index"])

        if self.version.startswith("v3") and "dataset_from_index" in ep:
            lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            rows = col[lo:hi]
            if any(int(e) != ep_idx for e in ep_col[lo:hi]):
                logger.warning("episode %d row range disagrees with episode_index column", ep_idx)
        else:
            rows = [v for v, e in zip(col, ep_col) if int(e) == ep_idx]

        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected [T, D] for {column}, got {arr.shape}")
        return arr


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert_dataset(
    source: LeRobotSource,
    out_task_dir: Path,
    qpos_source: str,
    target_hw: Tuple[int, int],
    start_index: int,
    limit_episodes: Optional[int],
    overwrite: bool,
) -> Tuple[int, List[np.ndarray]]:
    """Convert one LeRobot dataset dir into <out_task_dir>. Returns (n_written, qpos_arrays)."""
    for sub in ("videos", "qpos", "metas", "umt5_wan"):
        (out_task_dir / sub).mkdir(parents=True, exist_ok=True)

    episodes = source.episodes
    if limit_episodes:
        episodes = episodes[:limit_episodes]

    # Group episodes by the mp4 file they live in, so each file is decoded once.
    groups: Dict[Tuple, List[dict]] = {}
    for ep in episodes:
        key = tuple(str(source.video_file_for(ep, k)) for k in CAM_KEYS)
        groups.setdefault(key, []).append(ep)

    written = 0
    collected: List[np.ndarray] = []

    for cam_paths, group in groups.items():
        group = sorted(group, key=lambda e: source.video_frame_range(e, CAM_HIGH)[0])
        spans = {int(ep["episode_index"]): source.video_frame_range(ep, CAM_HIGH) for ep in group}
        pending = {int(ep["episode_index"]): ep for ep in group}

        # Episode -> output index, assigned before decoding so writers can open lazily.
        out_index = {}
        for ep in group:
            out_index[int(ep["episode_index"])] = start_index + written
            written += 1

        writers: Dict[int, VideoWriter] = {}
        counts: Dict[int, int] = {k: 0 for k in pending}

        iters = [iter_video_frames(Path(p)) for p in cam_paths]
        frame_idx = 0
        try:
            for high, left, right in zip(*iters):
                active = [e for e, (lo, hi) in spans.items() if lo <= frame_idx < hi]
                if active:
                    stitched = resize_keep_aspect(stitch_three_view(high, left, right), target_hw)
                    for ep_idx in active:
                        if ep_idx not in writers:
                            out_path = out_task_dir / "videos" / f"{out_index[ep_idx]}.mp4"
                            if out_path.exists() and not overwrite:
                                logger.info("skip existing %s", out_path)
                            writers[ep_idx] = VideoWriter(
                                out_path, source.fps, stitched.shape[:2]
                            )
                        writers[ep_idx].write(stitched)
                        counts[ep_idx] += 1
                frame_idx += 1
        finally:
            for w in writers.values():
                w.close()

        # qpos + metas alongside each video
        for ep in group:
            ep_idx = int(ep["episode_index"])
            oi = out_index[ep_idx]
            qpos = source.read_qpos(ep, qpos_source)
            n_video = counts.get(ep_idx, 0)
            if n_video == 0:
                raise RuntimeError(
                    f"no video frames decoded for episode {ep_idx} of {source.root}; "
                    "check that the mp4 codec is decodable (AV1 needs PyAV/libdav1d)"
                )
            if n_video != qpos.shape[0]:
                n = min(n_video, qpos.shape[0])
                logger.warning(
                    "episode %d: %d video frames vs %d qpos rows -> truncating to %d",
                    ep_idx, n_video, qpos.shape[0], n,
                )
                qpos = qpos[:n]

            import torch

            torch.save(torch.from_numpy(qpos.astype(np.float32)), out_task_dir / "qpos" / f"{oi}.pt")
            (out_task_dir / "metas" / f"{oi}.txt").write_text(
                f"{SCENE_PREFIX}{source.task_of(ep)}\n", encoding="utf-8"
            )
            collected.append(qpos)
            logger.info("  episode %d -> %s (%d frames)", ep_idx, f"{oi}.mp4", qpos.shape[0])

    return written, collected


# ---------------------------------------------------------------------------
# T5 pre-encoding
# ---------------------------------------------------------------------------
def encode_t5(output_root: Path, split: str, wan_path: Path, device: str, text_len: int = 512):
    """Fill umt5_wan/{i}.pt for every metas/{i}.txt under <output_root>/<split>."""
    import torch

    sys.path.insert(0, str(POLICY_DIR / "Motus" / "inference" / "robotwin" / "Motus" / "bak"))
    from wan.modules.t5 import T5EncoderModel  # noqa: E402

    ckpt = wan_path / "models_t5_umt5-xxl-enc-bf16.pth"
    tok = wan_path / "google" / "umt5-xxl"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {ckpt}")

    logger.info("Loading umT5-xxl encoder on %s", device)
    encoder = T5EncoderModel(
        text_len=text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(ckpt),
        tokenizer_path=str(tok),
    )

    cache: Dict[str, "torch.Tensor"] = {}
    n = 0
    for meta_file in sorted((output_root / split).glob("*/metas/*.txt")):
        out_pt = meta_file.parent.parent / "umt5_wan" / f"{meta_file.stem}.pt"
        if out_pt.exists():
            continue
        lines = [ln.strip() for ln in meta_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        embeddings = []
        for line in lines:
            if line not in cache:
                out = encoder([line], device)
                emb = out[0] if isinstance(out, (list, tuple)) else out
                if emb.dim() == 3:
                    emb = emb.squeeze(0)
                cache[line] = emb.detach().to("cpu")
            embeddings.append(cache[line])
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        # RobotWinTaskDataset indexes this list with the metas line number.
        torch.save(embeddings, out_pt)
        n += 1
    logger.info("Wrote %d umt5_wan files (%d unique instructions)", n, len(cache))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def write_stats(qpos_arrays: Sequence[np.ndarray], stat_path: Path, key: str):
    if not qpos_arrays:
        logger.warning("no qpos collected; skipping stats")
        return
    stacked = np.concatenate([a for a in qpos_arrays if a.size], axis=0)
    entry = {
        "min": np.min(stacked, axis=0).astype(float).tolist(),
        "max": np.max(stacked, axis=0).astype(float).tolist(),
        "q01": np.quantile(stacked, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(stacked, 0.99, axis=0).astype(float).tolist(),
        "action_dim": int(stacked.shape[1]),
        "frame_count": int(stacked.shape[0]),
    }
    stats = {}
    if stat_path.exists():
        stats = json.loads(stat_path.read_text())
    stats[key] = entry
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    stat_path.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote stats[%r] to %s", key, stat_path)
    logger.info("  gripper dim 6:  [%.4f, %.4f]", entry["min"][6], entry["max"][6])
    logger.info("  gripper dim 13: [%.4f, %.4f]", entry["min"][13], entry["max"][13])


# ---------------------------------------------------------------------------
def find_dataset_roots(lerobot_root: Path, tasks: Optional[Sequence[str]]) -> List[Tuple[str, Path]]:
    """Return (task_name, dataset_root) for every LeRobot dataset under lerobot_root."""
    found: List[Tuple[str, Path]] = []
    for info in sorted(lerobot_root.glob("**/meta/info.json")):
        ds_root = info.parent.parent
        # <lerobot_root>/<task>/<dataset>/meta/info.json -> task name is the first level
        try:
            rel = ds_root.relative_to(lerobot_root)
            task = rel.parts[0]
        except ValueError:
            task = ds_root.name
        if tasks and task not in tasks:
            continue
        found.append((task, ds_root))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lerobot-root", type=Path,
                        help="directory containing <task>/<dataset>/meta/info.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="clean", choices=["clean", "randomized"],
                        help="Motus data_mode bucket to write into")
    parser.add_argument("--tasks", nargs="*", default=None, help="subset of task directory names")
    parser.add_argument("--qpos-source", default="action", choices=["action", "observation.state"],
                        help="'action' = recorded command (default, matches what env.step consumes); "
                             "'observation.state' = measured qpos")
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--emit-stats", action="store_true",
                        help="write min/max/q01/q99 into --stat-path")
    parser.add_argument("--stat-path", type=Path, default=POLICY_DIR / "configs" / "stat.json")
    parser.add_argument("--stat-key", default="robosyn")
    parser.add_argument("--encode-t5", action="store_true", help="also fill umt5_wan/ (needs --wan-path)")
    parser.add_argument("--t5-only", action="store_true", help="skip conversion, only fill umt5_wan/")
    parser.add_argument("--wan-path", type=Path, default=None)
    parser.add_argument("--t5-device", default="cuda")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.t5_only:
        if args.lerobot_root is None:
            parser.error("--lerobot-root is required unless --t5-only")
        roots = find_dataset_roots(args.lerobot_root, args.tasks)
        if not roots:
            parser.error(f"no LeRobot datasets found under {args.lerobot_root}")

        logger.info("Found %d dataset(s) under %s", len(roots), args.lerobot_root)
        all_qpos: List[np.ndarray] = []
        per_task_counter: Dict[str, int] = {}

        for task, ds_root in roots:
            logger.info("[%s] %s", task, ds_root)
            source = LeRobotSource(ds_root)
            missing = [k for k in CAM_KEYS if k not in source.features]
            if missing:
                logger.error("  skipping: missing camera features %s", missing)
                continue
            if args.qpos_source not in source.features:
                logger.error("  skipping: no %r feature", args.qpos_source)
                continue

            out_task_dir = args.output_root / args.split / task
            start = per_task_counter.get(task, 0)
            n, qpos_arrays = convert_dataset(
                source=source,
                out_task_dir=out_task_dir,
                qpos_source=args.qpos_source,
                target_hw=(args.video_height, args.video_width),
                start_index=start,
                limit_episodes=args.limit_episodes,
                overwrite=args.overwrite,
            )
            per_task_counter[task] = start + n
            all_qpos.extend(qpos_arrays)
            logger.info("  wrote %d episodes -> %s", n, out_task_dir)

        logger.info("Total: %d episodes across %d tasks",
                    sum(per_task_counter.values()), len(per_task_counter))

        if args.emit_stats:
            write_stats(all_qpos, args.stat_path, args.stat_key)

    if args.encode_t5 or args.t5_only:
        if args.wan_path is None:
            parser.error("--wan-path is required for T5 encoding")
        encode_t5(args.output_root, args.split, args.wan_path, args.t5_device)
    else:
        logger.warning(
            "umt5_wan/ is EMPTY. RobotWinTaskDataset requires it; run this script again with "
            "--t5-only --wan-path <Wan2.2-TI2V-5B> on a machine that can load the T5 encoder."
        )


if __name__ == "__main__":
    main()
