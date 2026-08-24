#!/usr/bin/env python3
"""Align RoboSynChallenge Real datasets to the Sim schema and merge them for co-training.

The published datasets are *not* directly mergeable::

                        Sim (cobotmagic_Sim_*)          Real (cobotmagic_Real_*)
    robot_type          cobotmagic                      aloha
    fps                 25                              10
    state / action      14  (J1..J7 per arm)            32  (J1..J6, gripper, ee_pos3, rot6d per arm)
    image keys          observation.images.cam_*        observation.images.cam_*
                        (Sim_click_bell is older:       shape [3,480,640] (CHW), h264
                         cam_*.color + observation.qpos)
    image shape/codec   [480,640,3] (HWC), av1
    extras              observation.qvel / qf, task     -
                        specific *_pose columns

``lerobot``'s ``merge`` (``aggregate_datasets``) requires *identical* fps, robot_type and
feature dicts, so the real data has to be rewritten into the sim schema first.  That is what
this script does, in one pass, emitting a ready-to-train LeRobot **v2.1** dataset:

  * state/action  32 -> 14 by keeping ``[0:7]`` (left J1..J6 + gripper) and ``[16:23]``
    (right J1..J6 + gripper); the joint order and units (rad, gripper normalised to [0,1])
    already agree with sim.
  * temporal resampling 10 Hz -> 25 Hz (linear interpolation of the joint targets, video
    re-timed with ffmpeg) so one 50-step action chunk spans the same 2 s in both domains.
  * videos re-encoded to the sim codec/pix_fmt and truncated to the exact frame count.
  * sim-only columns (qvel, qf, ``*_pose``) dropped, legacy sim keys renamed, image feature
    metadata unified, robot_type unified.
  * per-episode stats sliced/recomputed and aggregated into ``meta/stats.json``.

Example::

    python scripts/prepare_sim_real_cotrain.py \
        --sim-root  /data/cobotmagic_Sim_sample_loading \
        --real-root /data/cobotmagic_Real_sample_loading \
        --out       policy/pi05/training_data/RoboSynChallenge/cobotmagic_Mix_sample_loading \
        --real-repeat 5 --jobs 8

Then point ``pi05_base_robosynchallenge_full`` at the merged repo id, re-run
``compute_norm_stats.py`` and train.  See ``docs/tutorials/policy/pi05.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Columns every LeRobot v2.1 episode parquet must carry.
INDEX_COLUMNS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
# Canonical (sim) names of the payload columns kept in the merged dataset.
STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
# Legacy sim key aliases (cobotmagic_Sim_click_bell was published with the older naming).
SIM_ALIASES = {
    "observation.qpos": STATE_KEY,
    "cam_high.color": "observation.images.cam_high",
    "cam_left_wrist.color": "observation.images.cam_left_wrist",
    "cam_right_wrist.color": "observation.images.cam_right_wrist",
}
STAT_KEYS = ["min", "max", "mean", "std", "count"]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def chunk_of(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def data_path(root: Path, info: dict, episode_index: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=chunk_of(episode_index, info["chunks_size"]), episode_index=episode_index
    )


def video_path(root: Path, info: dict, video_key: str, episode_index: int) -> Path:
    return root / info["video_path"].format(
        episode_chunk=chunk_of(episode_index, info["chunks_size"]),
        video_key=video_key,
        episode_index=episode_index,
    )


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # cross-device: fall back to a copy
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    shutil.copy2(src, dst)


def col_to_array(table: pa.Table, name: str) -> np.ndarray:
    """Read a (possibly list-typed) parquet column as a 2-D float array."""
    values = table.column(name).to_pylist()
    return np.asarray(values, dtype=np.float32).reshape(len(values), -1)


# --------------------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------------------
def numeric_stats(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def slice_stats(stats: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
    out = {}
    for key in STAT_KEYS:
        if key == "count":
            out[key] = list(stats[key])
        else:
            out[key] = np.asarray(stats[key], dtype=np.float64)[keep].tolist()
    return out


def aggregate_stats(per_episode: list[dict[str, Any]]) -> dict[str, Any]:
    """Weighted aggregation of per-episode stats into dataset-level stats."""
    keys = per_episode[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        entries = [ep[key] for ep in per_episode if key in ep]
        counts = np.asarray([float(np.asarray(e["count"]).reshape(-1)[0]) for e in entries])
        total = counts.sum()
        mins = np.stack([np.asarray(e["min"], dtype=np.float64) for e in entries])
        maxs = np.stack([np.asarray(e["max"], dtype=np.float64) for e in entries])
        means = np.stack([np.asarray(e["mean"], dtype=np.float64) for e in entries])
        stds = np.stack([np.asarray(e["std"], dtype=np.float64) for e in entries])
        weights = counts.reshape((-1,) + (1,) * (means.ndim - 1)) / total
        mean = (means * weights).sum(axis=0)
        # variance of the mixture = E[var] + var of the per-episode means
        var = ((stds**2 + (means - mean) ** 2) * weights).sum(axis=0)
        out[key] = {
            "min": mins.min(axis=0).tolist(),
            "max": maxs.max(axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [int(total)],
        }
    return out


def keep_stat_keys(stats: dict[str, Any]) -> dict[str, Any]:
    """Drop quantile entries so sim (which has q01..q99) and real stats stay consistent."""
    return {k: v for k, v in stats.items() if k in STAT_KEYS}


# --------------------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------------------
def ffmpeg_encoder(codec: str) -> list[str]:
    if codec == "av1":
        return ["-c:v", "libsvtav1", "-crf", "30", "-preset", "8"]
    if codec == "h264":
        return ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast"]
    raise ValueError(f"Unsupported target codec: {codec}")


def resample_video(src: Path, dst: Path, out_fps: int, num_frames: int, codec: str, pix_fmt: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"fps={out_fps}",
        "-frames:v", str(num_frames),
        *ffmpeg_encoder(codec),
        "-pix_fmt", pix_fmt,
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    frames = probe_frame_count(dst)
    if frames < num_frames:
        raise RuntimeError(f"{dst} has {frames} frames, expected {num_frames} (source {src})")


def probe_frame_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


# --------------------------------------------------------------------------------------
# writing episodes
# --------------------------------------------------------------------------------------
def write_episode_parquet(
    out_root: Path,
    info: dict,
    episode_index: int,
    state: np.ndarray,
    action: np.ndarray,
    task_index: int,
    index_offset: int,
    fps: int,
) -> None:
    length = state.shape[0]
    table = pa.table({
        STATE_KEY: pa.array([row.tolist() for row in state.astype(np.float32)],
                            type=pa.list_(pa.float32())),
        ACTION_KEY: pa.array([row.tolist() for row in action.astype(np.float32)],
                             type=pa.list_(pa.float32())),
        "timestamp": pa.array((np.arange(length) / fps).astype(np.float32), type=pa.float32()),
        "frame_index": pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
        "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
        "index": pa.array(np.arange(index_offset, index_offset + length, dtype=np.int64), type=pa.int64()),
        "task_index": pa.array(np.full(length, task_index, dtype=np.int64), type=pa.int64()),
    })
    out_path = data_path(out_root, info, episode_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def resample_to_fps(values: np.ndarray, src_fps: int, dst_fps: int) -> np.ndarray:
    """Linear interpolation of a (T, D) trajectory from src_fps onto a dst_fps grid."""
    length = values.shape[0]
    src_t = np.arange(length) / src_fps
    out_len = int(math.floor((length - 1) * dst_fps / src_fps)) + 1
    dst_t = np.arange(out_len) / dst_fps
    out = np.empty((out_len, values.shape[1]), dtype=np.float64)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(dst_t, src_t, values[:, dim])
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_index_map(spec: str) -> np.ndarray:
    keep: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            start, stop = part.split(":")
            keep.extend(range(int(start), int(stop)))
        else:
            keep.append(int(part))
    return np.asarray(keep, dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-root", type=Path, required=True, help="LeRobot v2.1 sim dataset directory")
    parser.add_argument("--real-root", type=Path, required=True, help="LeRobot v2.1 real dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory of the merged dataset")
    parser.add_argument("--sim-episodes", type=int, default=None, help="use only the first N sim episodes")
    parser.add_argument("--real-episodes", type=int, default=None, help="use only the first N real episodes")
    parser.add_argument("--real-repeat", type=int, default=1,
                        help="duplicate every real episode N times to up-weight real data in the mixture")
    parser.add_argument("--real-state-map", default="0:7,16:23",
                        help="indices of the 32-dim real state/action kept as the 14-dim sim vector")
    parser.add_argument("--real-action", choices=["next-state", "as-is"], default="next-state",
                        help="'next-state' rebuilds the action as the next resampled state (sim convention); "
                             "'as-is' interpolates the recorded action column")
    parser.add_argument("--video-mode", choices=["hardlink", "copy", "symlink"], default="hardlink",
                        help="how sim videos are placed in the output (real videos are always re-encoded)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists():
        if not args.overwrite:
            print(f"error: {args.out} already exists (use --overwrite)", file=sys.stderr)
            return 1
        shutil.rmtree(args.out)

    sim_info = json.loads((args.sim_root / "meta" / "info.json").read_text())
    real_info = json.loads((args.real_root / "meta" / "info.json").read_text())
    fps = int(sim_info["fps"])
    real_fps = int(real_info["fps"])
    chunks_size = int(sim_info["chunks_size"])

    # ---- canonical feature dict, taken from the sim side ------------------------------
    sim_features = {SIM_ALIASES.get(k, k): v for k, v in sim_info["features"].items()}
    missing = [k for k in [STATE_KEY, ACTION_KEY, *IMAGE_KEYS] if k not in sim_features]
    if missing:
        print(f"error: sim dataset is missing {missing}", file=sys.stderr)
        return 1
    features = {k: sim_features[k] for k in [STATE_KEY, ACTION_KEY, *IMAGE_KEYS, *INDEX_COLUMNS]}
    video_codec = features[IMAGE_KEYS[0]]["info"]["video.codec"]
    pix_fmt = features[IMAGE_KEYS[0]]["info"].get("video.pix_fmt", "yuv420p")

    out_info = {
        "codebase_version": "v2.1",
        "robot_type": sim_info.get("robot_type", "cobotmagic"),
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {},
        "data_path": sim_info["data_path"],
        "video_path": sim_info["video_path"],
        "features": features,
    }

    # ---- tasks (sim and real ship the same instruction text per task) -----------------
    sim_tasks = {r["task_index"]: r["task"] for r in read_jsonl(args.sim_root / "meta" / "tasks.jsonl")}
    real_tasks = {r["task_index"]: r["task"] for r in read_jsonl(args.real_root / "meta" / "tasks.jsonl")}
    task_to_index: dict[str, int] = {}
    for task in list(sim_tasks.values()) + list(real_tasks.values()):
        task_to_index.setdefault(task, len(task_to_index))

    sim_eps = read_jsonl(args.sim_root / "meta" / "episodes.jsonl")
    real_eps = read_jsonl(args.real_root / "meta" / "episodes.jsonl")
    if args.sim_episodes is not None:
        sim_eps = sim_eps[: args.sim_episodes]
    if args.real_episodes is not None:
        real_eps = real_eps[: args.real_episodes]

    sim_stats = {r["episode_index"]: r["stats"]
                 for r in read_jsonl(args.sim_root / "meta" / "episodes_stats.jsonl")}
    real_stats = {r["episode_index"]: r["stats"]
                  for r in read_jsonl(args.real_root / "meta" / "episodes_stats.jsonl")}

    sim_image_keys = {k: SIM_ALIASES.get(k, k) for k in sim_info["features"]
                      if sim_info["features"][k]["dtype"] == "video"}
    sim_state_col = "observation.qpos" if "observation.qpos" in sim_info["features"] else STATE_KEY
    keep = parse_index_map(args.real_state_map)
    state_dim = features[STATE_KEY]["shape"][0]
    if keep.size != state_dim:
        print(f"error: --real-state-map selects {keep.size} dims, sim state has {state_dim}", file=sys.stderr)
        return 1

    out_episodes: list[dict] = []
    out_stats: list[dict] = []
    index_offset = 0
    next_ep = 0
    video_jobs: list[tuple] = []

    # ---- sim episodes: drop extra columns, keep frames/videos untouched ---------------
    for ep in sim_eps:
        src_ep = ep["episode_index"]
        table = pq.read_table(data_path(args.sim_root, sim_info, src_ep))
        state = col_to_array(table, sim_state_col)
        action = col_to_array(table, ACTION_KEY)
        task = ep["tasks"][0]
        write_episode_parquet(args.out, out_info, next_ep, state, action,
                              task_to_index[task], index_offset, fps)
        for src_key, dst_key in sim_image_keys.items():
            video_jobs.append(("link",
                               video_path(args.sim_root, sim_info, src_key, src_ep),
                               video_path(args.out, out_info, dst_key, next_ep)))
        stats = {SIM_ALIASES.get(k, k): keep_stat_keys(v) for k, v in sim_stats[src_ep].items()}
        stats = {k: v for k, v in stats.items() if k in features}
        out_stats.append({"episode_index": next_ep, "stats": stats})
        out_episodes.append({"episode_index": next_ep, "tasks": [task], "length": int(state.shape[0])})
        index_offset += state.shape[0]
        next_ep += 1

    n_sim_frames = index_offset
    print(f"sim : {len(sim_eps)} episodes, {n_sim_frames} frames @ {fps} Hz")

    # ---- real episodes: slice, resample, re-encode ------------------------------------
    real_first_copy: dict[int, int] = {}  # source episode -> first output episode (for repeats)
    for repeat in range(args.real_repeat):
        for ep in real_eps:
            src_ep = ep["episode_index"]
            table = pq.read_table(data_path(args.real_root, real_info, src_ep))
            state32 = col_to_array(table, STATE_KEY)
            action32 = col_to_array(table, ACTION_KEY)
            state = resample_to_fps(state32[:, keep], real_fps, fps)
            if args.real_action == "next-state":
                action = np.concatenate([state[1:], state[-1:]], axis=0)
            else:
                action = resample_to_fps(action32[:, keep], real_fps, fps)
            length = state.shape[0]
            task = ep["tasks"][0]
            write_episode_parquet(args.out, out_info, next_ep, state, action,
                                  task_to_index[task], index_offset, fps)

            for key in IMAGE_KEYS:
                dst = video_path(args.out, out_info, key, next_ep)
                if src_ep in real_first_copy:
                    video_jobs.append(("link",
                                       video_path(args.out, out_info, key, real_first_copy[src_ep]),
                                       dst))
                else:
                    video_jobs.append(("encode",
                                       video_path(args.real_root, real_info, key, src_ep),
                                       dst, length))
            real_first_copy.setdefault(src_ep, next_ep)

            stats = {k: keep_stat_keys(v) for k, v in real_stats[src_ep].items() if k in features}
            stats[STATE_KEY] = numeric_stats(state)
            stats[ACTION_KEY] = numeric_stats(action)
            stats["timestamp"] = numeric_stats(np.arange(length) / fps)
            stats["frame_index"] = numeric_stats(np.arange(length))
            stats["episode_index"] = numeric_stats(np.full(length, next_ep))
            stats["index"] = numeric_stats(np.arange(index_offset, index_offset + length))
            stats["task_index"] = numeric_stats(np.full(length, task_to_index[task]))
            out_stats.append({"episode_index": next_ep, "stats": stats})
            out_episodes.append({"episode_index": next_ep, "tasks": [task], "length": length})
            index_offset += length
            next_ep += 1

    print(f"real: {len(real_eps) * args.real_repeat} episodes "
          f"({len(real_eps)} unique x{args.real_repeat}), {index_offset - n_sim_frames} frames "
          f"after {real_fps}->{fps} Hz resampling")

    # ---- videos ----------------------------------------------------------------------
    def run_job(job):
        if job[0] == "link":
            link_or_copy(job[1], job[2], args.video_mode if job[1].is_relative_to(args.sim_root) else "hardlink")
        else:
            resample_video(job[1], job[2], fps, job[3], video_codec, pix_fmt)

    # Encodes first: the copies made for --real-repeat link against the freshly encoded files.
    encode_jobs = [j for j in video_jobs if j[0] == "encode"]
    link_jobs = [j for j in video_jobs if j[0] == "link"]
    print(f"videos: {len(link_jobs)} linked/copied, {len(encode_jobs)} re-encoded to {video_codec}@{fps} "
          f"(jobs={args.jobs})")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for _ in pool.map(run_job, encode_jobs):
            pass
        for _ in pool.map(run_job, link_jobs):
            pass

    # ---- metadata --------------------------------------------------------------------
    out_info["total_episodes"] = next_ep
    out_info["total_frames"] = index_offset
    out_info["total_tasks"] = len(task_to_index)
    out_info["total_videos"] = next_ep * len(IMAGE_KEYS)
    out_info["total_chunks"] = chunk_of(max(next_ep - 1, 0), chunks_size) + 1
    out_info["splits"] = {"train": f"0:{next_ep}"}

    meta = args.out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(out_info, indent=4))
    write_jsonl(meta / "tasks.jsonl",
                [{"task_index": i, "task": t} for t, i in sorted(task_to_index.items(), key=lambda kv: kv[1])])
    write_jsonl(meta / "episodes.jsonl", out_episodes)
    write_jsonl(meta / "episodes_stats.jsonl", out_stats)
    (meta / "stats.json").write_text(json.dumps(aggregate_stats([e["stats"] for e in out_stats]), indent=4))

    print(f"\nmerged dataset written to {args.out}")
    print(f"  {next_ep} episodes / {index_offset} frames / {len(task_to_index)} task(s) @ {fps} Hz")
    print("  next: set repo_id in policy/pi05/src/openpi/training/config.py, then")
    print("        uv run scripts/compute_norm_stats.py --config-name <your_config>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
