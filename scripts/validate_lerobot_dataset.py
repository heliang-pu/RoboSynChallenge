#!/usr/bin/env python3
"""Strict, training-oriented validation for local LeRobot v2.1/v3.0 datasets.

The command exits non-zero on the first failed gate, so collection/merge queues
can use it as a hard barrier.  Run it with the Python environment used by the
target model to make the final LeRobotDataset check meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq


EXPECTED_VIDEO_KEYS = {
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise ValidationError(f"cannot parse JSON {path}: {exc}") from exc


def load_episodes(root: Path, version: str) -> list[dict[str, Any]]:
    if version == "v3.0":
        paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
        require(paths, "v3.0 dataset has no meta/episodes parquet files")
        records: list[dict[str, Any]] = []
        for path in paths:
            try:
                records.extend(pq.read_table(path).to_pylist())
            except Exception as exc:
                raise ValidationError(f"unreadable episode metadata {path}: {exc}") from exc
        return records

    episodes_path = root / "meta" / "episodes.jsonl"
    require(episodes_path.is_file(), f"missing file: {episodes_path}")
    records = []
    for line_number, line in enumerate(episodes_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except Exception as exc:
            raise ValidationError(
                f"invalid JSON in {episodes_path}:{line_number}: {exc}"
            ) from exc
    return records


def validate_parquet_and_counts(
    root: Path, info: dict[str, Any], episodes: list[dict[str, Any]]
) -> tuple[int, int]:
    meta_episodes = root / "meta" / "episodes"
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    meta_paths = [
        path
        for path in sorted((root / "meta").glob("**/*.parquet"))
        if meta_episodes not in path.parents
    ]
    require(data_paths, "dataset has no data parquet files")

    # Read every non-episode metadata parquet in full. Episode metadata was read
    # in full by load_episodes().
    for path in meta_paths:
        try:
            pq.read_table(path)
        except Exception as exc:
            raise ValidationError(f"unreadable parquet {path}: {exc}") from exc

    expected_by_episode = {
        int(record["episode_index"]): int(record["length"]) for record in episodes
    }
    require(
        len(expected_by_episode) == len(episodes),
        "episode metadata contains duplicate episode_index values",
    )

    frame_counts: Counter[int] = Counter()
    total_rows = 0
    next_global_index = 0
    required_columns = {"episode_index", "frame_index", "index", "action", "observation.state"}

    for path in data_paths:
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise ValidationError(f"unreadable parquet {path}: {exc}") from exc

        missing = required_columns - set(table.column_names)
        require(not missing, f"{path} is missing columns: {sorted(missing)}")
        rows = table.num_rows
        total_rows += rows

        episode_values = np.asarray(table["episode_index"].to_numpy())
        unique, counts = np.unique(episode_values, return_counts=True)
        frame_counts.update({int(ep): int(count) for ep, count in zip(unique, counts)})

        global_indices = np.asarray(table["index"].to_numpy())
        expected_indices = np.arange(next_global_index, next_global_index + rows)
        require(
            np.array_equal(global_indices, expected_indices),
            f"global index is not contiguous at {path}",
        )
        next_global_index += rows

    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    require(total_episodes == len(episodes), (
        f"info total_episodes={total_episodes}, episode metadata rows={len(episodes)}"
    ))
    require(sorted(expected_by_episode) == list(range(total_episodes)), (
        "episode_index values are not contiguous from 0 to total_episodes-1"
    ))

    summed_lengths = sum(expected_by_episode.values())
    require(total_frames == summed_lengths, (
        f"info total_frames={total_frames}, sum(episode lengths)={summed_lengths}"
    ))
    require(total_frames == total_rows, (
        f"info total_frames={total_frames}, parquet rows={total_rows}"
    ))
    require(dict(frame_counts) == expected_by_episode, (
        "per-episode parquet frame counts do not match episode metadata lengths"
    ))
    return total_episodes, total_frames


def video_keys(info: dict[str, Any]) -> set[str]:
    return {
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") in {"video", "image"} and key.startswith("observation.images.")
    }


def format_path(template: str, **values: Any) -> Path:
    try:
        return Path(template.format(**values))
    except KeyError as exc:
        raise ValidationError(f"unsupported path placeholder {exc} in {template!r}") from exc


def expected_video_files(
    root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    keys: set[str],
) -> set[Path]:
    template = info.get("video_path")
    require(isinstance(template, str), "info.json has no video_path template")
    chunks_size = int(info.get("chunks_size", 1000))
    fps = float(info["fps"])
    paths: set[Path] = set()

    for record in episodes:
        episode_index = int(record["episode_index"])
        length = int(record["length"])
        for key in keys:
            prefix = f"videos/{key}"
            if info.get("codebase_version") == "v3.0":
                chunk_col = f"{prefix}/chunk_index"
                file_col = f"{prefix}/file_index"
                from_col = f"{prefix}/from_timestamp"
                to_col = f"{prefix}/to_timestamp"
                for column in (chunk_col, file_col, from_col, to_col):
                    require(column in record, f"episode {episode_index} is missing {column}")
                duration = float(record[to_col]) - float(record[from_col])
                require(
                    math.isclose(duration, length / fps, abs_tol=1.5 / fps),
                    f"episode {episode_index} {key} duration {duration:.6f}s "
                    f"does not match {length} frames at {fps:g} fps",
                )
                relative = format_path(
                    template,
                    video_key=key,
                    chunk_index=int(record[chunk_col]),
                    file_index=int(record[file_col]),
                    episode_index=episode_index,
                    episode_chunk=episode_index // chunks_size,
                )
            else:
                relative = format_path(
                    template,
                    video_key=key,
                    episode_index=episode_index,
                    episode_chunk=episode_index // chunks_size,
                    chunk_index=episode_index // chunks_size,
                    file_index=episode_index,
                )
            paths.add(root / relative)
    return paths


def decode_first_and_last(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty video: {path}")
    try:
        with av.open(str(path)) as container:
            require(bool(container.streams.video), f"no video stream: {path}")
            stream = container.streams.video[0]
            first_frame = next(container.decode(stream), None)
            require(first_frame is not None, f"cannot decode first frame: {path}")
            first_shape = first_frame.to_ndarray(format="rgb24").shape

        # Reopen before seeking so decoder state from the first frame cannot
        # affect the last-frame check.
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                seek_pts = max(0, int(stream.duration) - max(1, int(2 / float(stream.time_base))))
                container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            elif container.duration is not None:
                seek_us = max(0, int(container.duration) - 2_000_000)
                container.seek(seek_us, backward=True, any_frame=False)
            last_frame = None
            for frame in container.decode(stream):
                last_frame = frame
            require(last_frame is not None, f"cannot decode last frame: {path}")
            last_shape = last_frame.to_ndarray(format="rgb24").shape
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"video decode failed for {path}: {exc}") from exc

    require(first_shape == last_shape and len(first_shape) == 3, (
        f"video first/last frame shapes differ for {path}: {first_shape} vs {last_shape}"
    ))
    return first_shape, last_shape


def validate_videos(
    root: Path, info: dict[str, Any], episodes: list[dict[str, Any]]
) -> tuple[int, tuple[int, ...]]:
    keys = video_keys(info)
    require(keys == EXPECTED_VIDEO_KEYS, (
        f"expected exactly three camera keys {sorted(EXPECTED_VIDEO_KEYS)}, got {sorted(keys)}"
    ))
    paths = expected_video_files(root, info, episodes, keys)
    require(paths, "no video files are referenced by metadata")
    shape: tuple[int, ...] | None = None
    for index, path in enumerate(sorted(paths), 1):
        first_shape, _ = decode_first_and_last(path)
        if shape is None:
            shape = first_shape
        require(first_shape == shape, f"inconsistent video frame shape at {path}")
        if index % 100 == 0 or index == len(paths):
            print(f"  video decode: {index}/{len(paths)}", flush=True)
    assert shape is not None
    return len(paths), shape


def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def validate_training_read(
    root: Path, info: dict[str, Any], total_frames: int, action_horizon: int
) -> dict[str, Any]:
    LeRobotDataset = import_lerobot_dataset()
    fps = float(info["fps"])
    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        download_videos=False,
        delta_timestamps={"action": [step / fps for step in range(action_horizon)]},
    )
    require(len(dataset) == total_frames, (
        f"LeRobotDataset length={len(dataset)}, expected total_frames={total_frames}"
    ))

    sample_shapes: dict[str, list[int]] = {}
    for sample_index in (0, len(dataset) - 1):
        sample = dataset[sample_index]
        required = {"action", "observation.state"} | EXPECTED_VIDEO_KEYS
        missing = required - set(sample)
        require(not missing, f"LeRobotDataset sample {sample_index} missing keys: {sorted(missing)}")
        for key in required:
            value = sample[key]
            shape = tuple(int(dim) for dim in value.shape)
            require(np.prod(shape) > 0, f"empty {key} in sample {sample_index}")
            sample_shapes[key] = list(shape)
        for key in ("action", "observation.state"):
            value = sample[key]
            array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            require(np.isfinite(array).all(), f"non-finite {key} in sample {sample_index}")

    require(sample_shapes["action"][0] == action_horizon, (
        f"training-style action horizon is {sample_shapes['action']}, expected first dimension {action_horizon}"
    ))
    return {"dataset_length": len(dataset), "sample_shapes": sample_shapes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Local LeRobot dataset directory")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--producer-exit-code", type=int)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    report: dict[str, Any] = {"dataset": str(root), "passed": False, "gates": {}}

    try:
        require(root.is_dir(), f"dataset directory does not exist: {root}")
        if args.producer_exit_code is not None:
            require(args.producer_exit_code == 0, (
                f"producer did not exit naturally: exit code {args.producer_exit_code}"
            ))
            report["gates"]["producer_exit"] = "pass"

        info = load_json(root / "meta" / "info.json")
        version = str(info.get("codebase_version"))
        require(version in {"v2.1", "v3.0"}, f"unsupported LeRobot version: {version}")
        episodes = load_episodes(root, version)

        total_episodes, total_frames = validate_parquet_and_counts(root, info, episodes)
        report["gates"]["parquet_and_counts"] = "pass"
        if args.expected_episodes is not None:
            require(total_episodes == args.expected_episodes, (
                f"expected {args.expected_episodes} episodes, found {total_episodes}"
            ))
        video_count, video_shape = validate_videos(root, info, episodes)
        report["gates"]["videos"] = "pass"
        training = validate_training_read(root, info, total_frames, args.action_horizon)
        report["gates"]["lerobot_training_read"] = "pass"

        report.update(
            passed=True,
            version=version,
            total_episodes=total_episodes,
            total_frames=total_frames,
            video_files=video_count,
            video_shape=list(video_shape),
            training=training,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
