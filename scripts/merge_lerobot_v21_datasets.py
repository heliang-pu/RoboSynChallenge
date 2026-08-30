#!/usr/bin/env python3
"""Merge compatible local LeRobot v2.1 datasets without touching sources.

The bundled ``lerobot-edit-dataset`` entry point is unavailable on this host.
This replacement handles the legacy per-episode v2.1 layout directly: it
renumbers episode/global-frame indices, hard-links videos when possible, and
rebuilds all metadata and aggregate statistics in a fresh output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STAT_KEYS = ("min", "max", "mean", "std", "count")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def format_data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )


def format_video_path(root: Path, info: dict[str, Any], key: str, episode_index: int) -> Path:
    return root / info["video_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
        video_key=key,
    )


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def without_quantiles(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: stats[key] for key in STAT_KEYS if key in stats}


def aggregate_stats(per_episode: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    all_keys = set().union(*(stats.keys() for stats in per_episode))
    for key in sorted(all_keys):
        entries = [stats[key] for stats in per_episode if key in stats]
        counts = np.asarray([float(np.asarray(item["count"]).reshape(-1)[0]) for item in entries])
        total = counts.sum()
        mins = np.stack([np.asarray(item["min"], dtype=np.float64) for item in entries])
        maxs = np.stack([np.asarray(item["max"], dtype=np.float64) for item in entries])
        means = np.stack([np.asarray(item["mean"], dtype=np.float64) for item in entries])
        stds = np.stack([np.asarray(item["std"], dtype=np.float64) for item in entries])
        weights = counts.reshape((-1,) + (1,) * (means.ndim - 1)) / total
        mean = (means * weights).sum(axis=0)
        variance = ((stds**2 + (means - mean) ** 2) * weights).sum(axis=0)
        output[key] = {
            "min": mins.min(axis=0).tolist(),
            "max": maxs.max(axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(total)],
        }
    return output


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.column_names.index(name)
    return table.set_column(index, name, pa.array(values))


def validate_sources(sources: list[Path]) -> tuple[dict[str, Any], list[str]]:
    base = read_json(sources[0] / "meta" / "info.json")
    if base.get("codebase_version") != "v2.1":
        raise ValueError(f"{sources[0]} is not LeRobot v2.1")
    video_keys = [key for key, value in base["features"].items() if value.get("dtype") == "video"]
    for source in sources:
        info = read_json(source / "meta" / "info.json")
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{source} is not LeRobot v2.1")
        for key in ("fps", "chunks_size", "data_path", "video_path", "robot_type", "features"):
            if info.get(key) != base.get(key):
                raise ValueError(f"{source} differs from the first source in {key}")
        for required in ("meta/episodes.jsonl", "meta/episodes_stats.jsonl", "meta/tasks.jsonl"):
            if not (source / required).is_file():
                raise FileNotFoundError(f"{source} is missing {required}")
    return base, video_keys


def merge(sources: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    info, video_keys = validate_sources(sources)
    output.mkdir(parents=True)
    (output / "meta").mkdir()

    task_to_index: dict[str, int] = {}
    for source in sources:
        for record in read_jsonl(source / "meta" / "tasks.jsonl"):
            task_to_index.setdefault(record["task"], len(task_to_index))

    output_episodes: list[dict[str, Any]] = []
    output_episode_stats: list[dict[str, Any]] = []
    stats_for_aggregate: list[dict[str, Any]] = []
    next_episode = 0
    index_offset = 0

    for source in sources:
        source_info = read_json(source / "meta" / "info.json")
        source_stats = {
            int(record["episode_index"]): record["stats"]
            for record in read_jsonl(source / "meta" / "episodes_stats.jsonl")
        }
        episodes = sorted(read_jsonl(source / "meta" / "episodes.jsonl"), key=lambda row: int(row["episode_index"]))
        for episode in episodes:
            source_episode = int(episode["episode_index"])
            length = int(episode["length"])
            task = episode["tasks"][0]
            task_index = task_to_index[task]
            table = pq.read_table(format_data_path(source, source_info, source_episode))
            if table.num_rows != length:
                raise ValueError(f"{source} episode {source_episode} length mismatch")
            required = {"episode_index", "frame_index", "index", "task_index"}
            missing = required - set(table.column_names)
            if missing:
                raise ValueError(f"{source} episode {source_episode} missing columns {sorted(missing)}")
            table = replace_column(table, "episode_index", np.full(length, next_episode, dtype=np.int64))
            table = replace_column(table, "frame_index", np.arange(length, dtype=np.int64))
            table = replace_column(table, "index", np.arange(index_offset, index_offset + length, dtype=np.int64))
            table = replace_column(table, "task_index", np.full(length, task_index, dtype=np.int64))
            destination_data = format_data_path(output, info, next_episode)
            destination_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.replace_schema_metadata(None), destination_data)

            for video_key in video_keys:
                source_video = format_video_path(source, source_info, video_key, source_episode)
                if not source_video.is_file():
                    raise FileNotFoundError(f"missing video: {source_video}")
                link_or_copy(source_video, format_video_path(output, info, video_key, next_episode))

            stats = {key: without_quantiles(value) for key, value in source_stats[source_episode].items()}
            stats["episode_index"] = numeric_stats(np.full(length, next_episode, dtype=np.int64))
            stats["frame_index"] = numeric_stats(np.arange(length, dtype=np.int64))
            stats["index"] = numeric_stats(np.arange(index_offset, index_offset + length, dtype=np.int64))
            stats["task_index"] = numeric_stats(np.full(length, task_index, dtype=np.int64))
            output_episodes.append({"episode_index": next_episode, "tasks": [task], "length": length})
            output_episode_stats.append({"episode_index": next_episode, "stats": stats})
            stats_for_aggregate.append(stats)
            next_episode += 1
            index_offset += length

    info["total_episodes"] = next_episode
    info["total_frames"] = index_offset
    info["total_tasks"] = len(task_to_index)
    info["total_videos"] = next_episode * len(video_keys)
    info["total_chunks"] = (next_episode - 1) // int(info["chunks_size"]) + 1 if next_episode else 0
    info["splits"] = {"train": f"0:{next_episode}"}
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    write_jsonl(output / "meta" / "tasks.jsonl", [
        {"task_index": index, "task": task}
        for task, index in sorted(task_to_index.items(), key=lambda pair: pair[1])
    ])
    write_jsonl(output / "meta" / "episodes.jsonl", output_episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", output_episode_stats)
    (output / "meta" / "stats.json").write_text(json.dumps(aggregate_stats(stats_for_aggregate), indent=4) + "\n")
    (output / "MERGE_SOURCES.json").write_text(json.dumps({
        "sources": [str(source) for source in sources],
        "episodes": next_episode,
        "frames": index_offset,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    merge(args.sources, args.out)


if __name__ == "__main__":
    main()
