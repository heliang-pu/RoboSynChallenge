#!/usr/bin/env python3
"""Validate a cleaned seeded v2.1 derivative against its immutable parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


STAT_KEYS = ("min", "max", "mean", "std", "count")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def data_path(root: Path, info: dict[str, Any], episode: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=episode // int(info["chunks_size"]),
        episode_index=episode,
        chunk_index=episode // int(info["chunks_size"]),
        file_index=episode,
    )


def video_path(root: Path, info: dict[str, Any], key: str, episode: int) -> Path:
    return root / info["video_path"].format(
        episode_chunk=episode // int(info["chunks_size"]),
        episode_index=episode,
        video_key=key,
        chunk_index=episode // int(info["chunks_size"]),
        file_index=episode,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def aggregate_stats(per_episode: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    all_keys = set().union(*(stats.keys() for stats in per_episode))
    for key in sorted(all_keys):
        entries = [stats[key] for stats in per_episode if key in stats]
        counts = np.asarray(
            [float(np.asarray(item["count"]).reshape(-1)[0]) for item in entries]
        )
        total = counts.sum()
        mins = np.stack([np.asarray(item["min"], dtype=np.float64) for item in entries])
        maxs = np.stack([np.asarray(item["max"], dtype=np.float64) for item in entries])
        means = np.stack([np.asarray(item["mean"], dtype=np.float64) for item in entries])
        stds = np.stack([np.asarray(item["std"], dtype=np.float64) for item in entries])
        weights = counts.reshape((-1,) + (1,) * (means.ndim - 1)) / total
        mean = (means * weights).sum(axis=0)
        variance = ((stds**2 + (means - mean) ** 2) * weights).sum(axis=0)
        output[key] = {
            "min": mins.min(axis=0),
            "max": maxs.max(axis=0),
            "mean": mean,
            "std": np.sqrt(np.maximum(variance, 0.0)),
            "count": np.asarray([int(total)]),
        }
    return output


def expected_numeric_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.asarray([values.shape[0]], dtype=np.float64),
    }


def validate_task(
    task: str,
    parent: Path,
    clean: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    parent_info = read_json(parent / "meta" / "info.json")
    clean_info = read_json(clean / "meta" / "info.json")
    parent_episodes = read_jsonl(parent / "meta" / "episodes.jsonl")
    clean_episodes = read_jsonl(clean / "meta" / "episodes.jsonl")
    parent_sidecar = read_json(parent / "episode_success.json")
    clean_sidecar = read_json(clean / "episode_success.json")
    source_batches_payload = read_json(clean / "SOURCE_BATCHES.json")
    mapping = read_jsonl(clean / "EPISODE_INDEX_MAP.jsonl")
    removed = set(int(value) for value in plan.get("remove_episodes", {}).get(task, []))
    removed_features = set(plan.get("remove_features_all_tasks", [])) | set(
        plan.get("remove_features", {}).get(task, [])
    )
    expected_parent_indices = [
        index for index in range(len(parent_episodes)) if index not in removed
    ]
    require(clean_info["codebase_version"] == "v2.1", f"{task}: not v2.1")
    require(
        int(clean_info["total_episodes"]) == len(expected_parent_indices),
        f"{task}: clean episode count mismatch",
    )
    require(
        [int(row["episode_index"]) for row in clean_episodes]
        == list(range(len(clean_episodes))),
        f"{task}: clean episode indices not contiguous",
    )
    require(
        [int(row["parent_episode_index"]) for row in mapping]
        == expected_parent_indices,
        f"{task}: parent mapping mismatch",
    )
    require(
        [int(row["clean_episode_index"]) for row in mapping]
        == list(range(len(clean_episodes))),
        f"{task}: clean mapping mismatch",
    )
    require(
        removed_features.isdisjoint(clean_info["features"]),
        f"{task}: removed feature remains in info.json",
    )
    require(
        set(clean_info["features"]) == set(parent_info["features"]) - removed_features,
        f"{task}: clean feature set mismatch",
    )
    for feature, spec in clean_info["features"].items():
        require(
            spec == parent_info["features"][feature],
            f"{task}: retained feature spec changed: {feature}",
        )

    global_stats = read_json(clean / "meta" / "stats.json")
    require(
        removed_features.isdisjoint(global_stats),
        f"{task}: removed feature remains in global stats",
    )
    require(
        set(global_stats) == set(clean_info["features"]),
        f"{task}: global stats key set mismatch",
    )
    episode_stats = read_jsonl(clean / "meta" / "episodes_stats.jsonl")
    parent_episode_stats = {
        int(row["episode_index"]): row["stats"]
        for row in read_jsonl(parent / "meta" / "episodes_stats.jsonl")
    }
    require(len(episode_stats) == len(clean_episodes), f"{task}: episode stats count")
    require(
        [int(row["episode_index"]) for row in episode_stats]
        == list(range(len(clean_episodes))),
        f"{task}: episode stats indices mismatch",
    )
    for row in episode_stats:
        require(
            removed_features.isdisjoint(row["stats"]),
            f"{task}: removed feature remains in episode stats",
        )
        require(
            set(row["stats"]) == set(clean_info["features"]),
            f"{task}: episode stats key set mismatch",
        )
    clean_episode_stats_by_ep = {
        int(row["episode_index"]): row["stats"] for row in episode_stats
    }

    parent_sidecar_by_ep = {
        int(row["episode_index"]): row for row in parent_sidecar["episodes"]
    }
    require(
        int(clean_sidecar["saved_episode_count"]) == len(clean_episodes)
        and len(clean_sidecar["episodes"]) == len(clean_episodes),
        f"{task}: clean sidecar count mismatch",
    )
    require(
        [int(row["episode_index"]) for row in clean_sidecar["episodes"]]
        == list(range(len(clean_episodes))),
        f"{task}: clean sidecar indices mismatch",
    )
    require(
        source_batches_payload.get("sources") == clean_sidecar.get("source_batches"),
        f"{task}: SOURCE_BATCHES differs from sidecar provenance",
    )
    clean_seeds = []
    total_frames = 0
    next_global_index = 0
    video_keys = [
        key for key, value in clean_info["features"].items() if value.get("dtype") == "video"
    ]
    require(len(video_keys) == 3, f"{task}: expected three cameras")
    actual_data_files = sorted((clean / "data").glob("**/*.parquet"))
    expected_data_files = sorted(
        data_path(clean, clean_info, episode) for episode in range(len(clean_episodes))
    )
    require(actual_data_files == expected_data_files, f"{task}: stale/missing parquet files")
    for key in video_keys:
        actual_video_files = sorted((clean / "videos" / "chunk-000" / key).glob("*.mp4"))
        expected_video_files = sorted(
            video_path(clean, clean_info, key, episode)
            for episode in range(len(clean_episodes))
        )
        require(actual_video_files == expected_video_files, f"{task}: stale/missing videos for {key}")
    for new_episode, map_row in enumerate(mapping):
        old_episode = int(map_row["parent_episode_index"])
        parent_episode = parent_episodes[old_episode]
        clean_episode = clean_episodes[new_episode]
        require(
            int(clean_episode["length"]) == int(parent_episode["length"]),
            f"{task}: episode {new_episode} length changed",
        )
        require(
            clean_episode["tasks"] == parent_episode["tasks"],
            f"{task}: episode {new_episode} task text changed",
        )
        length = int(clean_episode["length"])
        total_frames += length

        clean_stats = clean_episode_stats_by_ep[new_episode]
        parent_stats = parent_episode_stats[old_episode]
        for key in set(clean_stats) - {"episode_index", "frame_index", "index", "task_index"}:
            require(
                clean_stats[key] == {name: parent_stats[key][name] for name in STAT_KEYS},
                f"{task}: episode {new_episode} retained stats changed: {key}",
            )
        rewritten_expected = {
            "episode_index": expected_numeric_stats(np.full(length, new_episode)),
            "frame_index": expected_numeric_stats(np.arange(length)),
            "index": expected_numeric_stats(
                np.arange(next_global_index, next_global_index + length)
            ),
            "task_index": expected_numeric_stats(np.zeros(length)),
        }
        for key, expected_stats in rewritten_expected.items():
            for stat_name, expected_value in expected_stats.items():
                require(
                    np.allclose(
                        np.asarray(clean_stats[key][stat_name], dtype=np.float64),
                        expected_value,
                        rtol=1e-12,
                        atol=1e-12,
                    ),
                    f"{task}: episode {new_episode} rewritten stats mismatch: "
                    f"{key}.{stat_name}",
                )

        parent_table = pq.read_table(data_path(parent, parent_info, old_episode))
        clean_table = pq.read_table(data_path(clean, clean_info, new_episode))
        expected_columns = [
            name for name in parent_table.column_names if name not in removed_features
        ]
        require(
            clean_table.column_names == expected_columns,
            f"{task}: episode {new_episode} parquet columns mismatch",
        )
        require(clean_table.num_rows == length, f"{task}: episode rows mismatch")
        for column in expected_columns:
            if column == "episode_index":
                require(
                    np.array_equal(
                        clean_table[column].to_numpy(), np.full(length, new_episode)
                    ),
                    f"{task}: episode_index rewrite mismatch",
                )
            elif column == "index":
                require(
                    np.array_equal(
                        clean_table[column].to_numpy(),
                        np.arange(next_global_index, next_global_index + length),
                    ),
                    f"{task}: global index rewrite mismatch",
                )
            else:
                require(
                    clean_table[column].equals(parent_table[column]),
                    f"{task}: episode {new_episode} retained column changed: {column}",
                )

        clean_side = clean_sidecar["episodes"][new_episode]
        parent_side = parent_sidecar_by_ep[old_episode]
        for key in ("seed", "source_batch", "source_episode_index"):
            require(
                int(map_row[key]) == int(parent_side[key]),
                f"{task}: episode {new_episode} index-map field changed: {key}",
            )
        for key in ("seed", "success", "env_steps", "source_batch", "source_episode_index"):
            require(
                clean_side.get(key) == parent_side.get(key),
                f"{task}: episode {new_episode} sidecar changed: {key}",
            )
        require(
            int(clean_side["env_steps"]) == length,
            f"{task}: episode {new_episode} sidecar env_steps mismatch",
        )
        require(
            int(clean_side["source_merged_episode_index"]) == old_episode,
            f"{task}: source merged index missing",
        )
        clean_seeds.append(int(clean_side["seed"]))

        for key in video_keys:
            parent_video = video_path(parent, parent_info, key, old_episode)
            clean_video = video_path(clean, clean_info, key, new_episode)
            require(
                clean_video.is_file() and clean_video.stat().st_size > 0,
                f"{task}: missing clean video {clean_video}",
            )
            require(
                clean_video.stat().st_size == parent_video.stat().st_size,
                f"{task}: retained video size changed",
            )
        next_global_index += length

    require(len(clean_seeds) == len(set(clean_seeds)), f"{task}: duplicate clean seeds")
    require(
        set(clean_seeds)
        == {int(row["seed"]) for index, row in parent_sidecar_by_ep.items() if index not in removed},
        f"{task}: clean seed set mismatch",
    )
    removed_payload = read_json(clean / "REMOVED_EPISODES.json")
    require(
        sorted(int(row["source_merged_episode_index"]) for row in removed_payload["episodes"])
        == sorted(removed),
        f"{task}: removed episode manifest mismatch",
    )
    expected_removed_seeds = {
        int(index): int(seed)
        for index, seed in plan.get("expected_removed_seeds", {}).get(task, {}).items()
    }
    actual_removed_seeds = {
        int(row["source_merged_episode_index"]): int(row["seed"])
        for row in removed_payload["episodes"]
    }
    require(
        actual_removed_seeds == expected_removed_seeds,
        f"{task}: removed episode seed mismatch",
    )
    parent_sources = {
        int(row["batch_index"]): row for row in parent_sidecar.get("source_batches", [])
    }
    clean_records_by_batch: dict[int, list[dict[str, Any]]] = {}
    removed_records_by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in clean_sidecar["episodes"]:
        clean_records_by_batch.setdefault(int(row["source_batch"]), []).append(row)
    for row in removed_payload["episodes"]:
        removed_records_by_batch.setdefault(int(row["source_batch"]), []).append(row)
    for clean_source in clean_sidecar.get("source_batches", []):
        batch = int(clean_source["batch_index"])
        parent_source = parent_sources[batch]
        kept = clean_records_by_batch.get(batch, [])
        removed_rows = removed_records_by_batch.get(batch, [])
        require(
            clean_source.get("parent_merged_episode_range")
            == parent_source.get("merged_episode_range"),
            f"{task}: batch {batch} parent range mismatch",
        )
        require(
            int(clean_source.get("clean_episodes", -1)) == len(kept)
            and int(clean_source.get("clean_frames", -1))
            == sum(int(row["env_steps"]) for row in kept),
            f"{task}: batch {batch} clean count/frame provenance mismatch",
        )
        require(
            clean_source.get("removed_source_episode_indices")
            == sorted(int(row["source_episode_index"]) for row in removed_rows),
            f"{task}: batch {batch} removed provenance mismatch",
        )
        expected_range = (
            [min(int(row["episode_index"]) for row in kept),
             max(int(row["episode_index"]) for row in kept)]
            if kept else None
        )
        require(
            clean_source.get("clean_merged_episode_range") == expected_range,
            f"{task}: batch {batch} clean range mismatch",
        )
    recomputed_global = aggregate_stats(
        [clean_episode_stats_by_ep[index] for index in range(len(clean_episodes))]
    )
    for key, expected in recomputed_global.items():
        for stat_name, expected_value in expected.items():
            actual_value = np.asarray(global_stats[key][stat_name], dtype=np.float64)
            require(
                np.allclose(actual_value, expected_value, rtol=1e-10, atol=1e-10),
                f"{task}: global stats mismatch: {key}.{stat_name}",
            )
    cleaning_manifest = read_json(clean / "CLEANING_MANIFEST.json")
    expected_video_count = len(clean_episodes) * 3
    require(
        int(cleaning_manifest.get("video_hardlinks", 0))
        + int(cleaning_manifest.get("video_copies", 0))
        == expected_video_count,
        f"{task}: video link/copy count mismatch",
    )
    if plan.get("require_video_hardlinks"):
        require(
            int(cleaning_manifest.get("video_hardlinks", -1)) == expected_video_count
            and int(cleaning_manifest.get("video_copies", -1)) == 0,
            f"{task}: expected every clean video to be a hardlink",
        )
    validation = read_json(clean / "VALIDATION_REPORT.json")
    require(
        validation.get("passed")
        and int(validation.get("total_episodes", -1)) == len(clean_episodes)
        and int(validation.get("total_frames", -1)) == total_frames
        and int(validation.get("video_files", -1)) == len(clean_episodes) * 3
        and validation.get("gates", {}).get("parquet_and_counts") == "pass"
        and validation.get("gates", {}).get("videos") == "pass"
        and validation.get("gates", {}).get("lerobot_training_read") == "pass",
        f"{task}: strict validation did not fully pass",
    )
    require(int(clean_info["total_frames"]) == total_frames, f"{task}: frame total mismatch")
    require(
        int(clean_info["total_videos"]) == len(clean_episodes) * 3,
        f"{task}: video total mismatch",
    )
    return {
        "episodes": len(clean_episodes),
        "frames": total_frames,
        "videos": len(clean_episodes) * 3,
        "removed_episodes": sorted(removed),
        "removed_features": sorted(removed_features),
        "retained_columns_verified": True,
        "retained_video_sizes_verified": True,
        "seed_mapping_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    tasks = sorted(
        path.name
        for path in args.parent_root.iterdir()
        if (path / "meta" / "info.json").is_file()
    )
    output: dict[str, Any] = {"passed": False, "tasks": {}}
    try:
        for task in tasks:
            print(f"validate clean {task}", flush=True)
            output["tasks"][task] = validate_task(
                task, args.parent_root / task, args.clean_root / task, plan
            )
        totals = {
            "episodes": sum(row["episodes"] for row in output["tasks"].values()),
            "frames": sum(row["frames"] for row in output["tasks"].values()),
            "videos": sum(row["videos"] for row in output["tasks"].values()),
        }
        root_manifest = read_json(args.clean_root / "CLEANING_MANIFEST.json")
        require(root_manifest["status"] == "complete", "root cleaning manifest incomplete")
        require(root_manifest["totals"]["episodes"] == totals["episodes"], "root episodes")
        require(root_manifest["totals"]["frames"] == totals["frames"], "root frames")
        require(root_manifest["totals"]["videos"] == totals["videos"], "root videos")
        require(totals == {"episodes": 9995, "frames": 2953124, "videos": 29985}, f"unexpected totals: {totals}")
        parent_total = sum(
            int(read_json(args.parent_root / task / "meta" / "info.json")["total_episodes"])
            for task in tasks
        )
        require(parent_total == 10000, "parent dataset changed or incomplete")
        output.update(passed=True, totals=totals, parent_episodes=parent_total)
        return 0
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
