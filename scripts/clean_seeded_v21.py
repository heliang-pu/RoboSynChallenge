#!/usr/bin/env python3
"""Create a verified clean LeRobot v2.1 derivative without touching its source.

The cleaner removes selected episodes and features, renumbers all episode and
frame indices, rebuilds per-episode/global statistics, hard-links videos when
the destination filesystem supports it, updates success/provenance sidecars,
and publishes each task only after strict target-runtime validation passes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from validate_clean_seeded_v21 import validate_task as validate_clean_task
except ImportError:  # Allows importing this file as scripts.clean_seeded_v21.
    from scripts.validate_clean_seeded_v21 import validate_task as validate_clean_task


STAT_KEYS = ("min", "max", "mean", "std", "count")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_fingerprint(source_root: Path, tasks: list[str]) -> dict[str, Any]:
    output = {"root": str(source_root), "tasks": {}}
    for task in tasks:
        root = source_root / task
        info = read_json(root / "meta" / "info.json")
        output["tasks"][task] = {
            "episodes": int(info["total_episodes"]),
            "frames": int(info["total_frames"]),
            "videos": int(info["total_videos"]),
            "info_sha256": sha256(root / "meta" / "info.json"),
            "episodes_sha256": sha256(root / "meta" / "episodes.jsonl"),
            "sidecar_sha256": sha256(root / "episode_success.json"),
        }
    return output


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


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
        counts = np.asarray(
            [float(np.asarray(item["count"]).reshape(-1)[0]) for item in entries]
        )
        total = counts.sum()
        if total <= 0:
            raise ValueError(f"invalid aggregate count for {key}: {total}")
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
            "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
            "count": [int(total)],
        }
    return output


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.column_names.index(name)
    return table.set_column(index, name, pa.array(values))


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


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def validate_plan(source_root: Path, plan: dict[str, Any]) -> list[str]:
    tasks = sorted(
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").is_file()
    )
    if set(plan.get("remove_episodes", {})) - set(tasks):
        raise ValueError("remove_episodes contains an unknown task")
    if set(plan.get("remove_features", {})) - set(tasks):
        raise ValueError("remove_features contains an unknown task")
    global_features = set(plan.get("remove_features_all_tasks", []))
    for task in tasks:
        root = source_root / task
        info = read_json(root / "meta" / "info.json")
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{task} is not LeRobot v2.1")
        episodes = read_jsonl(root / "meta" / "episodes.jsonl")
        if len(episodes) != int(info["total_episodes"]):
            raise ValueError(f"{task} episode count mismatch")
        remove_episodes = [int(value) for value in plan.get("remove_episodes", {}).get(task, [])]
        if len(remove_episodes) != len(set(remove_episodes)):
            raise ValueError(f"{task} remove_episodes contains duplicates")
        invalid = set(remove_episodes) - set(range(len(episodes)))
        if invalid:
            raise ValueError(f"{task} remove_episodes out of range: {sorted(invalid)}")
        sidecar = read_json(root / "episode_success.json")
        sidecar_by_episode = {
            int(row["episode_index"]): row for row in sidecar["episodes"]
        }
        expected_seeds = plan.get("expected_removed_seeds", {}).get(task, {})
        require_expected = {str(index) for index in remove_episodes}
        if set(expected_seeds) != require_expected:
            raise ValueError(f"{task} expected_removed_seeds does not match removal list")
        for index in remove_episodes:
            if int(sidecar_by_episode[index]["seed"]) != int(expected_seeds[str(index)]):
                raise ValueError(f"{task} episode {index} seed pin mismatch")
        remove_features = global_features | set(
            plan.get("remove_features", {}).get(task, [])
        )
        missing = remove_features - set(info["features"])
        if missing:
            raise ValueError(f"{task} missing features requested for removal: {sorted(missing)}")
        protected = {
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        }
        forbidden = remove_features & protected
        if forbidden:
            raise ValueError(f"{task} plan removes protected features: {sorted(forbidden)}")
    return tasks


def safe_remove_partial(path: Path, output_root: Path) -> None:
    resolved, root = path.resolve(), output_root.resolve()
    if root not in resolved.parents or ".partial" not in path.name:
        raise RuntimeError(f"refusing to remove unsafe path: {path}")
    if path.exists():
        shutil.rmtree(path)


def task_build_complete(
    path: Path,
    source: Path,
    expected_episodes: int,
    expected_frames: int,
    removed_episodes: set[int],
    removed_features: set[str],
) -> bool:
    try:
        marker = read_json(path / ".build_complete.json")
        info = read_json(path / "meta" / "info.json")
        manifest = read_json(path / "CLEANING_MANIFEST.json")
        return (
            marker.get("status") == "build_complete"
            and int(info.get("total_episodes", -1)) == expected_episodes
            and int(info.get("total_frames", -1)) == expected_frames
            and manifest.get("parent_dataset") == str(source)
            and set(manifest.get("removed_episode_indices", [])) == removed_episodes
            and set(manifest.get("removed_features", [])) == removed_features
        )
    except Exception:
        return False


def rebuild_source_batches(
    original: list[dict[str, Any]],
    kept_sidecar_records: list[dict[str, Any]],
    removed_sidecar_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kept_by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    removed_by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in kept_sidecar_records:
        kept_by_batch[int(record["source_batch"])].append(record)
    for record in removed_sidecar_records:
        removed_by_batch[int(record["source_batch"])].append(record)

    output = []
    for source in original:
        batch_index = int(source["batch_index"])
        item = dict(source)
        old_range = item.pop("merged_episode_range", None)
        if old_range is not None:
            item["parent_merged_episode_range"] = old_range
        kept = kept_by_batch.get(batch_index, [])
        removed = removed_by_batch.get(batch_index, [])
        item["clean_episodes"] = len(kept)
        item["clean_frames"] = sum(int(record.get("env_steps", 0)) for record in kept)
        item["removed_source_episode_indices"] = sorted(
            int(record["source_episode_index"]) for record in removed
        )
        item["clean_merged_episode_range"] = (
            [min(int(record["episode_index"]) for record in kept),
             max(int(record["episode_index"]) for record in kept)]
            if kept
            else None
        )
        output.append(item)
    return output


def build_task(
    task: str,
    source_root: Path,
    output_root: Path,
    plan: dict[str, Any],
) -> Path:
    source = source_root / task
    partial = output_root / f".{task}.partial"
    source_info = read_json(source / "meta" / "info.json")
    source_episodes = sorted(
        read_jsonl(source / "meta" / "episodes.jsonl"),
        key=lambda row: int(row["episode_index"]),
    )
    source_stats = {
        int(row["episode_index"]): row["stats"]
        for row in read_jsonl(source / "meta" / "episodes_stats.jsonl")
    }
    source_sidecar = read_json(source / "episode_success.json")
    sidecar_by_episode = {
        int(row["episode_index"]): row for row in source_sidecar["episodes"]
    }
    if set(sidecar_by_episode) != set(range(len(source_episodes))):
        raise ValueError(f"{task} sidecar indices do not match episode metadata")

    removed_episodes = set(
        int(value) for value in plan.get("remove_episodes", {}).get(task, [])
    )
    removed_features = set(plan.get("remove_features_all_tasks", [])) | set(
        plan.get("remove_features", {}).get(task, [])
    )
    expected_episodes = len(source_episodes) - len(removed_episodes)
    expected_frames = sum(
        int(episode["length"])
        for index, episode in enumerate(source_episodes)
        if index not in removed_episodes
    )
    if task_build_complete(
        partial,
        source,
        expected_episodes,
        expected_frames,
        removed_episodes,
        removed_features,
    ):
        log(f"SKIP built {task}")
        return partial
    safe_remove_partial(partial, output_root)
    partial.mkdir(parents=True)

    info = json.loads(json.dumps(source_info))
    for feature in removed_features:
        info["features"].pop(feature, None)
    video_keys = [
        key for key, value in info["features"].items() if value.get("dtype") == "video"
    ]
    output_episodes: list[dict[str, Any]] = []
    output_episode_stats: list[dict[str, Any]] = []
    aggregate_inputs: list[dict[str, Any]] = []
    output_sidecar_records: list[dict[str, Any]] = []
    removed_sidecar_records: list[dict[str, Any]] = []
    episode_index_map: list[dict[str, Any]] = []
    global_index = 0
    hardlinks = 0
    copies = 0

    for old_episode, episode in enumerate(source_episodes):
        if old_episode in removed_episodes:
            removed = dict(sidecar_by_episode[old_episode])
            expected_seed = (
                plan.get("expected_removed_seeds", {})
                .get(task, {})
                .get(str(old_episode))
            )
            if expected_seed is None or int(removed["seed"]) != int(expected_seed):
                raise ValueError(
                    f"{task} episode {old_episode} seed mismatch: "
                    f"{removed.get('seed')} != {expected_seed}"
                )
            removed["source_merged_episode_index"] = old_episode
            removed["removal_reason"] = plan.get("episode_reasons", {}).get(
                task, {}
            ).get(str(old_episode), "QC clean-expert exclusion")
            removed_sidecar_records.append(removed)
            continue

        new_episode = len(output_episodes)
        length = int(episode["length"])
        table = pq.read_table(data_path(source, source_info, old_episode))
        if table.num_rows != length:
            raise ValueError(f"{task} episode {old_episode} row mismatch")
        missing_columns = removed_features - set(table.column_names)
        if missing_columns:
            raise ValueError(
                f"{task} episode {old_episode} missing removal columns {sorted(missing_columns)}"
            )
        table = table.drop(sorted(removed_features))
        table = replace_column(
            table, "episode_index", np.full(length, new_episode, dtype=np.int64)
        )
        table = replace_column(
            table, "frame_index", np.arange(length, dtype=np.int64)
        )
        table = replace_column(
            table, "index", np.arange(global_index, global_index + length, dtype=np.int64)
        )
        table = replace_column(
            table, "task_index", np.zeros(length, dtype=np.int64)
        )
        destination_data = data_path(partial, info, new_episode)
        destination_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(None), destination_data)

        for video_key in video_keys:
            method = link_or_copy(
                video_path(source, source_info, video_key, old_episode),
                video_path(partial, info, video_key, new_episode),
            )
            hardlinks += method == "hardlink"
            copies += method == "copy"

        stats = {
            key: without_quantiles(value)
            for key, value in source_stats[old_episode].items()
            if key not in removed_features
        }
        stats["episode_index"] = numeric_stats(
            np.full(length, new_episode, dtype=np.int64)
        )
        stats["frame_index"] = numeric_stats(np.arange(length, dtype=np.int64))
        stats["index"] = numeric_stats(
            np.arange(global_index, global_index + length, dtype=np.int64)
        )
        stats["task_index"] = numeric_stats(np.zeros(length, dtype=np.int64))
        output_episodes.append(
            {"episode_index": new_episode, "tasks": episode["tasks"], "length": length}
        )
        output_episode_stats.append({"episode_index": new_episode, "stats": stats})
        aggregate_inputs.append(stats)

        sidecar_record = dict(sidecar_by_episode[old_episode])
        sidecar_record["source_merged_episode_index"] = old_episode
        sidecar_record["episode_index"] = new_episode
        output_sidecar_records.append(sidecar_record)
        episode_index_map.append(
            {
                "parent_episode_index": old_episode,
                "clean_episode_index": new_episode,
                "seed": int(sidecar_record["seed"]),
                "source_batch": int(sidecar_record["source_batch"]),
                "source_episode_index": int(sidecar_record["source_episode_index"]),
            }
        )
        global_index += length

    info["total_episodes"] = len(output_episodes)
    info["total_frames"] = global_index
    info["total_tasks"] = 1
    info["total_videos"] = len(output_episodes) * len(video_keys)
    info["total_chunks"] = (
        (len(output_episodes) - 1) // int(info["chunks_size"]) + 1
        if output_episodes
        else 0
    )
    info["splits"] = {"train": f"0:{len(output_episodes)}"}
    (partial / "meta").mkdir(parents=True, exist_ok=True)
    atomic_json(partial / "meta" / "info.json", info)
    shutil.copy2(source / "meta" / "tasks.jsonl", partial / "meta" / "tasks.jsonl")
    write_jsonl(partial / "meta" / "episodes.jsonl", output_episodes)
    write_jsonl(partial / "meta" / "episodes_stats.jsonl", output_episode_stats)
    atomic_json(partial / "meta" / "stats.json", aggregate_stats(aggregate_inputs))

    clean_sidecar = {
        key: value for key, value in source_sidecar.items()
        if key not in {"episodes", "source_batches", "saved_episode_count", "all_success"}
    }
    clean_sidecar.update(
        {
            "parent_dataset": str(source),
            "saved_episode_count": len(output_sidecar_records),
            "all_success": all(bool(row.get("success")) for row in output_sidecar_records),
            "source_batches": rebuild_source_batches(
                source_sidecar.get("source_batches", []),
                output_sidecar_records,
                removed_sidecar_records,
            ),
            "episodes": output_sidecar_records,
        }
    )
    atomic_json(partial / "episode_success.json", clean_sidecar)
    atomic_json(partial / "SOURCE_BATCHES.json", {
        "task": task,
        "parent_dataset": str(source),
        "sources": clean_sidecar["source_batches"],
    })
    atomic_json(partial / "REMOVED_EPISODES.json", {
        "task": task,
        "count": len(removed_sidecar_records),
        "episodes": removed_sidecar_records,
    })
    atomic_json(partial / "REMOVED_FEATURES.json", {
        "task": task,
        "features": sorted(removed_features),
    })
    write_jsonl(partial / "EPISODE_INDEX_MAP.jsonl", episode_index_map)
    atomic_json(partial / "CLEANING_MANIFEST.json", {
        "status": "build_complete",
        "task": task,
        "parent_dataset": str(source),
        "parent_episodes": len(source_episodes),
        "clean_episodes": len(output_episodes),
        "clean_frames": global_index,
        "removed_episode_indices": sorted(removed_episodes),
        "removed_features": sorted(removed_features),
        "video_hardlinks": hardlinks,
        "video_copies": copies,
        "built_at": utc_now(),
    })
    atomic_json(partial / ".build_complete.json", {
        "status": "build_complete",
        "episodes": len(output_episodes),
        "frames": global_index,
        "built_at": utc_now(),
    })
    log(
        f"BUILT {task}: {len(output_episodes)} episodes, {global_index} frames, "
        f"videos hardlink={hardlinks} copy={copies}"
    )
    return partial


def validation_matches(
    report: dict[str, Any], expected_episodes: int, expected_frames: int
) -> bool:
    return bool(
        report.get("passed")
        and int(report.get("total_episodes", -1)) == expected_episodes
        and int(report.get("total_frames", -1)) == expected_frames
        and int(report.get("video_files", -1)) == expected_episodes * 3
        and report.get("gates", {}).get("parquet_and_counts") == "pass"
        and report.get("gates", {}).get("videos") == "pass"
        and report.get("gates", {}).get("lerobot_training_read") == "pass"
    )


def validate_and_publish(
    task: str,
    partial: Path,
    output_root: Path,
    validation_python: Path,
    validator: Path,
    cache_root: Path,
    source_root: Path,
    plan: dict[str, Any],
) -> Path:
    final = output_root / task
    if (final / ".complete.json").is_file():
        manifest = read_json(final / "CLEANING_MANIFEST.json")
        partial_manifest = read_json(partial / "CLEANING_MANIFEST.json") if partial.exists() else manifest
        if (
            manifest.get("parent_dataset") != partial_manifest.get("parent_dataset")
            or manifest.get("removed_episode_indices") != partial_manifest.get("removed_episode_indices")
            or manifest.get("removed_features") != partial_manifest.get("removed_features")
        ):
            raise RuntimeError(f"published {task} does not match current cleaning plan")
        log(f"SKIP published {task}")
        return final
    if final.exists():
        raise FileExistsError(f"incomplete final output exists: {final}")
    info = read_json(partial / "meta" / "info.json")
    expected_episodes = int(info["total_episodes"])
    expected_frames = int(info["total_frames"])
    report_path = partial / "VALIDATION_REPORT.json"
    report = read_json(report_path) if report_path.is_file() else None
    if report is None or not validation_matches(report, expected_episodes, expected_frames):
        environment = os.environ.copy()
        environment["HF_HOME"] = str(cache_root / task / "hf")
        environment["HF_DATASETS_CACHE"] = str(cache_root / task / "hf" / "datasets")
        command = [
            str(validation_python),
            str(validator),
            str(partial),
            "--expected-episodes",
            str(expected_episodes),
            "--action-horizon",
            "50",
            "--report",
            str(report_path),
        ]
        if plan.get("reuse_parent_video_validation"):
            command.append("--skip-video-decode")
        log(f"VALIDATE {task}")
        result = subprocess.run(command, env=environment)
        report = read_json(report_path) if report_path.is_file() else None
        if report is None or not validation_matches(report, expected_episodes, expected_frames):
            raise RuntimeError(
                f"validation failed for {task}: exit={result.returncode}, report={report}"
            )
        if result.returncode != 0:
            log(f"WARNING {task} validator exited {result.returncode} after passed report")

    invariant_report = validate_clean_task(task, source_root / task, partial, plan)
    atomic_json(partial / "CLEAN_INVARIANTS_REPORT.json", invariant_report)

    manifest = read_json(partial / "CLEANING_MANIFEST.json")
    manifest.update(status="complete", validated=True, validated_at=utc_now())
    atomic_json(partial / "CLEANING_MANIFEST.json", manifest)
    atomic_json(partial / ".complete.json", {
        "status": "complete",
        "task": task,
        "episodes": expected_episodes,
        "frames": expected_frames,
        "validated": True,
        "completed_at": utc_now(),
    })
    os.replace(partial, final)
    report = read_json(final / "VALIDATION_REPORT.json")
    old_path = report.get("dataset")
    report["dataset"] = str(final)
    if old_path != str(final):
        report["validated_before_atomic_rename"] = old_path
    atomic_json(final / "VALIDATION_REPORT.json", report)
    log(f"PUBLISHED {task}")
    return final


def write_root_artifacts(
    source_root: Path,
    output_root: Path,
    plan: dict[str, Any],
    tasks: list[str],
) -> None:
    task_rows = {}
    total_episodes = 0
    total_frames = 0
    total_videos = 0
    for task in tasks:
        root = output_root / task
        info = read_json(root / "meta" / "info.json")
        manifest = read_json(root / "CLEANING_MANIFEST.json")
        task_rows[task] = {
            "episodes": int(info["total_episodes"]),
            "frames": int(info["total_frames"]),
            "videos": int(info["total_videos"]),
            "removed_episodes": manifest["removed_episode_indices"],
            "removed_features": manifest["removed_features"],
        }
        total_episodes += int(info["total_episodes"])
        total_frames += int(info["total_frames"])
        total_videos += int(info["total_videos"])
    manifest = {
        "status": "complete",
        "format": "LeRobot v2.1",
        "parent_dataset": str(source_root),
        "clean_dataset": str(output_root),
        "plan": plan,
        "tasks": task_rows,
        "totals": {
            "tasks": len(tasks),
            "episodes": total_episodes,
            "frames": total_frames,
            "videos": total_videos,
        },
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "CLEANING_MANIFEST.json", manifest)
    atomic_json(output_root / ".complete.json", {
        "status": "complete",
        **manifest["totals"],
        "completed_at": manifest["completed_at"],
    })
    lines = [
        "# Seeded-1000 LeRobot v2.1 clean derivative",
        "",
        f"Parent: `{source_root}`",
        "",
        "The parent dataset was not modified. QC-selected episodes and confirmed all-zero features were removed.",
        "",
        "| Task | Episodes | Frames | Videos | Removed episodes | Removed features |",
        "|---|---:|---:|---:|---|---|",
    ]
    for task in tasks:
        row = task_rows[task]
        lines.append(
            f"| {task} | {row['episodes']} | {row['frames']} | {row['videos']} | "
            f"{row['removed_episodes'] or '-'} | {', '.join(row['removed_features'])} |"
        )
    lines.extend([
        "",
        f"Totals: {total_episodes} episodes, {total_frames} frames, {total_videos} videos.",
        "",
    ])
    (output_root / "README.md").write_text("\n".join(lines))


def published_task_matches(
    task: str,
    source_root: Path,
    output_root: Path,
    plan: dict[str, Any],
) -> bool:
    final = output_root / task
    if not (final / ".complete.json").is_file():
        return False
    source = source_root / task
    source_episodes = read_jsonl(source / "meta" / "episodes.jsonl")
    removed_episodes = set(
        int(value) for value in plan.get("remove_episodes", {}).get(task, [])
    )
    removed_features = set(plan.get("remove_features_all_tasks", [])) | set(
        plan.get("remove_features", {}).get(task, [])
    )
    expected_frames = sum(
        int(row["length"])
        for index, row in enumerate(source_episodes)
        if index not in removed_episodes
    )
    manifest = read_json(final / "CLEANING_MANIFEST.json")
    info = read_json(final / "meta" / "info.json")
    return bool(
        manifest.get("parent_dataset") == str(source)
        and set(manifest.get("removed_episode_indices", [])) == removed_episodes
        and set(manifest.get("removed_features", [])) == removed_features
        and int(info.get("total_episodes", -1)) == len(source_episodes) - len(removed_episodes)
        and int(info.get("total_frames", -1)) == expected_frames
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validation-python", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--build-workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.absolute()
    output_root = args.output_root.absolute()
    source_resolved = source_root.resolve()
    output_resolved = output_root.resolve()
    if source_resolved == output_resolved:
        raise ValueError("source and output roots must differ")
    if source_resolved in output_resolved.parents or output_resolved in source_resolved.parents:
        raise ValueError("source and output roots must be siblings, not nested")
    plan = read_json(args.plan)
    tasks = validate_plan(source_root, plan)
    fingerprint_before = parent_fingerprint(source_root, tasks)
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume only for this cleaner: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path("/tmp/robosyn_seeded_v21_clean_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        final = output_root / task
        if (final / ".complete.json").is_file() and not published_task_matches(
            task, source_root, output_root, plan
        ):
            raise RuntimeError(f"published task does not match current plan: {task}")

    log(f"BUILD {len(tasks)} tasks with {args.build_workers} workers")
    partials: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.build_workers) as pool:
        futures = {
            pool.submit(build_task, task, source_root, output_root, plan): task
            for task in tasks
            if not (output_root / task / ".complete.json").is_file()
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            partials[task] = future.result()
    for task in tasks:
        if (output_root / task / ".complete.json").is_file():
            continue
        partial = partials.get(task, output_root / f".{task}.partial")
        validate_and_publish(
            task,
            partial,
            output_root,
            args.validation_python.absolute(),
            args.validator.absolute(),
            cache_root,
            source_root,
            plan,
        )
    deep_task_reports = {}
    for task in tasks:
        deep_task_reports[task] = validate_clean_task(
            task, source_root / task, output_root / task, plan
        )
    deep_totals = {
        "episodes": sum(row["episodes"] for row in deep_task_reports.values()),
        "frames": sum(row["frames"] for row in deep_task_reports.values()),
        "videos": sum(row["videos"] for row in deep_task_reports.values()),
    }
    if deep_totals != {"episodes": 9995, "frames": 2953124, "videos": 29985}:
        raise RuntimeError(f"unexpected clean totals before publication: {deep_totals}")
    atomic_json(output_root / "CLEAN_VALIDATION_REPORT.json", {
        "passed": True,
        "tasks": deep_task_reports,
        "totals": deep_totals,
        "validated_at": utc_now(),
    })
    write_root_artifacts(source_root, output_root, plan, tasks)
    fingerprint_after = parent_fingerprint(source_root, tasks)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError("parent dataset changed while cleaning")
    atomic_json(output_root / "PARENT_FINGERPRINT.json", fingerprint_before)
    totals = read_json(output_root / "CLEANING_MANIFEST.json")["totals"]
    log(f"ALL DONE {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
