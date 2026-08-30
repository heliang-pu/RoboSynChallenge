#!/usr/bin/env python3
"""Merge seeded-clean and official-pruned LeRobot v2.1 datasets by task.

The seeded-clean schema is canonical. Official Parquet files are renamed,
projected, and safely cast to that schema; all videos are hard-linked without
transcoding. Neither source tree is modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
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
    from clean_seeded_v21 import aggregate_stats, atomic_json, numeric_stats, write_jsonl
except ImportError:
    from scripts.clean_seeded_v21 import aggregate_stats, atomic_json, numeric_stats, write_jsonl


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STAT_KEYS = ("min", "max", "mean", "std", "count")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def without_quantiles(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: stats[key] for key in STAT_KEYS if key in stats}


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


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.column_names.index(name)
    return table.set_column(index, name, pa.array(values))


def official_aliases(task: str) -> tuple[dict[str, str], dict[str, str]]:
    if task != "click_bell":
        return {}, {}
    parquet_aliases = {"observation.state": "observation.qpos"}
    video_aliases = {
        "observation.images.cam_high": "cam_high.color",
        "observation.images.cam_left_wrist": "cam_left_wrist.color",
        "observation.images.cam_right_wrist": "cam_right_wrist.color",
    }
    return parquet_aliases, video_aliases


def cast_official_table(
    task: str,
    table: pa.Table,
    canonical_schema: pa.Schema,
    episode_index: int,
    global_offset: int,
) -> pa.Table:
    parquet_aliases, _ = official_aliases(task)
    arrays = []
    for field in canonical_schema:
        source_name = parquet_aliases.get(field.name, field.name)
        if source_name not in table.column_names:
            raise KeyError(f"{task}: official parquet missing {source_name}")
        arrays.append(table[source_name])
    projected = pa.table(arrays, names=canonical_schema.names)
    projected = projected.cast(canonical_schema, safe=True)
    length = projected.num_rows
    projected = replace_column(
        projected, "episode_index", np.full(length, episode_index, dtype=np.int64)
    )
    projected = replace_column(
        projected, "frame_index", np.arange(length, dtype=np.int64)
    )
    projected = replace_column(
        projected, "index", np.arange(global_offset, global_offset + length, dtype=np.int64)
    )
    projected = replace_column(
        projected, "task_index", np.zeros(length, dtype=np.int64)
    )
    return projected.replace_schema_metadata(None)


def stats_from_table(table: pa.Table) -> dict[str, Any]:
    output = {}
    frame_count = table.num_rows
    for name in table.column_names:
        values = np.asarray(table[name].to_pylist())
        # LeRobot v2.1 episode stats reduce every leading/sample dimension and
        # retain only the final feature dimension (pose [T,4,4] -> [4],
        # contact [T,N,3] -> [3]). Match the seeded-clean stats convention.
        if values.ndim > 2:
            values = values.reshape(-1, values.shape[-1])
        output[name] = numeric_stats(values)
        # v2.1 count means dataset frames, even when min/mean/std reduce extra
        # contact/pose axes as well.
        output[name]["count"] = [frame_count]
    return output


def validate_sources(task: str, seeded: Path, official: Path) -> tuple[dict, dict, list, list]:
    seeded_info = read_json(seeded / "meta" / "info.json")
    official_info = read_json(official / "meta" / "info.json")
    if seeded_info.get("codebase_version") != "v2.1" or official_info.get("codebase_version") != "v2.1":
        raise ValueError(f"{task}: both sources must be v2.1")
    for key in ("fps", "robot_type", "chunks_size"):
        if seeded_info.get(key) != official_info.get(key):
            raise ValueError(f"{task}: source mismatch in {key}")
    seeded_tasks = read_jsonl(seeded / "meta" / "tasks.jsonl")
    official_tasks = read_jsonl(official / "meta" / "tasks.jsonl")
    if seeded_tasks != official_tasks:
        raise ValueError(f"{task}: prompt/task metadata mismatch")
    seeded_episodes = read_jsonl(seeded / "meta" / "episodes.jsonl")
    official_episodes = read_jsonl(official / "meta" / "episodes.jsonl")
    if len(seeded_episodes) != int(seeded_info["total_episodes"]):
        raise ValueError(f"{task}: seeded episode count mismatch")
    if len(official_episodes) != int(official_info["total_episodes"]):
        raise ValueError(f"{task}: official episode count mismatch")
    return seeded_info, official_info, seeded_episodes, official_episodes


def build_task(
    task: str,
    seeded_root: Path,
    official_root: Path,
    output_root: Path,
) -> Path:
    seeded = seeded_root / task
    official = official_root / f"cobotmagic_Sim_{task}"
    partial = output_root / f".{task}.partial"
    final = output_root / task
    if (final / ".complete.json").is_file():
        log(f"SKIP built/published {task}")
        return final
    if (partial / ".build_complete.json").is_file():
        marker = read_json(partial / ".build_complete.json")
        log(f"RESUME built {task}: {marker['episodes']} episodes")
        return partial
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    seeded_info, official_info, seeded_episodes, official_episodes = validate_sources(
        task, seeded, official
    )
    canonical_info = json.loads(json.dumps(seeded_info))
    canonical_schema = pq.read_schema(
        data_path(seeded, seeded_info, 0)
    ).remove_metadata()
    seeded_stats_rows = read_jsonl(seeded / "meta" / "episodes_stats.jsonl")
    official_stats_rows = {
        int(row["episode_index"]): row["stats"]
        for row in read_jsonl(official / "meta" / "episodes_stats.jsonl")
    }
    seeded_sidecar = read_json(seeded / "episode_success.json")
    if len(seeded_sidecar["episodes"]) != len(seeded_episodes):
        raise ValueError(f"{task}: seeded sidecar count mismatch")
    _, official_video_aliases = official_aliases(task)

    output_episodes: list[dict[str, Any]] = []
    output_stats_rows: list[dict[str, Any]] = []
    aggregate_inputs: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    sidecar_records: list[dict[str, Any]] = []
    hardlinked_data = 0
    hardlinked_videos = 0

    # Seeded-clean is first, so its episode/global indices remain unchanged.
    for episode_index, episode in enumerate(seeded_episodes):
        source_data = data_path(seeded, seeded_info, episode_index)
        destination_data = data_path(partial, canonical_info, episode_index)
        hardlink(source_data, destination_data)
        hardlinked_data += 1
        for camera in CAMERAS:
            hardlink(
                video_path(seeded, seeded_info, camera, episode_index),
                video_path(partial, canonical_info, camera, episode_index),
            )
            hardlinked_videos += 1
        output_episodes.append(dict(episode))
        stats = {
            key: without_quantiles(value)
            for key, value in seeded_stats_rows[episode_index]["stats"].items()
        }
        output_stats_rows.append({"episode_index": episode_index, "stats": stats})
        aggregate_inputs.append(stats)
        record = dict(seeded_sidecar["episodes"][episode_index])
        record["merge_source"] = "seeded_clean"
        sidecar_records.append(record)
        provenance.append(
            {
                "episode_index": episode_index,
                "source_kind": "seeded_clean",
                "source_dataset": str(seeded),
                "source_episode_index": episode_index,
                "seed": record.get("seed"),
            }
        )

    global_index = int(seeded_info["total_frames"])
    offset = len(seeded_episodes)
    for official_episode_index, episode in enumerate(official_episodes):
        new_episode = offset + official_episode_index
        source_table = pq.read_table(
            data_path(official, official_info, official_episode_index)
        )
        table = cast_official_table(
            task, source_table, canonical_schema, new_episode, global_index
        )
        length = int(episode["length"])
        if table.num_rows != length:
            raise ValueError(f"{task}: official episode {official_episode_index} row mismatch")
        destination_data = data_path(partial, canonical_info, new_episode)
        destination_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination_data)

        for camera in CAMERAS:
            source_camera = official_video_aliases.get(camera, camera)
            hardlink(
                video_path(official, official_info, source_camera, official_episode_index),
                video_path(partial, canonical_info, camera, new_episode),
            )
            hardlinked_videos += 1

        output_episodes.append(
            {"episode_index": new_episode, "tasks": episode["tasks"], "length": length}
        )
        stats = stats_from_table(table)
        source_video_stats = official_stats_rows[official_episode_index]
        for camera in CAMERAS:
            source_camera = official_video_aliases.get(camera, camera)
            stats[camera] = without_quantiles(source_video_stats[source_camera])
        if set(stats) != set(canonical_info["features"]):
            raise ValueError(
                f"{task}: canonical stats keys mismatch for official episode {official_episode_index}"
            )
        output_stats_rows.append({"episode_index": new_episode, "stats": stats})
        aggregate_inputs.append(stats)
        sidecar_records.append(
            {
                "episode_index": new_episode,
                "success": None,
                "success_label_available": False,
                "env_steps": length,
                "merge_source": "official_pruned",
                "source_dataset": str(official),
                "source_episode_index": official_episode_index,
                "seed_available": False,
            }
        )
        provenance.append(
            {
                "episode_index": new_episode,
                "source_kind": "official_pruned",
                "source_dataset": str(official),
                "source_episode_index": official_episode_index,
                "seed": None,
            }
        )
        global_index += length
        if (official_episode_index + 1) % 200 == 0:
            log(
                f"{task}: normalized official {official_episode_index + 1}/{len(official_episodes)}"
            )

    total_episodes = len(output_episodes)
    canonical_info["total_episodes"] = total_episodes
    canonical_info["total_frames"] = global_index
    canonical_info["total_videos"] = total_episodes * 3
    canonical_info["total_chunks"] = (
        (total_episodes - 1) // int(canonical_info["chunks_size"]) + 1
    )
    canonical_info["splits"] = {"train": f"0:{total_episodes}"}
    (partial / "meta").mkdir(parents=True, exist_ok=True)
    atomic_json(partial / "meta" / "info.json", canonical_info)
    shutil.copy2(seeded / "meta" / "tasks.jsonl", partial / "meta" / "tasks.jsonl")
    write_jsonl(partial / "meta" / "episodes.jsonl", output_episodes)
    write_jsonl(partial / "meta" / "episodes_stats.jsonl", output_stats_rows)
    atomic_json(partial / "meta" / "stats.json", aggregate_stats(aggregate_inputs))
    atomic_json(
        partial / "episode_success.json",
        {
            "labels_field": "episode_success",
            "saved_episode_count": total_episodes,
            "all_success": None,
            "note": "Seeded-clean success labels are retained; official-pruned success labels were not available.",
            "sources": [
                {"kind": "seeded_clean", "dataset": str(seeded), "episodes": len(seeded_episodes)},
                {"kind": "official_pruned", "dataset": str(official), "episodes": len(official_episodes)},
            ],
            "episodes": sidecar_records,
        },
    )
    write_jsonl(partial / "EPISODE_PROVENANCE.jsonl", provenance)
    atomic_json(
        partial / "MERGE_PROVENANCE.json",
        {
            "task": task,
            "canonical_schema_source": str(seeded),
            "seeded_clean": {
                "dataset": str(seeded),
                "episodes": len(seeded_episodes),
                "frames": int(seeded_info["total_frames"]),
            },
            "official_pruned": {
                "dataset": str(official),
                "episodes": len(official_episodes),
                "frames": int(official_info["total_frames"]),
                "schema_normalization": {
                    "parquet_aliases": official_aliases(task)[0],
                    "video_aliases": official_aliases(task)[1],
                    "projected_features": list(canonical_info["features"]),
                    "safe_arrow_cast": True,
                },
            },
            "combined": {
                "episodes": total_episodes,
                "frames": global_index,
                "videos": total_episodes * 3,
            },
            "hardlinked_seeded_parquets": hardlinked_data,
            "hardlinked_videos": hardlinked_videos,
            "video_copies": 0,
            "built_at": utc_now(),
        },
    )
    atomic_json(
        partial / ".build_complete.json",
        {
            "status": "build_complete",
            "task": task,
            "episodes": total_episodes,
            "frames": global_index,
            "videos": total_episodes * 3,
            "built_at": utc_now(),
        },
    )
    log(f"BUILT {task}: {total_episodes} episodes, {global_index} frames")
    return partial


def validation_matches(report: dict[str, Any], episodes: int, frames: int) -> bool:
    return bool(
        report.get("passed")
        and int(report.get("total_episodes", -1)) == episodes
        and int(report.get("total_frames", -1)) == frames
        and int(report.get("video_files", -1)) == episodes * 3
        and report.get("gates", {}).get("parquet_and_counts") == "pass"
        and report.get("gates", {}).get("videos") == "pass"
        and report.get("gates", {}).get("lerobot_training_read") == "pass"
    )


def validate_task_invariants(task: str, root: Path) -> dict[str, Any]:
    info = read_json(root / "meta" / "info.json")
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    stats_rows = read_jsonl(root / "meta" / "episodes_stats.jsonl")
    provenance = read_jsonl(root / "EPISODE_PROVENANCE.jsonl")
    sidecar = read_json(root / "episode_success.json")
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    if len(episodes) != total_episodes or len(stats_rows) != total_episodes:
        raise ValueError(f"{task}: metadata row count mismatch")
    if len(provenance) != total_episodes or len(sidecar["episodes"]) != total_episodes:
        raise ValueError(f"{task}: provenance/sidecar count mismatch")
    if [int(row["episode_index"]) for row in episodes] != list(range(total_episodes)):
        raise ValueError(f"{task}: episode indices not contiguous")
    if [int(row["episode_index"]) for row in provenance] != list(range(total_episodes)):
        raise ValueError(f"{task}: provenance indices not contiguous")
    if sum(int(row["length"]) for row in episodes) != total_frames:
        raise ValueError(f"{task}: total frame mismatch")
    if set(read_json(root / "meta" / "stats.json")) != set(info["features"]):
        raise ValueError(f"{task}: global stats feature mismatch")
    schemas = {
        str(pq.read_schema(path).remove_metadata())
        for path in (root / "data").glob("**/*.parquet")
    }
    if len(schemas) != 1:
        raise ValueError(f"{task}: multiple parquet schema variants")
    data_count = sum(1 for _ in (root / "data").glob("**/*.parquet"))
    video_count = sum(1 for _ in (root / "videos").glob("**/*.mp4"))
    if data_count != total_episodes or video_count != total_episodes * 3:
        raise ValueError(f"{task}: physical file count mismatch")
    report = read_json(root / "VALIDATION_REPORT.json")
    if not validation_matches(report, total_episodes, total_frames):
        raise ValueError(f"{task}: strict validation mismatch")
    return {
        "episodes": total_episodes,
        "frames": total_frames,
        "videos": total_episodes * 3,
        "schema_variants": 1,
        "provenance_verified": True,
    }


def validate_and_publish(
    task: str,
    partial: Path,
    output_root: Path,
    validation_python: Path,
    validator: Path,
    cache_root: Path,
) -> Path:
    info = read_json(partial / "meta" / "info.json")
    episodes, frames = int(info["total_episodes"]), int(info["total_frames"])
    report_path = partial / "VALIDATION_REPORT.json"
    report = read_json(report_path) if report_path.is_file() else {}
    if not validation_matches(report, episodes, frames):
        environment = os.environ.copy()
        environment["HF_HOME"] = str(cache_root / task / "hf")
        environment["HF_DATASETS_CACHE"] = str(cache_root / task / "hf" / "datasets")
        command = [
            str(validation_python),
            str(validator),
            str(partial),
            "--expected-episodes",
            str(episodes),
            "--action-horizon",
            "50",
            "--report",
            str(report_path),
            "--skip-video-decode",
        ]
        log(f"VALIDATE {task}")
        result = subprocess.run(command, env=environment)
        report = read_json(report_path) if report_path.is_file() else {}
    if not validation_matches(report, episodes, frames):
        raise RuntimeError(f"{task}: validation report did not pass")
    invariant = validate_task_invariants(task, partial)
    atomic_json(partial / "COMBINED_INVARIANTS_REPORT.json", invariant)
    atomic_json(
        partial / ".complete.json",
        {
            "status": "complete",
            "task": task,
            "episodes": episodes,
            "frames": frames,
            "videos": episodes * 3,
            "validated": True,
            "completed_at": utc_now(),
        },
    )
    final = output_root / task
    os.replace(partial, final)
    report = read_json(final / "VALIDATION_REPORT.json")
    old_path = report["dataset"]
    report["dataset"] = str(final)
    report["validated_before_atomic_rename"] = old_path
    atomic_json(final / "VALIDATION_REPORT.json", report)
    log(f"PUBLISHED {task}")
    return final


def write_root(task_reports: dict[str, dict[str, Any]], output_root: Path, seeded_root: Path, official_root: Path) -> None:
    totals = {
        "tasks": len(task_reports),
        "episodes": sum(row["episodes"] for row in task_reports.values()),
        "frames": sum(row["frames"] for row in task_reports.values()),
        "videos": sum(row["videos"] for row in task_reports.values()),
    }
    expected = {"tasks": 10, "episodes": 19510, "frames": 5612112, "videos": 58530}
    if totals != expected:
        raise RuntimeError(f"combined totals mismatch: {totals} != {expected}")
    manifest = {
        "status": "complete",
        "format": "LeRobot v2.1",
        "order": ["seeded_clean", "official_pruned"],
        "seeded_clean_root": str(seeded_root),
        "official_pruned_root": str(official_root),
        "tasks": task_reports,
        "totals": totals,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "MERGE_MANIFEST.json", manifest)
    atomic_json(output_root / ".complete.json", {"status": "complete", **totals, "completed_at": manifest["completed_at"]})
    lines = [
        "# Seeded-clean + official-pruned LeRobot v2.1",
        "",
        f"Seeded clean source: `{seeded_root}`",
        f"Official pruned source: `{official_root}`",
        "",
        "Seeded-clean episodes come first; official-pruned episodes are appended after schema normalization.",
        "All videos are hard-linked and were not transcoded.",
        "",
        "| Task | Episodes | Frames | Videos |",
        "|---|---:|---:|---:|",
    ]
    for task, row in sorted(task_reports.items()):
        lines.append(f"| {task} | {row['episodes']} | {row['frames']} | {row['videos']} |")
    lines.extend(["", f"Totals: {totals['episodes']} episodes, {totals['frames']} frames, {totals['videos']} videos.", ""])
    (output_root / "README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeded-clean-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-python", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--build-workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeded_root = args.seeded_clean_root.resolve()
    official_root = args.official_root.resolve()
    output_root = args.output_root.resolve()
    for source in (seeded_root, official_root):
        if source == output_root or source in output_root.parents or output_root in source.parents:
            raise ValueError("output must be a new sibling tree")
    if output_root.exists() and not args.resume:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = sorted(
        path.name
        for path in seeded_root.iterdir()
        if (path / "meta" / "info.json").is_file()
    )
    cache_root = Path("/tmp/robosyn_clean_official_merge_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    partials: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.build_workers) as pool:
        futures = {
            pool.submit(build_task, task, seeded_root, official_root, output_root): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            partials[task] = future.result()
    task_reports = {}
    for task in tasks:
        final = validate_and_publish(
            task,
            partials[task],
            output_root,
            args.validation_python.absolute(),
            args.validator.absolute(),
            cache_root,
        )
        task_reports[task] = validate_task_invariants(task, final)
    write_root(task_reports, output_root, seeded_root, official_root)
    log(f"ALL DONE {read_json(output_root / 'MERGE_MANIFEST.json')['totals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
