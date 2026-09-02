#!/usr/bin/env python3
"""Convert seeded LeRobot v3 batches to v2.1 and merge them by task.

The source tree is treated as immutable.  Each non-empty batch is converted
into a staging directory, compatible batches are merged into one v2.1 dataset
per task, and the merged dataset is validated with the target LeRobot runtime
before it becomes visible under its final task name.

The collection-side ``episode_success.json`` files are not handled by either
the v3-to-v2.1 converter or the v2.1 merger.  This driver therefore rebuilds a
merged sidecar while preserving the original seed and source episode identity.
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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import pyarrow.parquet as pq


EXPECTED_TASKS = (
    "click_bell",
    "drawer_open_place",
    "handle_basket",
    "item_assembly",
    "items_handover",
    "manipulate_pipette",
    "mixer_operating",
    "sample_loading",
    "table_rearrangement",
    "water_pouring",
)


@dataclass(frozen=True)
class SourceBatch:
    task: str
    batch_index: int
    batch_dir: Path
    dataset: Path
    episodes: int
    frames: int
    master_seed: int
    sidecar: dict[str, Any]
    complete: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.task}/batch_{self.batch_index:02d}"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def parse_batch_index(batch_dir: Path) -> int:
    try:
        return int(batch_dir.name.removeprefix("batch_"))
    except ValueError as exc:
        raise ValueError(f"invalid batch directory name: {batch_dir}") from exc


def validate_source_candidate(
    task: str, batch_dir: Path, dataset: Path
) -> tuple[SourceBatch | None, str | None]:
    info_path = dataset / "meta" / "info.json"
    try:
        info = read_json(info_path)
    except Exception as exc:
        return None, f"unreadable info.json: {type(exc).__name__}: {exc}"
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    if episodes == 0:
        return None, "empty dataset root (0 episodes)"
    if info.get("codebase_version") != "v3.0":
        return None, f"unexpected codebase_version={info.get('codebase_version')!r}"
    required = (
        dataset / "meta" / "tasks.parquet",
        dataset / "episode_success.json",
        batch_dir / ".complete.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return None, f"missing required files: {missing}"
    if not list((dataset / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        return None, "missing episode metadata parquet"
    if not list((dataset / "data").glob("chunk-*/file-*.parquet")):
        return None, "missing data parquet"

    sidecar = read_json(dataset / "episode_success.json")
    complete = read_json(batch_dir / ".complete.json")
    records = sidecar.get("episodes", [])
    if sidecar.get("saved_episode_count") != episodes or len(records) != episodes:
        return None, "episode_success count does not match info.json"
    expected_indices = list(range(episodes))
    if sorted(int(record["episode_index"]) for record in records) != expected_indices:
        return None, "episode_success indices are not contiguous"
    if not all(bool(record.get("success")) for record in records):
        return None, "source contains an unsuccessful episode"
    master_seed = int(sidecar["master_seed"])
    if int(complete.get("seed", -1)) != master_seed:
        return None, "master_seed does not match .complete.json"

    return (
        SourceBatch(
            task=task,
            batch_index=parse_batch_index(batch_dir),
            batch_dir=batch_dir,
            dataset=dataset,
            episodes=episodes,
            frames=frames,
            master_seed=master_seed,
            sidecar=sidecar,
            complete=complete,
        ),
        None,
    )


def discover_sources(
    source_root: Path,
    expected_batches: int,
    expected_episodes_per_batch: int,
    allowed_missing: set[tuple[str, int]],
) -> tuple[dict[str, list[SourceBatch]], list[dict[str, Any]]]:
    found_tasks = {
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "batches").is_dir()
    }
    if found_tasks != set(EXPECTED_TASKS):
        raise RuntimeError(
            f"task set mismatch: expected={list(EXPECTED_TASKS)}, found={sorted(found_tasks)}"
        )

    selected: dict[str, list[SourceBatch]] = {}
    excluded: list[dict[str, Any]] = []
    for task in EXPECTED_TASKS:
        batches: list[SourceBatch] = []
        seen_indices: set[int] = set()
        for batch_dir in sorted((source_root / task / "batches").glob("batch_*")):
            batch_index = parse_batch_index(batch_dir)
            if batch_index in seen_indices:
                raise RuntimeError(f"duplicate batch index in {task}: {batch_index}")
            seen_indices.add(batch_index)
            valid: list[SourceBatch] = []
            info_paths = sorted(batch_dir.glob("*/meta/info.json"))
            for info_path in info_paths:
                candidate, reason = validate_source_candidate(task, batch_dir, info_path.parent.parent)
                if candidate is None:
                    excluded.append(
                        {
                            "task": task,
                            "batch_index": batch_index,
                            "dataset": str(info_path.parent.parent),
                            "reason": reason,
                        }
                    )
                else:
                    valid.append(candidate)
            if len(valid) != 1:
                raise RuntimeError(
                    f"{task}/batch_{batch_index:02d} has {len(valid)} valid non-empty dataset roots; "
                    f"candidates={info_paths}"
                )
            if valid[0].episodes != expected_episodes_per_batch:
                raise RuntimeError(
                    f"{valid[0].label} has {valid[0].episodes} episodes, "
                    f"expected {expected_episodes_per_batch}"
                )
            batches.append(valid[0])

        missing = set(range(expected_batches)) - seen_indices
        unexpected_missing = {(task, index) for index in missing} - allowed_missing
        if unexpected_missing:
            raise RuntimeError(f"unexpected missing batches: {sorted(unexpected_missing)}")
        unobserved_allowances = {
            item for item in allowed_missing if item[0] == task and item[1] not in missing
        }
        if unobserved_allowances:
            raise RuntimeError(f"allowed-missing entries are actually present: {sorted(unobserved_allowances)}")
        selected[task] = sorted(batches, key=lambda item: item.batch_index)

    observed_missing = {
        (task, index)
        for task, batches in selected.items()
        for index in set(range(expected_batches)) - {batch.batch_index for batch in batches}
    }
    if observed_missing != allowed_missing:
        raise RuntimeError(
            f"missing-batch set mismatch: observed={sorted(observed_missing)}, "
            f"allowed={sorted(allowed_missing)}"
        )
    return selected, excluded


def stage_is_valid(stage: Path, source: SourceBatch) -> bool:
    try:
        info = read_json(stage / "meta" / "info.json")
        episodes = read_jsonl(stage / "meta" / "episodes.jsonl")
    except Exception:
        return False
    return (
        info.get("codebase_version") == "v2.1"
        and int(info.get("total_episodes", -1)) == source.episodes
        and int(info.get("total_frames", -1)) == source.frames
        and len(episodes) == source.episodes
        and (stage / ".converted.json").is_file()
    )


def safe_remove_partial(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    root = output_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"refusing to remove path outside output root: {resolved}")
    if ".partial" not in path.name:
        raise RuntimeError(f"refusing to remove non-partial path: {path}")
    if path.exists():
        shutil.rmtree(path)


def tail(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "<log unavailable>"


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"last log lines from {log_path}:\n{tail(log_path)}"
        )


def convert_one(
    source: SourceBatch,
    staging_root: Path,
    converter: Path,
    runtime_python: Path,
    logs_root: Path,
    output_root: Path,
) -> Path:
    task_stage = staging_root / source.task
    final = task_stage / f"batch_{source.batch_index:02d}"
    partial = task_stage / f".batch_{source.batch_index:02d}.partial"
    if stage_is_valid(final, source):
        log(f"SKIP converted {source.label}")
        return final
    if final.exists():
        raise RuntimeError(f"invalid existing stage requires manual review: {final}")
    safe_remove_partial(partial, output_root)
    task_stage.mkdir(parents=True, exist_ok=True)
    command = [
        str(runtime_python),
        str(converter),
        "--root",
        str(source.dataset.parent),
        "--repo-id",
        source.dataset.name,
        "--output-root",
        str(partial),
    ]
    log(f"CONVERT {source.label} ({source.episodes} episodes, {source.frames} frames)")
    run_logged(command, logs_root / source.task / f"batch_{source.batch_index:02d}.log")
    atomic_json(
        partial / ".converted.json",
        {
            "source": str(source.dataset),
            "task": source.task,
            "batch_index": source.batch_index,
            "episodes": source.episodes,
            "frames": source.frames,
            "master_seed": source.master_seed,
            "completed_at": utc_now(),
        },
    )
    if not stage_is_valid(partial, source):
        raise RuntimeError(f"converted stage failed self-check: {partial}")
    os.replace(partial, final)
    return final


def merged_sidecar(sources: list[SourceBatch]) -> dict[str, Any]:
    merged_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    offset = 0
    for source in sources:
        records = sorted(source.sidecar["episodes"], key=lambda row: int(row["episode_index"]))
        start = offset
        for record in records:
            source_episode = int(record["episode_index"])
            merged = dict(record)
            merged["episode_index"] = offset
            merged["source_task"] = source.task
            merged["source_batch"] = source.batch_index
            merged["source_episode_index"] = source_episode
            merged["source_v3_relpath"] = (
                f"{source.task}/batches/batch_{source.batch_index:02d}/{source.dataset.name}"
            )
            merged_records.append(merged)
            offset += 1
        source_records.append(
            {
                "task": source.task,
                "batch_index": source.batch_index,
                "dataset": str(source.dataset),
                "master_seed": source.master_seed,
                "source_episodes": source.episodes,
                "source_frames": source.frames,
                "merged_episode_range": [start, offset - 1],
                "complete": source.complete,
                "gym_config": source.sidecar.get("gym_config"),
                "gym_config_sha1": source.sidecar.get("gym_config_sha1"),
                "git_commit": source.sidecar.get("git_commit"),
                "task_description": source.sidecar.get("task_description"),
                "info_sha256": sha256(source.dataset / "meta" / "info.json"),
                "episode_success_sha256": sha256(source.dataset / "episode_success.json"),
            }
        )
    seeds = [int(record["seed"]) for record in merged_records]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("episode seeds are not unique after merge")
    return {
        "labels_field": "episode_success",
        "collection_plan": "seeded_1000_20260827",
        "saved_episode_count": len(merged_records),
        "all_success": all(bool(record.get("success")) for record in merged_records),
        "source_batches": source_records,
        "episodes": merged_records,
    }


def video_path(dataset: Path, info: dict[str, Any], key: str, episode_index: int) -> Path:
    return dataset / info["video_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
        video_key=key,
        chunk_index=episode_index // int(info["chunks_size"]),
        file_index=episode_index,
    )


def inspect_video(path: Path, expected_frames: int, fps: float) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty video: {path}")
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise RuntimeError(f"expected one video stream: {path}")
        stream = container.streams.video[0]
        frame_count = int(stream.frames or 0)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else None
        )
        codec = str(stream.codec_context.codec.name)
        result = {
            "frames": frame_count,
            "duration": duration,
            "width": int(stream.width),
            "height": int(stream.height),
            "pix_fmt": str(stream.pix_fmt),
            "fps": float(stream.average_rate) if stream.average_rate is not None else 0.0,
            "codec": codec,
        }
    errors = []
    if result["frames"] != expected_frames:
        errors.append(f"frames={result['frames']} expected={expected_frames}")
    if result["width"] != 640 or result["height"] != 480:
        errors.append(f"size={result['width']}x{result['height']}")
    if result["pix_fmt"] != "yuv420p":
        errors.append(f"pix_fmt={result['pix_fmt']}")
    if abs(result["fps"] - fps) > 1e-6:
        errors.append(f"fps={result['fps']} expected={fps}")
    if result["codec"] not in {"av1", "dav1d", "libdav1d"}:
        errors.append(f"codec={result['codec']}")
    expected_duration = expected_frames / fps
    if result["duration"] is None or abs(result["duration"] - expected_duration) > 1.5 / fps:
        errors.append(f"duration={result['duration']} expected={expected_duration}")
    if errors:
        raise RuntimeError(f"video metadata mismatch at {path}: {'; '.join(errors)}")
    return result


def validate_stats_schemas_and_video_metadata(
    dataset: Path, expected_episodes: int, expected_frames: int, workers: int = 16
) -> dict[str, Any]:
    info = read_json(dataset / "meta" / "info.json")
    episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
    if len(episodes) != expected_episodes:
        raise RuntimeError(f"episode metadata count mismatch at {dataset}")
    data_paths = sorted((dataset / "data").glob("**/*.parquet"))
    if len(data_paths) != expected_episodes:
        raise RuntimeError(
            f"per-episode parquet count mismatch at {dataset}: {len(data_paths)} != {expected_episodes}"
        )
    schemas = {str(pq.read_schema(path).remove_metadata()) for path in data_paths}
    if len(schemas) != 1:
        raise RuntimeError(f"multiple parquet schema variants at {dataset}: {len(schemas)}")

    stats = read_json(dataset / "meta" / "stats.json")
    video_keys = {
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    }
    for key, values in stats.items():
        count = int(values["count"][0])
        if key not in video_keys and count != expected_frames:
            raise RuntimeError(
                f"global stats count mismatch for {key}: {count} != {expected_frames}"
            )
        if key in video_keys and not (0 < count <= expected_frames):
            raise RuntimeError(f"invalid sampled video stats count for {key}: {count}")

    expected_video_keys = {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }
    if video_keys != expected_video_keys:
        raise RuntimeError(f"unexpected video keys at {dataset}: {sorted(video_keys)}")
    fps = float(info["fps"])
    jobs: list[tuple[Path, int]] = []
    for record in episodes:
        episode_index = int(record["episode_index"])
        length = int(record["length"])
        for key in sorted(video_keys):
            jobs.append((video_path(dataset, info, key, episode_index), length))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(inspect_video, path, length, fps): path for path, length in jobs
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"video probe failed for {path}: {exc}") from exc
            if index % 500 == 0 or index == len(futures):
                log(f"VIDEO METADATA {dataset.name}: {index}/{len(futures)}")
    return {
        "parquet_files": len(data_paths),
        "parquet_schema_variants": len(schemas),
        "video_files": len(jobs),
        "video_metadata": "pass",
        "global_stats": "pass",
    }


def delivery_quick_check(dataset: Path, expected_episodes: int) -> None:
    info = read_json(dataset / "meta" / "info.json")
    if int(info.get("total_episodes", -1)) != expected_episodes:
        raise RuntimeError(f"delivery reopen episode mismatch: {dataset}")
    parquet_count = sum(1 for _ in (dataset / "data").glob("**/*.parquet"))
    video_count = sum(1 for _ in (dataset / "videos").glob("**/*.mp4"))
    if parquet_count != expected_episodes or video_count != expected_episodes * 3:
        raise RuntimeError(
            f"delivery file count mismatch at {dataset}: parquet={parquet_count}, video={video_count}"
        )
    episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
    for record in (episodes[0], episodes[-1]):
        episode_index = int(record["episode_index"])
        length = int(record["length"])
        for key, feature in info["features"].items():
            if feature.get("dtype") == "video":
                inspect_video(video_path(dataset, info, key, episode_index), length, float(info["fps"]))


def normalize_delivery_report(dataset: Path) -> None:
    report_path = dataset / "VALIDATION_REPORT.json"
    if not report_path.is_file():
        return
    report = read_json(report_path)
    old_path = report.get("dataset")
    report["dataset"] = str(dataset)
    if old_path and old_path != str(dataset):
        report["validated_before_atomic_rename"] = old_path
    atomic_json(report_path, report)


def merge_and_validate_task(
    task: str,
    sources: list[SourceBatch],
    stage_paths: list[Path],
    output_root: Path,
    merger: Path,
    validator: Path,
    runtime_python: Path,
    validation_python: Path,
    logs_root: Path,
    cache_root: Path,
) -> Path:
    final = output_root / task
    partial = output_root / f".{task}.partial"
    if (final / ".complete.json").is_file():
        info = read_json(final / "meta" / "info.json")
        expected_episodes = sum(source.episodes for source in sources)
        if info.get("codebase_version") != "v2.1" or int(info.get("total_episodes", -1)) != expected_episodes:
            raise RuntimeError(f"existing completed task is inconsistent: {final}")
        log(f"SKIP merged {task}")
        return final
    if final.exists():
        raise RuntimeError(f"existing task output has no completion marker: {final}")
    expected_episodes = sum(source.episodes for source in sources)
    expected_frames = sum(source.frames for source in sources)
    reusable_partial = False
    if partial.is_dir():
        try:
            partial_info = read_json(partial / "meta" / "info.json")
            partial_video_report = read_json(partial / "VIDEO_METADATA_REPORT.json")
            partial_sidecar = read_json(partial / "episode_success.json")
            reusable_partial = (
                partial_info.get("codebase_version") == "v2.1"
                and int(partial_info.get("total_episodes", -1)) == expected_episodes
                and int(partial_info.get("total_frames", -1)) == expected_frames
                and partial_video_report.get("video_metadata") == "pass"
                and partial_video_report.get("global_stats") == "pass"
                and int(partial_sidecar.get("saved_episode_count", -1)) == expected_episodes
            )
        except Exception:
            reusable_partial = False
    if reusable_partial:
        log(f"RESUME verified merge partial for {task}")
    else:
        safe_remove_partial(partial, output_root)
        merge_command = [str(runtime_python), str(merger), "--out", str(partial)] + [
            str(path) for path in stage_paths
        ]
        log(f"MERGE {task}: {len(sources)} batches")
        run_logged(merge_command, logs_root / task / "merge.log")

        sidecar = merged_sidecar(sources)
        atomic_json(partial / "episode_success.json", sidecar)
        atomic_json(
            partial / "SOURCE_BATCHES.json",
            {
                "task": task,
                "sources": sidecar["source_batches"],
                "excluded_failed_partial": True,
            },
        )
        atomic_json(
            partial / "MERGE_SOURCES.json",
            {
                "sources": [str(source.dataset) for source in sources],
                "source_kind": "durable_v3_roots",
                "episodes": expected_episodes,
                "frames": expected_frames,
            },
        )

        metadata_report = validate_stats_schemas_and_video_metadata(
            partial, expected_episodes, expected_frames
        )
        atomic_json(partial / "VIDEO_METADATA_REPORT.json", metadata_report)
    validation_report = partial / "VALIDATION_REPORT.json"
    validation_env = os.environ.copy()
    validation_env["HF_HOME"] = str(cache_root / task / "hf")
    validation_env["HF_DATASETS_CACHE"] = str(cache_root / task / "hf" / "datasets")
    def validation_report_matches(candidate: dict[str, Any]) -> bool:
        return bool(
            candidate.get("passed")
            and int(candidate.get("total_episodes", -1)) == expected_episodes
            and int(candidate.get("total_frames", -1)) == expected_frames
            and int(candidate.get("video_files", -1)) == expected_episodes * 3
            and candidate.get("gates", {}).get("videos") == "pass"
            and candidate.get("gates", {}).get("lerobot_training_read") == "pass"
        )

    report: dict[str, Any] | None = None
    if validation_report.is_file():
        try:
            candidate = read_json(validation_report)
            if validation_report_matches(candidate):
                report = candidate
                log(f"RESUME passed training validation for {task}")
        except Exception:
            report = None
    if report is None:
        validate_command = [
            str(validation_python),
            str(validator),
            str(partial),
            "--expected-episodes",
            str(expected_episodes),
            "--action-horizon",
            "50",
            "--report",
            str(validation_report),
        ]
        log(f"VALIDATE {task}: parquet, all three-view videos, and target training read")
        try:
            run_logged(validate_command, logs_root / task / "validate.log", env=validation_env)
        except RuntimeError:
            # Some old LeRobot/PyAV combinations can linger during interpreter
            # shutdown and receive SIGTERM even after every gate passed and the
            # validator atomically wrote its final report.  Accept only the
            # complete, count-matched report; otherwise retain the failure.
            if not validation_report.is_file():
                raise
            candidate = read_json(validation_report)
            if not validation_report_matches(candidate):
                raise
            log(
                f"WARNING validator process ended abnormally after writing a fully passed report for {task}"
            )
        report = read_json(validation_report)
    if not report.get("passed"):
        raise RuntimeError(f"validation report did not pass: {validation_report}")
    if int(report.get("total_frames", -1)) != expected_frames:
        raise RuntimeError(
            f"validation frame count mismatch for {task}: "
            f"{report.get('total_frames')} != {expected_frames}"
        )

    atomic_json(
        partial / ".complete.json",
        {
            "task": task,
            "format": "LeRobot v2.1",
            "episodes": expected_episodes,
            "frames": expected_frames,
            "source_batches": len(sources),
            "validated": True,
            "completed_at": utc_now(),
        },
    )
    os.replace(partial, final)
    normalize_delivery_report(final)
    log(f"DONE {task}: {expected_episodes} episodes, {expected_frames} frames")
    return final


def parse_allowed_missing(values: list[str]) -> set[tuple[str, int]]:
    output: set[tuple[str, int]] = set()
    for value in values:
        task, separator, batch = value.partition(":")
        if not separator:
            raise ValueError(f"invalid --allow-missing value {value!r}; expected TASK:BATCH")
        output.add((task, int(batch)))
    return output


def write_root_readme(output_root: Path, manifest: dict[str, Any]) -> None:
    rows = [
        "# Seeded 1000 — LeRobot v2.1 task-merged datasets",
        "",
        f"Source: `{manifest['source_root']}`",
        "",
        "The source v3.0 tree was not modified. Failed partial collections and empty dataset roots were excluded.",
        "Each task directory is an independently loadable LeRobot v2.1 dataset.",
        "`episode_success.json` preserves the original episode seed and source batch identity after renumbering.",
        "",
        "| Task | Batches | Episodes | Frames |",
        "|---|---:|---:|---:|",
    ]
    for task in EXPECTED_TASKS:
        item = manifest["tasks"][task]
        rows.append(f"| {task} | {item['batches']} | {item['episodes']} | {item['frames']} |")
    rows.append("")
    missing = manifest.get("allowed_missing_batches", [])
    if missing:
        rendered = ", ".join(
            f"`{item['task']}/batch_{int(item['batch_index']):02d}`" for item in missing
        )
        rows.append(f"Known incomplete source batches excluded: {rendered}.")
    else:
        rows.append("All ten expected batches are present for every task (1,000 episodes per task).")
    if manifest.get("excluded_dataset_roots"):
        rows.append(
            "A zero-episode `sample_loading/batch_09/cobotmagic_Sim_sample_loading_001` "
            "recorder shell was excluded."
        )
    rows.append("")
    (output_root / "README.md").write_text("\n".join(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--merger", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--validation-python", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--expected-batches", type=int, default=10)
    parser.add_argument("--expected-episodes-per-batch", type=int, default=100)
    parser.add_argument("--allow-missing", action="append", default=[])
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    for path, label in (
        (source_root, "source root"),
        (args.converter, "converter"),
        (args.merger, "merger"),
        (args.validator, "validator"),
        (args.runtime_python, "runtime python"),
        (args.validation_python, "validation python"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse output without --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / "_staging"
    logs_root = output_root / "_conversion_logs"
    cache_root = Path(os.environ.get("ROBOSYN_V21_CACHE", "/tmp/robosyn_seeded_v21_cache"))
    staging_root.mkdir(exist_ok=True)
    logs_root.mkdir(exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    allowed_missing = parse_allowed_missing(args.allow_missing)
    sources_by_task, excluded = discover_sources(
        source_root,
        args.expected_batches,
        args.expected_episodes_per_batch,
        allowed_missing,
    )
    total_batches = sum(len(items) for items in sources_by_task.values())
    total_episodes = sum(source.episodes for items in sources_by_task.values() for source in items)
    total_frames = sum(source.frames for items in sources_by_task.values() for source in items)
    all_episode_seeds = [
        int(record["seed"])
        for items in sources_by_task.values()
        for source in items
        for record in source.sidecar["episodes"]
    ]
    if len(all_episode_seeds) != len(set(all_episode_seeds)):
        raise RuntimeError("episode seeds are not globally unique")

    manifest: dict[str, Any] = {
        "status": "in_progress",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "created_at": utc_now(),
        "format": "LeRobot v2.1",
        "merge_unit": "task",
        "source_totals": {
            "batches": total_batches,
            "episodes": total_episodes,
            "frames": total_frames,
            "unique_episode_seeds": len(set(all_episode_seeds)),
        },
        "allowed_missing_batches": [
            {"task": task, "batch_index": batch} for task, batch in sorted(allowed_missing)
        ],
        "excluded_dataset_roots": excluded,
        "failed_partial_included": False,
        "scripts": {
            "driver": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "converter": {"path": str(args.converter.resolve()), "sha256": sha256(args.converter)},
            "merger": {"path": str(args.merger.resolve()), "sha256": sha256(args.merger)},
            "validator": {"path": str(args.validator.resolve()), "sha256": sha256(args.validator)},
        },
        "tasks": {},
    }
    for task, sources in sources_by_task.items():
        manifest["tasks"][task] = {
            "status": "pending",
            "batches": len(sources),
            "episodes": sum(source.episodes for source in sources),
            "frames": sum(source.frames for source in sources),
            "source_batch_indices": [source.batch_index for source in sources],
        }
    atomic_json(output_root / "CONVERSION_MANIFEST.json", manifest)

    for task in EXPECTED_TASKS:
        sources = sources_by_task[task]
        completed_task = output_root / task
        if (completed_task / ".complete.json").is_file():
            expected_task_episodes = sum(source.episodes for source in sources)
            delivery_quick_check(completed_task, expected_task_episodes)
            normalize_delivery_report(completed_task)
            manifest["tasks"][task]["status"] = "complete"
            manifest["tasks"][task]["completed_at"] = read_json(
                completed_task / ".complete.json"
            ).get("completed_at")
            atomic_json(output_root / "CONVERSION_MANIFEST.json", manifest)
            log(f"SKIP completed delivery {task}")
            continue
        log(f"TASK {task}: converting {len(sources)} batches with {args.workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    convert_one,
                    source,
                    staging_root,
                    args.converter.resolve(),
                    args.runtime_python.absolute(),
                    logs_root,
                    output_root,
                ): source
                for source in sources
            }
            stage_by_batch: dict[int, Path] = {}
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                stage_by_batch[source.batch_index] = future.result()
                log(f"CONVERTED {source.label}")
        stage_paths = [stage_by_batch[source.batch_index] for source in sources]
        merge_and_validate_task(
            task,
            sources,
            stage_paths,
            output_root,
            args.merger.resolve(),
            args.validator.resolve(),
            args.runtime_python.absolute(),
            args.validation_python.absolute(),
            logs_root,
            cache_root,
        )
        manifest["tasks"][task]["status"] = "complete"
        manifest["tasks"][task]["completed_at"] = utc_now()
        atomic_json(output_root / "CONVERSION_MANIFEST.json", manifest)
        if not args.keep_staging:
            task_stage = staging_root / task
            if task_stage.is_dir():
                shutil.rmtree(task_stage)
                log(f"CLEANED staging for {task}")
        delivery_quick_check(output_root / task, sum(source.episodes for source in sources))
        log(f"REOPENED delivery {task} after staging cleanup")

    if not args.keep_staging and staging_root.is_dir() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["output_totals"] = {
        "tasks": len(EXPECTED_TASKS),
        "episodes": total_episodes,
        "frames": total_frames,
    }
    atomic_json(output_root / "CONVERSION_MANIFEST.json", manifest)
    write_root_readme(output_root, manifest)
    atomic_json(
        output_root / ".complete.json",
        {
            "status": "complete",
            "tasks": len(EXPECTED_TASKS),
            "episodes": total_episodes,
            "frames": total_frames,
            "completed_at": utc_now(),
        },
    )
    log(f"ALL DONE: {len(EXPECTED_TASKS)} tasks, {total_episodes} episodes, {total_frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
