#!/usr/bin/env python3
"""Read-only multimodal QC triage for LeRobot datasets stored under Syn.

The script intentionally never changes a dataset.  It combines end-frame pose
checks with three-camera end-frame contact sheets so a human can quickly
inspect potentially dirty episodes before deciding whether to delete them.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
POSE_COLUMNS = {
    "sample_loading": ("cube_pose", "rack_pose"),
    "items_handover": ("pen_pose", "holder_pose"),
    "manipulate_pipette": ("pipette_pose", "beaker1_pose"),
    "water_pouring": ("bottle_pose", "cup_pose"),
}


def task_name(dataset: Path) -> str:
    name = dataset.name.lower()
    full = str(dataset).lower()
    if "sample_loading" in full:
        return "sample_loading"
    if "items_handover" in name:
        return "items_handover"
    if "manipulate_pipette" in full:
        return "manipulate_pipette"
    if "water_pouring" in name:
        return "water_pouring"
    if "click_bell" in full:
        return "click_bell"
    if "table_rearrangement" in full:
        return "table_rearrangement"
    return name


def dataset_roots(root: Path):
    for info in sorted(root.glob("**/meta/info.json")):
        dataset = info.parent.parent
        if "failed_old_success_predicate" in dataset.name:
            continue
        yield dataset


def rotation_angle(axis: np.ndarray, expected: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(axis, expected), -1.0, 1.0))))


def score_sample_loading(cube: np.ndarray, rack: np.ndarray):
    rr = rack[:3, :3]
    rel_center = rr.T @ (cube[:3, 3] - rack[:3, 3])
    rel_axis = rr.T @ cube[:3, 2]
    tube_half = 0.09751
    rack_top, rack_bottom = 0.03964, -0.03964
    safe_z = max(float(rel_axis[2]), 1e-6)
    top_xy = rel_center[:2] + (rack_top - rel_center[2]) / safe_z * rel_axis[:2]
    hole_x = 0.0123 + 0.02218 * np.arange(-6, 6)
    hole_y = -0.0016 + 0.0209 * np.arange(-2, 3)
    holes = np.array([(x, y) for x in hole_x for y in hole_y])
    hole_dist = float(np.linalg.norm(holes - top_xy, axis=1).min())
    tube_angle = rotation_angle(rel_axis, np.array([0.0, 0.0, 1.0]))
    rack_angle = rotation_angle(rr[:, 2], np.array([0.0, 0.0, 1.0]))
    tube_bottom = float(rel_center[2] - tube_half * rel_axis[2])
    depth = rack_top - tube_bottom
    ok = hole_dist <= 0.009 and depth >= 0.003 and tube_bottom >= rack_bottom - 0.012 and tube_angle <= 10.5 and rack_angle <= 10.0
    score = max(hole_dist / 0.009, max(0.0, 0.003 - depth) / 0.003, tube_angle / 10.5, rack_angle / 10.0)
    detail = f"hole={hole_dist*1000:.1f}mm depth={depth*1000:.1f}mm tube={tube_angle:.1f}deg rack={rack_angle:.1f}deg"
    return ok, score, detail


def score_handover(pen: np.ndarray, holder: np.ndarray):
    dist = float(np.linalg.norm(pen[:2, 3] - holder[:2, 3]))
    pen_angle = rotation_angle(pen[:3, 0], np.array([0.0, 0.0, 1.0]))
    holder_angle = rotation_angle(holder[:3, 1], np.array([0.0, 0.0, 1.0]))
    # The official task marks a pen/holder as fallen only at 75 degrees.
    # Use the same threshold rather than the stricter 10-degree rack rule.
    ok = dist <= 0.03 and pen_angle < 75.0 and holder_angle < 75.0
    score = max(dist / 0.03, pen_angle / 75.0, holder_angle / 75.0)
    return ok, score, f"pen-holder={dist*1000:.1f}mm pen={pen_angle:.1f}deg holder={holder_angle:.1f}deg"


def score_pipette(pipette: np.ndarray, beaker: np.ndarray):
    beaker_angle = rotation_angle(beaker[:3, 2], np.array([0.0, 0.0, 1.0]))
    affine = bool(np.allclose(pipette[3], [0, 0, 0, 1], atol=1e-4))
    # The recorder has historically written zero pipette poses.  Flag it as
    # metadata quality, not as a task failure, and let images decide it.
    ok = beaker_angle < 20.0
    score = max(beaker_angle / 20.0, 1.1 if not affine else 0.0)
    tag = "pipette_pose_invalid" if not affine else "pipette_pose_ok"
    return ok, score, f"beaker={beaker_angle:.1f}deg {tag}"


def score_water(bottle: np.ndarray, cup: np.ndarray):
    cup_angle = rotation_angle(cup[:3, 2], np.array([0.0, 0.0, 1.0]))
    bottle_angle = rotation_angle(bottle[:3, 1], np.array([0.0, 0.0, 1.0]))
    ok = cup_angle < 45.0 and bottle_angle < 45.0
    score = max(cup_angle / 45.0, bottle_angle / 45.0)
    return ok, score, f"cup={cup_angle:.1f}deg bottle={bottle_angle:.1f}deg"


def final_poses(dataset: Path, columns: tuple[str, ...]):
    finals = {}
    for path in sorted((dataset / "data").glob("**/*.parquet")):
        try:
            schema = pq.read_schema(path)
        except Exception:
            continue
        use = ["episode_index", *[c for c in columns if c in schema.names]]
        if len(use) <= 1:
            return {}
        table = pq.read_table(path, columns=use)
        ids = table["episode_index"].to_numpy()
        vals = {c: np.asarray(table[c].to_pylist(), dtype=np.float64) for c in use[1:]}
        for index in np.flatnonzero(np.r_[ids[1:] != ids[:-1], True]):
            finals[int(ids[index])] = {c: vals[c][index] for c in vals}
    return finals


def episode_records(dataset: Path):
    rows = []
    for path in sorted((dataset / "meta" / "episodes").glob("**/*.parquet")):
        try:
            table = pq.read_table(path)
        except Exception:
            continue
        rows.extend(table.to_pylist())
    return {int(row["episode_index"]): row for row in rows}


def evaluate(dataset: Path):
    task = task_name(dataset)
    columns = POSE_COLUMNS.get(task, ())
    finals = final_poses(dataset, columns) if columns else {}
    records = episode_records(dataset)
    results = []
    for ep, record in records.items():
        status, score, detail = "review", 0.0, "no object pose recorded; visual review only"
        poses = finals.get(ep, {})
        if task == "sample_loading" and len(poses) == 2:
            ok, score, detail = score_sample_loading(poses["cube_pose"], poses["rack_pose"])
            status = "task_failed" if not ok else "pass"
        elif task == "items_handover" and len(poses) == 2:
            ok, score, detail = score_handover(poses["pen_pose"], poses["holder_pose"])
            status = "task_failed" if not ok else "pass"
        elif task == "manipulate_pipette" and len(poses) == 2:
            ok, score, detail = score_pipette(poses["pipette_pose"], poses["beaker1_pose"])
            status = "task_failed" if not ok else "review"
        elif task == "water_pouring" and len(poses) == 2:
            ok, score, detail = score_water(poses["bottle_pose"], poses["cup_pose"])
            status = "task_failed" if not ok else "review"
        results.append({"dataset": str(dataset), "task": task, "episode_index": ep, "status": status, "score": float(score), "detail": detail, "record": record})
    return results


def video_path(dataset: Path, camera: str, record: dict) -> tuple[Path, float]:
    key = f"videos/observation.images.{camera}"
    chunk, file_idx = record.get(f"{key}/chunk_index"), record.get(f"{key}/file_index")
    timestamp = record.get(f"{key}/to_timestamp")
    path = dataset / "videos" / f"observation.images.{camera}" / f"chunk-{int(chunk):03d}" / f"file-{int(file_idx):03d}.mp4"
    return path, max(0.0, float(timestamp) - 0.08)


def read_frame(path: Path, timestamp: float):
    # NAS videos use AV1.  OpenCV's bundled FFmpeg cannot reliably decode it,
    # while the system ffmpeg has a software AV1 decoder.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
        "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
    ]
    try:
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=20).stdout
        if not output:
            return None
        return Image.open(io.BytesIO(output)).convert("RGB")
    except (subprocess.SubprocessError, OSError):
        return None


def draw_sheet(candidate: dict, out: Path):
    dataset = Path(candidate["dataset"])
    tiles = []
    for camera in CAMERAS:
        path, timestamp = video_path(dataset, camera, candidate["record"])
        tile = read_frame(path, timestamp)
        if tile is None:
            tile = Image.new("RGB", (640, 480), "black")
        tile.thumbnail((480, 360))
        framed = Image.new("RGB", (480, 390), "white")
        framed.paste(tile, ((480 - tile.width) // 2, 26))
        ImageDraw.Draw(framed).text((8, 5), camera, fill="black")
        tiles.append(framed)
    header = f"{candidate['task']} | episode {candidate['episode_index']} | {candidate['status']} | score={candidate['score']:.2f}"
    image = Image.new("RGB", (1440, 430), "white")
    drawer = ImageDraw.Draw(image)
    drawer.text((8, 5), header, fill="red" if candidate["status"] == "task_failed" else "black")
    drawer.text((8, 23), candidate["detail"], fill="black")
    for index, tile in enumerate(tiles):
        image.paste(tile, (index * 480, 40))
    image.save(out, quality=92)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[-110:]


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    """Select a compact temporal spread for visual-only tasks."""
    if len(rows) <= count:
        return rows
    ordered = sorted(rows, key=lambda item: item["episode_index"])
    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-dataset", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    contact_dir = args.out / "three_view_candidates"
    contact_dir.mkdir(exist_ok=True)

    all_results = []
    structural_errors = []
    for dataset in dataset_roots(args.root):
        for path in list((dataset / "data").glob("**/*.parquet")) + list((dataset / "meta" / "episodes").glob("**/*.parquet")):
            try:
                pq.read_schema(path)
            except Exception as exc:
                structural_errors.append((str(dataset), str(path), type(exc).__name__))
        all_results.extend(evaluate(dataset))

    selected = []
    by_dataset = defaultdict(list)
    for row in all_results:
        if row["status"] == "task_failed":
            by_dataset[row["dataset"]].append(row)
    # When no hard failure is visible from pose data, include the highest-score
    # records for human review.  This covers missing pose fields and visual-only tasks.
    for dataset, group in defaultdict(list, {k: [r for r in all_results if r["dataset"] == k] for k in {r["dataset"] for r in all_results}}).items():
        hard = by_dataset.get(dataset, [])
        remaining = [r for r in group if r not in hard]
        picks = sorted(hard, key=lambda r: r["score"], reverse=True)
        if len(picks) < args.max_per_dataset:
            missing = args.max_per_dataset - len(picks)
            max_score = max((row["score"] for row in remaining), default=0.0)
            min_score = min((row["score"] for row in remaining), default=0.0)
            if math.isclose(max_score, min_score, abs_tol=1e-8):
                picks.extend(evenly_spaced(remaining, missing))
            else:
                picks.extend(sorted(remaining, key=lambda r: r["score"], reverse=True)[:missing])
        selected.extend(picks[: args.max_per_dataset])

    selected.sort(key=lambda r: (r["status"] != "task_failed", -r["score"], r["dataset"], r["episode_index"]))
    fieldnames = ["dataset", "task", "episode_index", "status", "score", "detail", "three_view_image"]
    with (args.out / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in selected:
            filename = f"{slug(Path(candidate['dataset']).relative_to(args.root).as_posix())}_ep_{candidate['episode_index']:06d}.jpg"
            draw_sheet(candidate, contact_dir / filename)
            writer.writerow({key: candidate.get(key, "") for key in fieldnames} | {"three_view_image": str(contact_dir / filename)})

    totals = defaultdict(int)
    for row in all_results:
        totals[(row["task"], row["status"])] += 1
    with (args.out / "summary.txt").open("w") as handle:
        handle.write(f"datasets={len({r['dataset'] for r in all_results})}\n")
        handle.write(f"episodes={len(all_results)}\n")
        for (task, status), count in sorted(totals.items()):
            handle.write(f"{task}\t{status}\t{count}\n")
        handle.write(f"rendered_candidates={len(selected)}\n")
        handle.write(f"structural_parquet_errors={len(structural_errors)}\n")
    with (args.out / "structural_errors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "file", "error"))
        writer.writerows(structural_errors)


if __name__ == "__main__":
    main()
