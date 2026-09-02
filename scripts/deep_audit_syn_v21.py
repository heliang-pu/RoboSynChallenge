#!/usr/bin/env python3
"""Read-only numeric and three-camera QC audit for Syn LeRobot v2.1 data."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def task_for(path: Path) -> str:
    text = str(path).lower()
    for task in ("sample_loading", "items_handover", "manipulate_pipette", "water_pouring", "click_bell", "table_rearrangement"):
        if task in text:
            return task
    return path.name


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def data_path(root: Path, info: dict, episode: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=episode // int(info["chunks_size"]), episode_index=episode
    )


def video_path(root: Path, info: dict, camera: str, episode: int) -> Path:
    return root / info["video_path"].format(
        episode_chunk=episode // int(info["chunks_size"]), episode_index=episode,
        video_key=f"observation.images.{camera}",
    )


def angle(axis: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.einsum("...i,i->...", axis, expected), -1.0, 1.0)))


def sample_loading_score(cube: np.ndarray, rack: np.ndarray):
    rr = rack[:3, :3]
    rel_center = rr.T @ (cube[:3, 3] - rack[:3, 3])
    rel_axis = rr.T @ cube[:3, 2]
    top, bottom, tube_half = 0.03964, -0.03964, 0.09751
    top_xy = rel_center[:2] + (top - rel_center[2]) / max(float(rel_axis[2]), 1e-6) * rel_axis[:2]
    holes = np.array([(x, y) for x in 0.0123 + 0.02218 * np.arange(-6, 6) for y in -0.0016 + 0.0209 * np.arange(-2, 3)])
    distance = float(np.linalg.norm(holes - top_xy, axis=1).min())
    tube_angle = float(angle(rel_axis, np.array([0.0, 0.0, 1.0])))
    rack_angle = float(angle(rr[:, 2], np.array([0.0, 0.0, 1.0])))
    tube_bottom = float(rel_center[2] - tube_half * rel_axis[2])
    depth = top - tube_bottom
    ok = distance <= .009 and depth >= .003 and tube_bottom >= bottom - .012 and tube_angle <= 10.5 and rack_angle <= 10.
    score = max(distance/.009, max(0., .003-depth)/.003, tube_angle/10.5, rack_angle/10.)
    return ok, score, f"hole={distance*1000:.1f}mm depth={depth*1000:.1f}mm tube={tube_angle:.1f}deg rack={rack_angle:.1f}deg"


def handover_score(pen: np.ndarray, holder: np.ndarray):
    distance = float(np.linalg.norm(pen[:2, 3] - holder[:2, 3]))
    pen_angle = float(angle(pen[:3, 0], np.array([0., 0., 1.])))
    holder_angle = float(angle(holder[:3, 1], np.array([0., 0., 1.])))
    ok = distance <= .03 and pen_angle < 75. and holder_angle < 75.
    score = max(distance/.03, pen_angle/75., holder_angle/75.)
    return ok, score, f"pen-holder={distance*1000:.1f}mm pen={pen_angle:.1f}deg holder={holder_angle:.1f}deg"


def water_score(bottle: np.ndarray, cup: np.ndarray):
    bottle_axis = bottle[:, :3, 1]
    bottle_angle = angle(bottle_axis, np.array([0., 0., 1.]))
    cup_angle = angle(cup[:, :3, 2], np.array([0., 0., 1.]))
    mouth = bottle[:, :3, 3] + .236 * bottle_axis
    delta = cup[:, :3, 3] - mouth
    geometry = (
        (np.linalg.norm(delta[:, :2], axis=1) < .08)
        & (mouth[:, 2] > cup[:, 2, 3] + .04)
        & (mouth[:, 2] < cup[:, 2, 3] + .30)
        & (np.sum(bottle_axis[:, :2] * delta[:, :2], axis=1) > -.02)
    )
    pouring = (bottle_angle > 45.) & (bottle_angle < 120.) & geometry
    ok = bool(pouring.any() and bottle_angle[-1] < 45. and cup_angle.max() < 45.)
    score = max(0. if pouring.any() else 2., bottle_angle[-1]/45., cup_angle.max()/45.)
    return ok, score, f"pour_frames={int(pouring.sum())} final_bottle={bottle_angle[-1]:.1f}deg max_cup={cup_angle.max():.1f}deg"


def numeric_audit(dataset: Path, task: str, info: dict, episodes: list[dict]):
    output = []
    for record in episodes:
        ep = int(record["episode_index"])
        path = data_path(dataset, info, ep)
        try:
            needed = {"sample_loading": ["cube_pose", "rack_pose"], "items_handover": ["pen_pose", "holder_pose"], "water_pouring": ["bottle_pose", "cup_pose"]}.get(task, [])
            table = pq.read_table(path, columns=needed) if needed else None
            if task == "sample_loading":
                cube = np.asarray(table["cube_pose"].to_pylist(), dtype=float)[-1]
                rack = np.asarray(table["rack_pose"].to_pylist(), dtype=float)[-1]
                ok, score, detail = sample_loading_score(cube, rack)
                status = "numeric_fail" if not ok else "pass"
            elif task == "items_handover":
                pen = np.asarray(table["pen_pose"].to_pylist(), dtype=float)[-1]
                holder = np.asarray(table["holder_pose"].to_pylist(), dtype=float)[-1]
                ok, score, detail = handover_score(pen, holder)
                status = "numeric_fail" if not ok else "pass"
            elif task == "water_pouring":
                bottle = np.asarray(table["bottle_pose"].to_pylist(), dtype=float)
                cup = np.asarray(table["cup_pose"].to_pylist(), dtype=float)
                ok, score, detail = water_score(bottle, cup)
                status = "numeric_fail" if not ok else "pass"
            else:
                status, score, detail = "visual_only", 0., "No sufficient task-state feature recorded"
        except Exception as exc:
            status, score, detail = "structural_fail", 99., f"{type(exc).__name__}: {exc}"
        output.append({"dataset": str(dataset), "task": task, "episode": ep, "status": status, "numeric_score": score, "detail": detail})
    return output


def decode_thumb(path: Path, width: int = 160):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-sseof", "-0.12", "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")
    except Exception:
        return None


def frame_metric(image: Image.Image | None):
    if image is None:
        return 99., "decode_failed"
    values = np.asarray(image, dtype=np.float32)
    return float(values.std()), f"mean={values.mean():.1f} std={values.std():.1f}"


def make_catalog(task: str, rows: list[dict], frames: dict, out: Path):
    page_size, cell_w, cell_h = 24, 160, 132
    for page_index in range((len(rows) + page_size - 1) // page_size):
        page_rows = rows[page_index * page_size:(page_index + 1) * page_size]
        image = Image.new("RGB", (cell_w * 6, cell_h * 4 + 24), "white")
        draw = ImageDraw.Draw(image)
        draw.text((4, 4), f"{task}: terminal high-camera frames, page {page_index + 1}", fill="black")
        for index, row in enumerate(page_rows):
            x, y = (index % 6) * cell_w, 24 + (index // 6) * cell_h
            frame = frames[(row["dataset"], row["episode"], "cam_high")]
            if frame is None:
                frame = Image.new("RGB", (cell_w, 120), "black")
            image.paste(frame, (x, y + 14))
            label = f"{Path(row['dataset']).name[-14:]} e{row['episode']}"
            draw.text((x + 2, y + 1), label, fill="red" if row["status"].endswith("fail") else "black")
        image.save(out / f"{task}_high_page_{page_index + 1:02d}.jpg", quality=90)


def make_triplet(row: dict, frames: dict, out: Path):
    image = Image.new("RGB", (1440, 430), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 5), f"{row['task']} | {Path(row['dataset']).relative_to(Path(row['root']))} | episode {row['episode']} | {row['status']}", fill="red" if row["status"].endswith("fail") else "black")
    draw.text((8, 22), row["detail"] + " | " + row.get("visual_detail", ""), fill="black")
    for idx, camera in enumerate(CAMERAS):
        frame = frames[(row["dataset"], row["episode"], camera)]
        if frame is None:
            frame = Image.new("RGB", (480, 360), "black")
        else:
            frame = frame.resize((480, 360))
        image.paste(frame, (480 * idx, 54))
        draw.text((480 * idx + 6, 38), camera, fill="black")
    safe = str(Path(row["dataset"]).relative_to(Path(row["root"]))).replace("/", "_")
    image.save(out / f"{safe}_ep_{row['episode']:06d}.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    catalog_dir = args.out / "high_camera_catalog"
    triplet_dir = args.out / "three_view_candidates"
    catalog_dir.mkdir(); triplet_dir.mkdir()

    all_rows = []
    for info_path in sorted(args.root.glob("**/meta/info.json")):
        dataset = info_path.parent.parent
        info = json.loads(info_path.read_text())
        if info.get("codebase_version") != "v2.1":
            continue
        episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
        all_rows.extend(numeric_audit(dataset, task_for(dataset), info, episodes))

    frame_jobs = {}
    for row in all_rows:
        info = json.loads((Path(row["dataset"]) / "meta" / "info.json").read_text())
        for camera in CAMERAS:
            frame_jobs[(row["dataset"], row["episode"], camera)] = video_path(Path(row["dataset"]), info, camera, row["episode"])
    frames = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(decode_thumb, path): key for key, path in frame_jobs.items()}
        for future in as_completed(futures):
            frames[futures[future]] = future.result()

    for row in all_rows:
        stds = [frame_metric(frames[(row["dataset"], row["episode"], camera)])[0] for camera in CAMERAS]
        row["visual_score"] = min(stds)
        row["visual_detail"] = "; ".join(f"{cam}:{frame_metric(frames[(row['dataset'], row['episode'], cam)])[1]}" for cam in CAMERAS)
        row["root"] = str(args.root)

    by_task = defaultdict(list)
    for row in all_rows:
        by_task[row["task"]].append(row)
    for task, rows in by_task.items():
        rows.sort(key=lambda item: (item["dataset"], item["episode"]))
        make_catalog(task, rows, frames, catalog_dir)

    candidates = []
    for task, rows in by_task.items():
        numeric_fails = [row for row in rows if row["status"] in {"numeric_fail", "structural_fail"}]
        borderline = sorted([row for row in rows if row not in numeric_fails], key=lambda item: (-item["numeric_score"], item["visual_score"]))[:4]
        visual = sorted(rows, key=lambda item: item["visual_score"])[:4]
        seen = set()
        for row in numeric_fails + borderline + visual:
            key = (row["dataset"], row["episode"])
            if key not in seen:
                candidates.append(row); seen.add(key)
    for row in candidates:
        make_triplet(row, frames, triplet_dir)

    fields = ["task", "dataset", "episode", "status", "numeric_score", "visual_score", "detail", "visual_detail"]
    with (args.out / "all_episode_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in all_rows: writer.writerow({field: row[field] for field in fields})
    with (args.out / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in candidates: writer.writerow({field: row[field] for field in fields})
    summary = defaultdict(int)
    for row in all_rows: summary[(row["task"], row["status"])] += 1
    with (args.out / "summary.txt").open("w") as handle:
        handle.write(f"episodes={len(all_rows)}\n")
        for (task, status), count in sorted(summary.items()): handle.write(f"{task}\t{status}\t{count}\n")
        handle.write(f"candidate_triplets={len(candidates)}\n")


if __name__ == "__main__":
    main()
