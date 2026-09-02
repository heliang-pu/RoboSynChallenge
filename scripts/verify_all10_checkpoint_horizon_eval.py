#!/usr/bin/env python3
"""Validate one all-task checkpoint/horizon rollout and publish an atomic result.

The evaluator writes a timestamped directory below ``--result-root``.  This
script accepts the newest *complete* run, verifies the episode/video contract,
and writes a small ``job_result.json`` consumed by the sweep report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_VIDEO_SIZE = "2560x480"
DEFAULT_PROTOCOL_REVISION = "all10_h64_v2_bounded_texture_pool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--complete-marker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--protocol-revision", default=DEFAULT_PROTOCOL_REVISION)
    return parser.parse_args()


def video_size(path: Path) -> str:
    return subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        text=True,
    ).strip()


def atomic_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    metrics_paths = sorted(
        args.result_root.rglob("evaluation_metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    errors: list[str] = []
    for metrics_path in metrics_paths:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary = metrics["summary"]
            episode_count = int(summary["episode_count"])
            success_count = int(summary["success_count"])
            if episode_count != args.expected_episodes:
                raise ValueError(
                    f"episode_count={episode_count}, expected={args.expected_episodes}"
                )
            if not 0 <= success_count <= episode_count:
                raise ValueError(
                    f"invalid success_count={success_count} for episode_count={episode_count}"
                )

            videos = sorted((metrics_path.parent / "videos").glob("*.mp4"))
            if len(videos) != args.expected_episodes:
                raise ValueError(
                    f"video_count={len(videos)}, expected={args.expected_episodes}"
                )
            sizes = {video_size(video) for video in videos}
            if sizes != {EXPECTED_VIDEO_SIZE}:
                raise ValueError(
                    f"video sizes={sorted(sizes)}, expected={EXPECTED_VIDEO_SIZE}"
                )

            config = metrics.get("config", {})
            recorded_seed = config.get("seed")
            if recorded_seed is not None and int(recorded_seed) != args.seed:
                raise ValueError(f"seed={recorded_seed}, expected={args.seed}")

            payload = {
                "schema_version": 1,
                "protocol_revision": args.protocol_revision,
                "checkpoint": args.checkpoint,
                "execution_horizon": args.horizon,
                "task": args.task,
                "seed": args.seed,
                "camera_keys": [
                    "cam_left_wrist",
                    "cam_right_wrist",
                    "cam_high",
                    "cam_third",
                ],
                "video_layout": {
                    "kind": "horizontal_four_view_composite",
                    "frame_size": EXPECTED_VIDEO_SIZE,
                    "per_view_size": "640x480",
                },
                "episode_count": episode_count,
                "success_count": success_count,
                "success_rate": success_count / episode_count,
                "video_count": len(videos),
                "average_action_steps": summary.get("average_action_steps"),
                "average_action_steps_ratio": summary.get(
                    "average_action_steps_ratio"
                ),
                "inference_call_count": summary.get("inference_call_count"),
                "average_inference_time_seconds": summary.get(
                    "average_inference_time_seconds"
                ),
                "average_inference_time_per_episode_seconds": summary.get(
                    "average_inference_time_per_episode_seconds"
                ),
                "metrics_path": str(metrics_path.resolve()),
                "video_dir": str((metrics_path.parent / "videos").resolve()),
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json_dump(args.output, payload)
            args.complete_marker.parent.mkdir(parents=True, exist_ok=True)
            args.complete_marker.touch()
            print(
                f"validated checkpoint={args.checkpoint} h={args.horizon} "
                f"task={args.task} success={success_count}/{episode_count} "
                f"videos={len(videos)}"
            )
            return
        except Exception as exc:  # Try an older timestamped run, if present.
            errors.append(f"{metrics_path}: {exc}")

    detail = "\n".join(errors[-5:]) if errors else "no evaluation_metrics.json found"
    raise RuntimeError(f"No complete run below {args.result_root}:\n{detail}")


if __name__ == "__main__":
    main()
