#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay one LeRobot episode on a robot with absolute and speed safety limits."""

import csv
import json
import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_piper_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    piper_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

PIPER_SAFE_REPLAY_ABSOLUTE_LIMITS: dict[str, tuple[float, float]] = {
    "left_joint_1.pos": (-145.0, 145.0),
    "left_joint_3.pos": (-170.0, 5.0),
    "left_joint_4.pos": (-85.0, 85.0),
    "left_joint_5.pos": (-85.0, 85.0),
    "left_joint_6.pos": (-165.0, 165.0),
    "left_gripper.pos": (0.0, 90.0),
    "right_joint_1.pos": (-145.0, 145.0),
    "right_joint_3.pos": (-170.0, 5.0),
    "right_joint_4.pos": (-85.0, 85.0),
    "right_joint_5.pos": (-85.0, 85.0),
    "right_joint_6.pos": (-165.0, 165.0),
    "right_gripper.pos": (0.0, 90.0),
}

PIPER_SAFE_REPLAY_SPEED_LIMITS: dict[str, tuple[float, float]] = {
    # Replay is used for offline trajectory verification, so keep absolute
    # joint limits but allow faster catch-up than deployment safety limits.
    "left_joint_1.pos": (-60.0, 90.0),
    "left_joint_2.pos": (-90.0, 150.0),
    "left_joint_3.pos": (-150.0, 120.0),
    "left_joint_4.pos": (-180.0, 150.0),
    "left_joint_5.pos": (-60.0, 90.0),
    "left_joint_6.pos": (-120.0, 120.0),
    "left_gripper.pos": (-300.0, 300.0),
    "right_joint_1.pos": (-60.0, 90.0),
    "right_joint_2.pos": (-90.0, 180.0),
    "right_joint_3.pos": (-90.0, 90.0),
    "right_joint_4.pos": (-240.0, 180.0),
    "right_joint_5.pos": (-120.0, 90.0),
    "right_joint_6.pos": (-120.0, 180.0),
    "right_gripper.pos": (-300.0, 300.0),
}


@dataclass
class SafeReplayDatasetConfig:
    repo_id: str
    episode: int
    root: str | Path | None = None
    fps: int = 30
    start_frame_index: int = 0


@dataclass
class SafeReplayConfig:
    robot: RobotConfig
    dataset: SafeReplayDatasetConfig
    source: str = OBS_STATE
    hold_current_s: float = 1.0
    max_frames: int | None = None
    interpolation_steps: int = 1
    wait_until_target: bool = False
    joint_tolerance_deg: float = 2.0
    gripper_tolerance: float = 3.0
    max_wait_per_target_s: float = 2.0
    align_first_frame: bool = True
    align_first_frame_timeout_s: float = 12.0
    align_first_frame_joint_tolerance_deg: float = 5.0
    align_first_frame_gripper_tolerance: float = 10.0
    save_feedback: bool = True
    feedback_output_path: str | None = None
    feedback_summary_path: str | None = None
    stop_on_pose: bool = False
    stop_pose_path: str = (
        "/home/phl/workspace/Evo-RL/configs/poses/"
        "phone_slot_ep55_final_right_arm_slot_above_pose.json"
    )
    stop_pose_tolerance_deg: float = 3.0
    stop_pose_gripper_tolerance: float = 5.0
    stop_pose_stable_frames: int = 30
    stop_pose_min_runtime_s: float = 5.0
    stop_pose_dry_run: bool = True
    stop_pose_tail_frames: int = 100
    stop_pose_tail_output_path: str | None = None
    log_interval_s: float = 1.0
    play_sounds: bool = False


def _require_finite_float(key: str, value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"Unsafe replay value for {key}: {value}")
    return value


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_feedback_path(path: str | None, cfg: SafeReplayConfig, suffix: str) -> Path:
    if path:
        output = Path(path).expanduser()
    else:
        safe_source = cfg.source.replace(".", "_")
        output = (
            Path("outputs/replay_feedback")
            / f"safe_replay_ep{cfg.dataset.episode}_{safe_source}_{_now_tag()}{suffix}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _resolve_stop_pose_tail_path(path: str | None) -> Path:
    if path:
        output = Path(path).expanduser()
    else:
        output = Path("outputs/replay_feedback") / f"stop_pose_tail_{_now_tag()}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _load_pose_file(path: str) -> dict[str, float]:
    with open(Path(path).expanduser()) as f:
        payload = json.load(f)
    joint_pos = payload["joint_pos"] if isinstance(payload, dict) and "joint_pos" in payload else payload
    return {str(key): float(value) for key, value in joint_pos.items() if str(key).endswith(".pos")}


def _write_stop_pose_tail(rows: list[dict[str, float | int | bool]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_stop_pose_tail(
    *,
    raw_observation: dict[str, Any],
    target: dict[str, float],
    buffer: deque,
    elapsed_s: float,
    stable_count: int,
    stable_required: int,
    joint_tolerance_deg: float,
    gripper_tolerance: float,
    min_runtime_reached: bool,
    dry_run: bool,
) -> tuple[int, float, bool]:
    max_error = 0.0
    all_within_tolerance = True
    target_errors: dict[str, float] = {}

    for key, target_value in target.items():
        if key not in raw_observation:
            all_within_tolerance = False
            target_errors[key] = math.nan
            continue
        current = _require_finite_float(key, raw_observation[key])
        tolerance = gripper_tolerance if "gripper" in key else joint_tolerance_deg
        error = target_value - current
        target_errors[key] = error
        max_error = max(max_error, abs(error))
        if abs(error) > tolerance:
            all_within_tolerance = False

    if not min_runtime_reached:
        all_within_tolerance = False

    stable_count = stable_count + 1 if all_within_tolerance else 0
    would_stop = all_within_tolerance and stable_count >= stable_required
    row: dict[str, float | int | bool] = {
        "elapsed_s": elapsed_s,
        "stable_count": stable_count,
        "stable_required": stable_required,
        "within_tolerance": all_within_tolerance,
        "would_stop": would_stop,
        "dry_run": dry_run,
        "max_error": max_error,
        "joint_tolerance": joint_tolerance_deg,
        "gripper_tolerance": gripper_tolerance,
        "min_runtime_reached": min_runtime_reached,
    }
    for key, value in raw_observation.items():
        if key.startswith("right_") and key.endswith(".pos"):
            row[f"{key}.current"] = _require_finite_float(key, value)
    for key, target_value in target.items():
        row[f"{key}.target"] = target_value
        row[f"{key}.error"] = target_errors.get(key, math.nan)
    buffer.append(row)
    return stable_count, max_error, would_stop


def _current_action_from_observation(observation: dict[str, Any], action_keys: list[str]) -> dict[str, float]:
    missing = [key for key in action_keys if key not in observation]
    if missing:
        raise RuntimeError(f"Robot observation is missing action key(s): {missing}")
    return {key: _require_finite_float(key, observation[key]) for key in action_keys}


def _target_from_frame(
    frame: dict[str, Any],
    source: str,
    source_names: list[str],
    action_keys: list[str],
    current_action: dict[str, float],
) -> dict[str, float]:
    source_values = frame[source]
    source_by_name = {
        name: _require_finite_float(name, source_values[i]) for i, name in enumerate(source_names)
    }
    return {key: source_by_name.get(key, current_action[key]) for key in action_keys}


def _absolute_bounded_target(target: dict[str, float]) -> tuple[dict[str, float], int]:
    bounded_target: dict[str, float] = {}
    num_abs_clipped = 0
    for key, value in target.items():
        bounded = _require_finite_float(key, value)
        if key in PIPER_SAFE_REPLAY_ABSOLUTE_LIMITS:
            low, high = PIPER_SAFE_REPLAY_ABSOLUTE_LIMITS[key]
            next_bounded = min(high, max(low, bounded))
            if next_bounded != bounded:
                num_abs_clipped += 1
            bounded = next_bounded
        bounded_target[key] = bounded
    return bounded_target, num_abs_clipped


def _target_reached(
    target: dict[str, float],
    current: dict[str, float],
    joint_tolerance_deg: float,
    gripper_tolerance: float,
) -> tuple[bool, float]:
    max_error = 0.0
    reached = True
    for key, target_value in target.items():
        if key not in current:
            return False, math.inf
        tolerance = gripper_tolerance if "gripper" in key else joint_tolerance_deg
        error = abs(_require_finite_float(key, target_value) - _require_finite_float(key, current[key]))
        max_error = max(max_error, error)
        if error > tolerance:
            reached = False
    return reached, max_error


def _interpolate_target_sequence(
    start: dict[str, float],
    target: dict[str, float],
    interpolation_steps: int,
) -> list[dict[str, float]]:
    if interpolation_steps <= 0:
        raise ValueError(f"interpolation_steps must be positive, got {interpolation_steps}")

    sequence: list[dict[str, float]] = []
    for step in range(1, interpolation_steps + 1):
        alpha = step / interpolation_steps
        sequence.append(
            {
                key: _require_finite_float(key, start[key])
                + (_require_finite_float(key, target_value) - _require_finite_float(key, start[key])) * alpha
                for key, target_value in target.items()
            }
        )
    return sequence


def _make_feedback_record(
    frame_idx: int,
    frame_count: int,
    sub_idx: int,
    interpolation_steps: int,
    iteration: int,
    elapsed_s: float,
    source: str,
    raw_target: dict[str, float],
    bounded_target: dict[str, float],
    sent_target: dict[str, float],
    feedback_before: dict[str, float],
    feedback_after: dict[str, float],
    reached: bool,
    max_error: float,
    safety_info: dict[str, float | int],
    action_keys: list[str],
) -> dict[str, float | int | str | bool]:
    row: dict[str, float | int | str | bool] = {
        "frame_index": frame_idx,
        "frame_count": frame_count,
        "substep_index": sub_idx,
        "interpolation_steps": interpolation_steps,
        "iteration": iteration,
        "elapsed_s": elapsed_s,
        "source": source,
        "reached": reached,
        "max_error": max_error,
        **safety_info,
    }
    for key in action_keys:
        row[f"{key}.replay_target"] = raw_target[key]
        row[f"{key}.bounded_target"] = bounded_target[key]
        row[f"{key}.sent_target"] = sent_target[key]
        row[f"{key}.feedback_before"] = feedback_before[key]
        row[f"{key}.feedback_after"] = feedback_after[key]
        row[f"{key}.error_before"] = bounded_target[key] - feedback_before[key]
        row[f"{key}.error_after"] = bounded_target[key] - feedback_after[key]
        row[f"{key}.sent_minus_feedback_before"] = sent_target[key] - feedback_before[key]
        row[f"{key}.sent_minus_feedback_after"] = sent_target[key] - feedback_after[key]
    return row


def _write_feedback_outputs(
    records: list[dict[str, float | int | str | bool]],
    action_keys: list[str],
    csv_path: Path,
    summary_path: Path,
) -> None:
    if not records:
        return

    fieldnames = list(records[0])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "Safe replay feedback comparison",
        f"records: {len(records)}",
        f"csv: {csv_path}",
        "",
        "Per-joint error compares bounded replay target vs motor feedback.",
        "Positive error means target is larger than feedback.",
        "",
        (
            "joint, mean_abs_error_before, max_abs_error_before, "
            "mean_abs_error_after, max_abs_error_after, "
            "mean_abs_sent_delta_before, max_abs_sent_delta_before"
        ),
    ]
    for key in action_keys:
        err_before = [abs(float(row[f"{key}.error_before"])) for row in records]
        err_after = [abs(float(row[f"{key}.error_after"])) for row in records]
        sent_delta_before = [abs(float(row[f"{key}.sent_minus_feedback_before"])) for row in records]
        lines.append(
            f"{key}, "
            f"{sum(err_before) / len(err_before):.3f}, {max(err_before):.3f}, "
            f"{sum(err_after) / len(err_after):.3f}, {max(err_after):.3f}, "
            f"{sum(sent_delta_before) / len(sent_delta_before):.3f}, {max(sent_delta_before):.3f}"
        )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _clip_safe_replay_target(
    target: dict[str, float],
    current: dict[str, float],
    fps: int,
) -> tuple[dict[str, float], dict[str, float | int]]:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    clipped: dict[str, float] = {}
    num_abs_clipped = 0
    num_speed_clipped = 0
    max_abs_delta_before_clip = 0.0
    max_abs_delta_after_clip = 0.0

    bounded_target, num_abs_clipped = _absolute_bounded_target(target)

    for key, bounded in bounded_target.items():
        if key in PIPER_SAFE_REPLAY_SPEED_LIMITS:
            current_value = _require_finite_float(key, current[key])
            min_speed, max_speed = PIPER_SAFE_REPLAY_SPEED_LIMITS[key]
            min_delta = min_speed / fps
            max_delta = max_speed / fps
            delta = bounded - current_value
            clipped_delta = min(max_delta, max(min_delta, delta))
            max_abs_delta_before_clip = max(max_abs_delta_before_clip, abs(delta))
            max_abs_delta_after_clip = max(max_abs_delta_after_clip, abs(clipped_delta))
            if clipped_delta != delta:
                num_speed_clipped += 1
            bounded = current_value + clipped_delta

        clipped[key] = bounded

    return clipped, {
        "num_abs_clipped": num_abs_clipped,
        "num_speed_clipped": num_speed_clipped,
        "max_abs_delta_before_clip": max_abs_delta_before_clip,
        "max_abs_delta_after_clip": max_abs_delta_after_clip,
    }


@parser.wrap()
def safe_replay(cfg: SafeReplayConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    if cfg.source not in {ACTION, OBS_STATE}:
        raise ValueError(f"--source must be either {ACTION!r} or {OBS_STATE!r}, got {cfg.source!r}.")
    if cfg.source not in dataset.features:
        raise ValueError(f"Replay source {cfg.source!r} is not in dataset features.")
    if cfg.joint_tolerance_deg < 0:
        raise ValueError(f"--joint_tolerance_deg must be non-negative, got {cfg.joint_tolerance_deg}.")
    if cfg.gripper_tolerance < 0:
        raise ValueError(f"--gripper_tolerance must be non-negative, got {cfg.gripper_tolerance}.")
    if cfg.max_wait_per_target_s < 0:
        raise ValueError(f"--max_wait_per_target_s must be non-negative, got {cfg.max_wait_per_target_s}.")
    if cfg.interpolation_steps <= 0:
        raise ValueError(f"--interpolation_steps must be positive, got {cfg.interpolation_steps}.")
    if cfg.dataset.start_frame_index < 0:
        raise ValueError(
            f"--dataset.start_frame_index must be non-negative, got {cfg.dataset.start_frame_index}."
        )
    if cfg.align_first_frame_timeout_s < 0:
        raise ValueError(
            f"--align_first_frame_timeout_s must be non-negative, got {cfg.align_first_frame_timeout_s}."
        )
    if cfg.align_first_frame_joint_tolerance_deg < 0:
        raise ValueError(
            "--align_first_frame_joint_tolerance_deg must be non-negative, "
            f"got {cfg.align_first_frame_joint_tolerance_deg}."
        )
    if cfg.align_first_frame_gripper_tolerance < 0:
        raise ValueError(
            "--align_first_frame_gripper_tolerance must be non-negative, "
            f"got {cfg.align_first_frame_gripper_tolerance}."
        )
    if cfg.stop_pose_tolerance_deg < 0:
        raise ValueError(f"--stop_pose_tolerance_deg must be non-negative, got {cfg.stop_pose_tolerance_deg}.")
    if cfg.stop_pose_gripper_tolerance < 0:
        raise ValueError(
            f"--stop_pose_gripper_tolerance must be non-negative, got {cfg.stop_pose_gripper_tolerance}."
        )
    if cfg.stop_pose_stable_frames <= 0:
        raise ValueError(f"--stop_pose_stable_frames must be positive, got {cfg.stop_pose_stable_frames}.")
    if cfg.stop_pose_min_runtime_s < 0:
        raise ValueError(f"--stop_pose_min_runtime_s must be non-negative, got {cfg.stop_pose_min_runtime_s}.")
    if cfg.stop_pose_tail_frames <= 0:
        raise ValueError(f"--stop_pose_tail_frames must be positive, got {cfg.stop_pose_tail_frames}.")

    feedback_csv_path = _resolve_feedback_path(cfg.feedback_output_path, cfg, ".csv")
    feedback_summary_path = _resolve_feedback_path(cfg.feedback_summary_path, cfg, ".txt")
    stop_pose_tail_path = _resolve_stop_pose_tail_path(cfg.stop_pose_tail_output_path)
    stop_pose_target = _load_pose_file(cfg.stop_pose_path) if cfg.stop_on_pose else {}
    stop_pose_tail_buffer = deque(maxlen=cfg.stop_pose_tail_frames)
    stop_pose_stable_count = 0
    feedback_records: list[dict[str, float | int | str | bool]] = []
    source_names = list(dataset.features[cfg.source]["names"])
    episode_frames = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == cfg.dataset.episode
        and x["frame_index"] >= cfg.dataset.start_frame_index
    )
    action_keys = list(robot.action_features)
    replay_fps = cfg.dataset.fps or dataset.fps
    frame_count = len(episode_frames)
    if cfg.max_frames is not None:
        frame_count = min(frame_count, cfg.max_frames)
    if frame_count == 0:
        raise RuntimeError(
            f"No frames found for episode {cfg.dataset.episode} starting at frame "
            f"{cfg.dataset.start_frame_index}."
        )

    robot.connect()

    try:
        log_say("Holding current robot state", cfg.play_sounds, blocking=True)
        hold_steps = max(0, round(cfg.hold_current_s * replay_fps))
        for _ in range(hold_steps):
            start_t = time.perf_counter()
            obs = robot.get_observation()
            current_action = _current_action_from_observation(obs, action_keys)
            processed_action = robot_action_processor((current_action, obs))
            robot.send_action(processed_action)
            precise_sleep(max(1 / replay_fps - (time.perf_counter() - start_t), 0.0))

        last_log_t = 0.0
        if cfg.align_first_frame and frame_count > 0:
            log_say("Aligning to first replay frame", cfg.play_sounds, blocking=True)
            obs = robot.get_observation()
            current_action = _current_action_from_observation(obs, action_keys)
            first_target = _target_from_frame(
                episode_frames[0],
                cfg.source,
                source_names,
                action_keys,
                current_action,
            )
            first_bounded_target, _ = _absolute_bounded_target(first_target)
            align_start_t = time.perf_counter()
            align_iterations = 0
            while True:
                start_t = time.perf_counter()
                safe_target, safety_info = _clip_safe_replay_target(first_target, current_action, replay_fps)
                processed_action = robot_action_processor((safe_target, obs))
                robot.send_action(processed_action)
                align_iterations += 1

                elapsed_s = time.perf_counter() - start_t
                precise_sleep(max(1 / replay_fps - elapsed_s, 0.0))

                obs = robot.get_observation()
                current_action = _current_action_from_observation(obs, action_keys)
                reached, max_error = _target_reached(
                    first_bounded_target,
                    current_action,
                    cfg.align_first_frame_joint_tolerance_deg,
                    cfg.align_first_frame_gripper_tolerance,
                )
                if (
                    safety_info["num_abs_clipped"]
                    or safety_info["num_speed_clipped"]
                    or time.perf_counter() - last_log_t >= cfg.log_interval_s
                ):
                    logging.info(
                        "Safe replay first-frame alignment iter=%d reached=%s max_error=%.2f "
                        "absolute_clipped=%s speed_clipped=%s max_delta %.2f -> %.2f",
                        align_iterations,
                        reached,
                        max_error,
                        safety_info["num_abs_clipped"],
                        safety_info["num_speed_clipped"],
                        safety_info["max_abs_delta_before_clip"],
                        safety_info["max_abs_delta_after_clip"],
                    )
                    last_log_t = time.perf_counter()
                if reached:
                    break
                if time.perf_counter() - align_start_t >= cfg.align_first_frame_timeout_s:
                    logging.warning(
                        "First-frame alignment timed out after %.2fs with max_error=%.2f; starting replay.",
                        cfg.align_first_frame_timeout_s,
                        max_error,
                    )
                    break

        log_say("Safe replaying episode", cfg.play_sounds, blocking=True)
        replay_start_t = time.perf_counter()
        last_log_t = 0.0
        stop_requested = False
        for idx in range(frame_count):
            if stop_requested:
                break
            obs = robot.get_observation()
            current_action = _current_action_from_observation(obs, action_keys)
            target = _target_from_frame(
                episode_frames[idx],
                cfg.source,
                source_names,
                action_keys,
                current_action,
            )
            previous_target = current_action if idx == 0 else previous_frame_target
            interpolated_targets = _interpolate_target_sequence(
                previous_target, target, cfg.interpolation_steps
            )
            previous_frame_target = target

            for sub_idx, sub_target in enumerate(interpolated_targets, start=1):
                bounded_target, _ = _absolute_bounded_target(sub_target)
                target_start_t = time.perf_counter()
                target_iterations = 0
                while True:
                    start_t = time.perf_counter()
                    feedback_before = dict(current_action)
                    safe_target, safety_info = _clip_safe_replay_target(sub_target, current_action, replay_fps)
                    processed_action = robot_action_processor((safe_target, obs))
                    robot.send_action(processed_action)
                    target_iterations += 1

                    elapsed_s = time.perf_counter() - start_t
                    precise_sleep(max(1 / replay_fps - elapsed_s, 0.0))

                    obs = robot.get_observation()
                    current_action = _current_action_from_observation(obs, action_keys)
                    feedback_after = dict(current_action)

                    now = time.perf_counter()
                    if cfg.stop_on_pose:
                        stop_pose_stable_count, stop_pose_error, stop_pose_would_stop = _update_stop_pose_tail(
                            raw_observation=feedback_after,
                            target=stop_pose_target,
                            buffer=stop_pose_tail_buffer,
                            elapsed_s=now - replay_start_t,
                            stable_count=stop_pose_stable_count,
                            stable_required=cfg.stop_pose_stable_frames,
                            joint_tolerance_deg=cfg.stop_pose_tolerance_deg,
                            gripper_tolerance=cfg.stop_pose_gripper_tolerance,
                            min_runtime_reached=(now - replay_start_t) >= cfg.stop_pose_min_runtime_s,
                            dry_run=cfg.stop_pose_dry_run,
                        )
                        if stop_pose_stable_count == 1 or stop_pose_stable_count % replay_fps == 0:
                            logging.info(
                                "Replay stop-on-pose check: stable=%d/%d max_error=%.2f "
                                "dry_run=%s.",
                                stop_pose_stable_count,
                                cfg.stop_pose_stable_frames,
                                stop_pose_error,
                                cfg.stop_pose_dry_run,
                            )
                        if stop_pose_would_stop:
                            if cfg.stop_pose_dry_run:
                                if (
                                    stop_pose_stable_count == cfg.stop_pose_stable_frames
                                    or stop_pose_stable_count % replay_fps == 0
                                ):
                                    logging.info(
                                        "Replay stop-on-pose would stop now at frame %d/%d "
                                        "with stable=%d max_error=%.2f. Dry-run keeps replay running.",
                                        idx + 1,
                                        frame_count,
                                        stop_pose_stable_count,
                                        stop_pose_error,
                                    )
                            else:
                                logging.info(
                                    "Replay stop-on-pose reached at frame %d/%d with stable=%d "
                                    "max_error=%.2f. Stopping replay.",
                                    idx + 1,
                                    frame_count,
                                    stop_pose_stable_count,
                                    stop_pose_error,
                                )
                                stop_requested = True

                    reached, max_error = _target_reached(
                        bounded_target,
                        current_action,
                        cfg.joint_tolerance_deg,
                        cfg.gripper_tolerance,
                    )
                    if (
                        safety_info["num_abs_clipped"]
                        or safety_info["num_speed_clipped"]
                        or now - last_log_t >= cfg.log_interval_s
                    ):
                        logging.info(
                            "Safe replay frame %d/%d substep=%d/%d iter=%d reached=%s max_error=%.2f "
                            "absolute_clipped=%s speed_clipped=%s max_delta %.2f -> %.2f",
                            idx + 1,
                            frame_count,
                            sub_idx,
                            cfg.interpolation_steps,
                            target_iterations,
                            reached,
                            max_error,
                            safety_info["num_abs_clipped"],
                            safety_info["num_speed_clipped"],
                            safety_info["max_abs_delta_before_clip"],
                            safety_info["max_abs_delta_after_clip"],
                        )
                        last_log_t = now

                    if cfg.save_feedback:
                        feedback_records.append(
                            _make_feedback_record(
                                frame_idx=idx + 1,
                                frame_count=frame_count,
                                sub_idx=sub_idx,
                                interpolation_steps=cfg.interpolation_steps,
                                iteration=target_iterations,
                                elapsed_s=now - replay_start_t,
                                source=cfg.source,
                                raw_target=sub_target,
                                bounded_target=bounded_target,
                                sent_target=safe_target,
                                feedback_before=feedback_before,
                                feedback_after=feedback_after,
                                reached=reached,
                                max_error=max_error,
                                safety_info=safety_info,
                                action_keys=action_keys,
                            )
                        )

                    if not cfg.wait_until_target:
                        break
                    if reached:
                        break
                    if stop_requested:
                        break
                    if time.perf_counter() - target_start_t >= cfg.max_wait_per_target_s:
                        logging.warning(
                            "Safe replay frame %d/%d substep=%d/%d timed out after %.2fs "
                            "with max_error=%.2f; moving to next target.",
                            idx + 1,
                            frame_count,
                            sub_idx,
                            cfg.interpolation_steps,
                            cfg.max_wait_per_target_s,
                            max_error,
                        )
                        break
                if stop_requested:
                    break
    finally:
        try:
            if cfg.save_feedback and feedback_records:
                _write_feedback_outputs(
                    feedback_records,
                    action_keys,
                    feedback_csv_path,
                    feedback_summary_path,
                )
                logging.info("Saved replay feedback CSV to %s", feedback_csv_path)
                logging.info("Saved replay feedback summary to %s", feedback_summary_path)
            if cfg.stop_on_pose and stop_pose_tail_buffer:
                _write_stop_pose_tail(list(stop_pose_tail_buffer), stop_pose_tail_path)
                logging.info("Saved replay stop-pose tail diagnostics to %s", stop_pose_tail_path)
        finally:
            robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    safe_replay()


if __name__ == "__main__":
    main()
