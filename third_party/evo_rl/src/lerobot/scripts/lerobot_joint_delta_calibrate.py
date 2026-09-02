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

"""
Teleoperate a follower arm and estimate per-cycle joint delta limits.

This script is meant for real-robot safety tuning. It runs one control loop that:
  1. reads follower observations,
  2. reads leader actions,
  3. sends leader actions to the follower,
  4. saves follower joint trajectories,
then writes position/delta/speed statistics to a txt file.

Run one joint at a time: keep all other joints still, move the selected joint at
the fastest speed you would still consider safe and smooth, then use the
reported q95/q99 deltas to choose a policy action limiter.
"""

import logging
import math
import time
import csv
import html
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from statistics import mean, median
from typing import Any

from lerobot.configs import parser
from lerobot.processor import RobotAction, RobotObservation, make_default_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_piper_follower,
    make_robot_from_config,
    piper_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_piper_leader,
    make_teleoperator_from_config,
    piper_leader,
)
from lerobot.utils.control_utils import sanity_check_bimanual_piper_pair
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up

logger = logging.getLogger(__name__)


@dataclass
class JointDeltaCalibrateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig

    # Match the deployment/control frequency. For the current phone-slot setup this should be 30.
    fps: int = 30
    # Data collection duration after warmup. Press Ctrl+C to stop early and still save partial results.
    duration_s: float = 20.0
    # Warmup/control-check time before samples are recorded.
    warmup_s: float = 3.0
    # Comma-separated keys to highlight in the txt, e.g. "left_joint_1.pos,right_joint_4.pos".
    focus_keys: str | None = None
    # Optional label written into the report, e.g. "left_joint_1_fast".
    label: str | None = None
    # Output txt path. If omitted, a timestamped txt is created under outputs/joint_delta_limits.
    output_path: str | None = None
    # Save every recorded follower state as a CSV sidecar for later plotting/inspection.
    save_raw_csv: bool = True
    raw_output_path: str | None = None
    # Save an HTML sidecar with per-joint angle curves.
    save_plot_html: bool = True
    plot_output_path: str | None = None
    plot_max_points: int = 1200
    # Wait for ENTER before the follower starts moving.
    require_enter_to_start: bool = True
    # Print live progress every N seconds.
    print_every_s: float = 1.0

    # Recommendation formula: recommended_delta ~= q95(abs(delta)) * recommendation_margin.
    recommendation_margin: float = 1.2
    min_joint_delta: float = 0.5
    max_joint_delta: float = 6.0
    min_gripper_delta: float = 1.0
    max_gripper_delta: float = 8.0
    # Below this motion range, the joint is treated as "not moved" for recommendations.
    moved_range_threshold: float = 0.3


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_output_path(path: str | None, label: str | None) -> Path:
    if path:
        output = Path(path).expanduser()
    else:
        suffix = f"_{label}" if label else ""
        output = Path("outputs/joint_delta_limits") / f"joint_delta_limits{suffix}_{_now_tag()}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _resolve_sidecar_path(path: str | None, output_path: Path, suffix: str) -> Path:
    if path:
        sidecar = Path(path).expanduser()
    else:
        sidecar = output_path.with_suffix(suffix)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    return sidecar


def _parse_focus_keys(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _looks_like_gripper(key: str) -> bool:
    return "gripper" in key


def _recommend_delta_limit(key: str, stats: dict[str, float], cfg: JointDeltaCalibrateConfig) -> float:
    if stats["range"] < cfg.moved_range_threshold and stats["delta_q95"] <= 0:
        return 0.0

    floor = cfg.min_gripper_delta if _looks_like_gripper(key) else cfg.min_joint_delta
    cap = cfg.max_gripper_delta if _looks_like_gripper(key) else cfg.max_joint_delta

    base = max(stats["delta_q95"] * cfg.recommendation_margin, stats["delta_q90"] * 1.3)
    if base <= 0 and stats["delta_max"] > 0:
        base = stats["delta_max"] * 0.5
    return _clamp(base, floor, cap)


def _collect_numeric_sample(obs: RobotObservation, keys: list[str] | None = None) -> dict[str, float]:
    sample: dict[str, float] = {}
    source_keys = keys if keys is not None else list(obs)
    for key in source_keys:
        if key not in obs:
            continue
        value = _as_float(obs[key])
        if value is not None:
            sample[key] = value
    return sample


def _make_key_order(robot: Robot, first_sample: dict[str, float]) -> list[str]:
    action_keys = [key for key in robot.action_features if key in first_sample]
    extra_keys = sorted(key for key in first_sample if key not in action_keys)
    return action_keys + extra_keys


def _compute_stats(
    samples: list[dict[str, float]], timestamps: list[float], keys: list[str], cfg: JointDeltaCalibrateConfig
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        positions = [sample[key] for sample in samples if key in sample]
        if not positions:
            continue

        deltas: list[float] = []
        speeds: list[float] = []
        previous_pos: float | None = None
        previous_t: float | None = None
        for sample, timestamp in zip(samples, timestamps, strict=True):
            if key not in sample:
                continue
            current = sample[key]
            if previous_pos is not None and previous_t is not None:
                delta = abs(current - previous_pos)
                dt = timestamp - previous_t
                deltas.append(delta)
                if dt > 0:
                    speeds.append(delta / dt)
            previous_pos = current
            previous_t = timestamp

        stats = {
            "samples": float(len(positions)),
            "min": min(positions),
            "max": max(positions),
            "range": max(positions) - min(positions),
            "mean": _safe_mean(positions),
            "median": _safe_median(positions),
            "delta_mean": _safe_mean(deltas),
            "delta_median": _safe_median(deltas),
            "delta_q90": _percentile(deltas, 90),
            "delta_q95": _percentile(deltas, 95),
            "delta_q99": _percentile(deltas, 99),
            "delta_max": max(deltas) if deltas else 0.0,
            "speed_min_per_s": min(speeds) if speeds else 0.0,
            "speed_mean_per_s": _safe_mean(speeds),
            "speed_median_per_s": _safe_median(speeds),
            "speed_q90_per_s": _percentile(speeds, 90),
            "speed_q95_per_s": _percentile(speeds, 95),
            "speed_q99_per_s": _percentile(speeds, 99),
            "speed_max_per_s": max(speeds) if speeds else 0.0,
        }
        stats["recommended_delta_limit"] = _recommend_delta_limit(key, stats, cfg)
        result[key] = stats
    return result


def _diffs(values: list[float]) -> list[float]:
    return [values[idx] - values[idx - 1] for idx in range(1, len(values)) if values[idx] > values[idx - 1]]


def _interval_summary(intervals_s: list[float]) -> dict[str, float]:
    if not intervals_s:
        return {
            "count": 0.0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
            "q90_s": 0.0,
            "q95_s": 0.0,
            "q99_s": 0.0,
            "mean_hz": 0.0,
        }
    mean_s = _safe_mean(intervals_s)
    return {
        "count": float(len(intervals_s)),
        "mean_s": mean_s,
        "median_s": _safe_median(intervals_s),
        "min_s": min(intervals_s),
        "max_s": max(intervals_s),
        "q90_s": _percentile(intervals_s, 90),
        "q95_s": _percentile(intervals_s, 95),
        "q99_s": _percentile(intervals_s, 99),
        "mean_hz": 1.0 / mean_s if mean_s > 0 else 0.0,
    }


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _format_interval_summary(name: str, summary: dict[str, float]) -> list[str]:
    return [
        f"{name}_interval_count: {int(summary['count'])}",
        f"{name}_mean_interval_s: {_format_float(summary['mean_s'])}",
        f"{name}_median_interval_s: {_format_float(summary['median_s'])}",
        f"{name}_min_interval_s: {_format_float(summary['min_s'])}",
        f"{name}_max_interval_s: {_format_float(summary['max_s'])}",
        f"{name}_q90_interval_s: {_format_float(summary['q90_s'])}",
        f"{name}_q95_interval_s: {_format_float(summary['q95_s'])}",
        f"{name}_q99_interval_s: {_format_float(summary['q99_s'])}",
        f"{name}_mean_hz: {_format_float(summary['mean_hz'])}",
    ]


def _write_report(
    output_path: Path,
    cfg: JointDeltaCalibrateConfig,
    stats: dict[str, dict[str, float]],
    focus_keys: list[str],
    sample_count: int,
    elapsed_s: float,
    actual_fps: float,
    sample_interval_summary: dict[str, float],
    control_interval_summary: dict[str, float],
) -> None:
    header = [
        "Joint delta limit calibration report",
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"label: {cfg.label or ''}",
        f"target_fps: {cfg.fps}",
        f"actual_fps: {_format_float(actual_fps)}",
        f"recorded_samples: {sample_count}",
        f"recorded_elapsed_s: {_format_float(elapsed_s)}",
        f"recommendation: q95(abs adjacent delta) * {cfg.recommendation_margin}",
        "",
        "Timing",
        "-" * 120,
        "sample_interval is the real data-save interval during recording.",
        "control_interval is the real teleop control-loop interval including warmup and recording.",
        *_format_interval_summary("sample", sample_interval_summary),
        *_format_interval_summary("control", control_interval_summary),
        "",
    ]

    lines = header
    if focus_keys:
        lines.extend(["Focused joints", "-" * 120])
        for key in focus_keys:
            if key not in stats:
                lines.append(f"{key}: not found in observations")
                continue
            item = stats[key]
            lines.append(
                " | ".join(
                    [
                        key,
                        f"range={_format_float(item['range'])}",
                        f"delta_q95={_format_float(item['delta_q95'])}",
                        f"delta_q99={_format_float(item['delta_q99'])}",
                        f"delta_max={_format_float(item['delta_max'])}",
                        f"speed_q95/s={_format_float(item['speed_q95_per_s'])}",
                        f"recommended_delta_limit={_format_float(item['recommended_delta_limit'])}",
                    ]
                )
            )
        lines.append("")

    columns = [
        "key",
        "samples",
        "min",
        "max",
        "range",
        "mean",
        "median",
        "delta_mean",
        "delta_median",
        "delta_q90",
        "delta_q95",
        "delta_q99",
        "delta_max",
        "speed_min_per_s",
        "speed_mean_per_s",
        "speed_median_per_s",
        "speed_q90_per_s",
        "speed_q95_per_s",
        "speed_q99_per_s",
        "speed_max_per_s",
        "recommended_delta_limit",
    ]
    lines.extend(["All joint statistics", "-" * 120, ",".join(columns)])
    for key, item in stats.items():
        row = [key]
        for column in columns[1:]:
            value = item[column]
            if column == "samples":
                row.append(str(int(value)))
            else:
                row.append(_format_float(value))
        lines.append(",".join(row))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raw_csv(
    output_path: Path, samples: list[dict[str, float]], timestamps: list[float], keys: list[str]
) -> None:
    start_t = timestamps[0]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "time_s", *keys])
        for idx, (sample, timestamp) in enumerate(zip(samples, timestamps, strict=True)):
            writer.writerow([idx, f"{timestamp - start_t:.6f}", *[sample.get(key, "") for key in keys]])


def _downsample_indices(length: int, max_points: int) -> list[int]:
    if length <= max_points:
        return list(range(length))
    if max_points < 2:
        return [0]
    return sorted({round(i * (length - 1) / (max_points - 1)) for i in range(max_points)})


def _polyline_points(
    times: list[float],
    values: list[float],
    width: int,
    height: int,
    margin_left: int,
    margin_top: int,
    margin_right: int,
    margin_bottom: int,
) -> str:
    t_min, t_max = min(times), max(times)
    v_min, v_max = min(values), max(values)
    if t_max <= t_min:
        t_max = t_min + 1.0
    if v_max <= v_min:
        v_max = v_min + 1.0

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    points: list[str] = []
    for t, v in zip(times, values, strict=True):
        x = margin_left + (t - t_min) / (t_max - t_min) * plot_w
        y = margin_top + (v_max - v) / (v_max - v_min) * plot_h
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _write_plot_html(
    output_path: Path,
    samples: list[dict[str, float]],
    timestamps: list[float],
    keys: list[str],
    max_points: int,
) -> None:
    indices = _downsample_indices(len(samples), max_points)
    start_t = timestamps[0]
    times = [timestamps[idx] - start_t for idx in indices]

    width = 1000
    height = 230
    margin_left = 70
    margin_top = 20
    margin_right = 20
    margin_bottom = 40

    blocks = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Joint angle curves</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#1f2937}"
        "h1{font-size:22px}h2{font-size:16px;margin-top:28px}"
        "svg{max-width:100%;height:auto;border:1px solid #d1d5db;background:#fff}"
        ".meta{color:#6b7280}</style></head><body>",
        "<h1>Joint Angle Curves</h1>",
        f"<p class='meta'>samples={len(samples)}, plotted_points={len(indices)}, duration_s={times[-1]:.3f}</p>",
    ]

    for key in keys:
        values = [samples[idx][key] for idx in indices if key in samples[idx]]
        key_times = [timestamps[idx] - start_t for idx in indices if key in samples[idx]]
        if not values:
            continue
        v_min, v_max = min(values), max(values)
        points = _polyline_points(
            key_times, values, width, height, margin_left, margin_top, margin_right, margin_bottom
        )
        x0, y0 = margin_left, height - margin_bottom
        x1, y1 = width - margin_right, margin_top
        safe_key = html.escape(key)
        blocks.extend(
            [
                f"<h2>{safe_key}</h2>",
                f"<p class='meta'>min={v_min:.4f}, max={v_max:.4f}, range={v_max - v_min:.4f}</p>",
                f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{safe_key} angle curve'>",
                f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#9ca3af'/>",
                f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#9ca3af'/>",
                f"<text x='{x0}' y='{height - 10}' font-size='12'>0s</text>",
                f"<text x='{x1 - 70}' y='{height - 10}' font-size='12'>{key_times[-1]:.1f}s</text>",
                f"<text x='8' y='{margin_top + 12}' font-size='12'>{v_max:.2f}</text>",
                f"<text x='8' y='{y0}' font-size='12'>{v_min:.2f}</text>",
                f"<polyline points='{points}' fill='none' stroke='#2563eb' stroke-width='1.8'/>",
                "</svg>",
            ]
        )

    blocks.append("</body></html>")
    output_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _print_live_progress(
    remaining_s: float,
    samples: list[dict[str, float]],
    focus_keys: list[str],
    stats_keys: list[str],
) -> None:
    focus = focus_keys or stats_keys[:4]
    print(f"Recording... remaining {remaining_s:5.1f}s | samples={len(samples)}")
    if samples:
        latest = samples[-1]
        for key in focus:
            if key in latest:
                print(f"{key:<26} {latest[key]:>9.3f}")
    if focus:
        move_cursor_up(len(focus) + 1)
    else:
        move_cursor_up(1)


def _teleop_record_loop(
    teleop: Teleoperator,
    robot: Robot,
    cfg: JointDeltaCalibrateConfig,
) -> tuple[list[dict[str, float]], list[float], list[str], list[float]]:
    teleop_action_processor, robot_action_processor, _ = make_default_processors()
    samples: list[dict[str, float]] = []
    timestamps: list[float] = []
    loop_timestamps: list[float] = []
    key_order: list[str] | None = None

    start = time.perf_counter()
    warmup_end = start + max(cfg.warmup_s, 0.0)
    record_end = warmup_end + max(cfg.duration_s, 0.0)
    last_print = 0.0

    print("\nWarmup/control check started. Move carefully; samples are not recorded yet.")
    print("When recording starts, move only the joint you are calibrating.")

    try:
        while True:
            loop_start = time.perf_counter()
            loop_timestamps.append(loop_start)

            obs = robot.get_observation()
            numeric_sample = _collect_numeric_sample(obs, key_order)
            if key_order is None and numeric_sample:
                key_order = _make_key_order(robot, numeric_sample)
                numeric_sample = _collect_numeric_sample(obs, key_order)

            raw_action = teleop.get_action()
            teleop_action: RobotAction = teleop_action_processor((raw_action, obs))
            robot_action_to_send: RobotAction = robot_action_processor((teleop_action, obs))
            robot.send_action(robot_action_to_send)

            now = time.perf_counter()
            if now >= warmup_end and numeric_sample:
                if not samples:
                    print("\nRecording started. Press Ctrl+C to stop early and save the report.")
                samples.append(numeric_sample)
                timestamps.append(now)

            if cfg.print_every_s > 0 and now - last_print >= cfg.print_every_s:
                if now < warmup_end:
                    print(f"Warmup... remaining {warmup_end - now:5.1f}s")
                    move_cursor_up(1)
                elif samples and key_order:
                    _print_live_progress(
                        record_end - now, samples, _parse_focus_keys(cfg.focus_keys), key_order
                    )
                last_print = now

            if now >= record_end:
                break

            precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - loop_start), 0.0))
    except KeyboardInterrupt:
        print("\nInterrupted during recording loop. Saving partial results.")

    return samples, timestamps, key_order or [], loop_timestamps


@parser.wrap()
def joint_delta_calibrate(cfg: JointDeltaCalibrateConfig) -> Path | None:
    init_logging()
    sanity_check_bimanual_piper_pair(cfg.robot, cfg.teleop)
    logging.info(pformat(asdict(cfg)))

    if cfg.fps <= 0:
        raise ValueError("`fps` must be > 0.")
    if cfg.duration_s <= 0:
        raise ValueError("`duration_s` must be > 0.")

    output_path = _resolve_output_path(cfg.output_path, cfg.label)
    focus_keys = _parse_focus_keys(cfg.focus_keys)

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    samples: list[dict[str, float]] = []
    timestamps: list[float] = []
    key_order: list[str] = []
    loop_timestamps: list[float] = []

    print("\nThis script will teleoperate the follower and record follower joint observations.")
    print("No policy is loaded. No cameras are required.")
    print(f"Output report: {output_path}")
    if focus_keys:
        print(f"Focused keys: {', '.join(focus_keys)}")
    if cfg.require_enter_to_start:
        input("Check that the workspace is safe, then press ENTER to connect and start...")

    try:
        print("Connecting leader teleop...")
        teleop.connect()
        print("Connecting follower robot...")
        robot.connect()
        print("Connected.")

        samples, timestamps, key_order, loop_timestamps = _teleop_record_loop(teleop, robot, cfg)
    except KeyboardInterrupt:
        print("\nInterrupted before recording loop returned. Saving partial results if available.")
    finally:
        print("\nDisconnecting devices...")
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

    if len(samples) < 2:
        print("Not enough samples recorded; no report was written.")
        return None

    elapsed_s = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
    actual_fps = (len(timestamps) - 1) / elapsed_s if elapsed_s > 0 else 0.0
    stats = _compute_stats(samples, timestamps, key_order, cfg)
    sample_interval_summary = _interval_summary(_diffs(timestamps))
    control_interval_summary = _interval_summary(_diffs(loop_timestamps))
    _write_report(
        output_path,
        cfg,
        stats,
        focus_keys,
        len(samples),
        elapsed_s,
        actual_fps,
        sample_interval_summary,
        control_interval_summary,
    )

    print(f"\nSaved joint delta report to: {output_path}")
    if cfg.save_raw_csv:
        raw_output_path = _resolve_sidecar_path(cfg.raw_output_path, output_path, ".raw_states.csv")
        _write_raw_csv(raw_output_path, samples, timestamps, key_order)
        print(f"Saved raw joint states to: {raw_output_path}")
    if cfg.save_plot_html:
        plot_output_path = _resolve_sidecar_path(cfg.plot_output_path, output_path, ".plot.html")
        _write_plot_html(plot_output_path, samples, timestamps, key_order, cfg.plot_max_points)
        print(f"Saved joint angle curves to: {plot_output_path}")
    if focus_keys:
        print("\nFocused recommendations:")
        for key in focus_keys:
            if key in stats:
                item = stats[key]
                print(
                    f"{key}: q95_delta={item['delta_q95']:.4f}, "
                    f"q99_delta={item['delta_q99']:.4f}, "
                    f"recommended_delta_limit={item['recommended_delta_limit']:.4f}"
                )
    return output_path


def main():
    register_third_party_plugins()
    joint_delta_calibrate()


if __name__ == "__main__":
    main()
