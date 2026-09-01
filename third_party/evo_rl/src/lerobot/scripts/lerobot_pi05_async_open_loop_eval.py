#!/usr/bin/env python

"""Offline async-style open-loop evaluation for a PI05 checkpoint on LeRobot data."""

from __future__ import annotations

import argparse
import csv
import html
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import TimedAction
from lerobot.async_inference.robot_client import clip_piper_async_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import PI05Config  # noqa: F401 - registers the pi05 config with draccus.
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


DEFAULT_DATASET_REPO_ID = "phl/phone_left_to_right_red_up_slot_1280x720_merged_122ep_positive"
DEFAULT_DATASET_ROOT = (
    "/home/phl/workspace/Evo-RL/DATA/phl/"
    "phone_left_to_right_red_up_slot_1280x720_merged_122ep_positive"
)
DEFAULT_POLICY_PATH = (
    "/home/phl/workspace/Evo-RL/outputs/train/"
    "pi05_phone_slot_expert_only_122ep_20260708_041237/"
    "checkpoints/004500/pretrained_model"
)
DEFAULT_OUTPUT_DIR = "/home/phl/workspace/Evo-RL/outputs/open_loop/pi05_004500_async_eval"


@dataclass
class EpisodeResult:
    episode: int
    csv_path: Path
    frame_count: int
    mae: float
    p90: float
    max_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episodes", default="", help="Comma-separated episode ids. Overrides random sampling.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--actions-per-chunk", type=int, default=30)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument("--aggregate-fn-name", default="conservative")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plot-max-points", type=int, default=900)
    return parser.parse_args()


def tensor_to_list(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().cpu().flatten().tolist()]


def episode_indices(dataset: LeRobotDataset, episode: int) -> list[int]:
    episode_col = dataset.hf_dataset["episode_index"]
    return [idx for idx, ep in enumerate(episode_col) if int(ep) == episode]


def choose_episodes(dataset: LeRobotDataset, args: argparse.Namespace) -> list[int]:
    if args.episodes.strip():
        return [int(x.strip()) for x in args.episodes.split(",") if x.strip()]

    episodes = sorted(int(x) for x in dataset.hf_dataset.unique("episode_index"))
    rng = random.Random(args.seed)
    return sorted(rng.sample(episodes, min(args.num_episodes, len(episodes))))


def load_policy_and_processors(policy_path: str, device: str):
    policy_cls = get_policy_class("pi05")
    policy = policy_cls.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()

    device_override = {"device": device}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )
    return policy, preprocessor, postprocessor


@torch.no_grad()
def predict_action_chunk(
    policy,
    preprocessor,
    postprocessor,
    frame: dict,
    actions_per_chunk: int,
) -> torch.Tensor:
    observation = {
        key: frame[key]
        for key in policy.config.input_features
        if key in frame
    }
    observation["task"] = frame["task"]
    observation = preprocessor(observation)
    chunk = policy.predict_action_chunk(observation)
    if chunk.ndim != 3:
        chunk = chunk.unsqueeze(0)
    chunk = chunk[:, :actions_per_chunk, :]

    processed_actions = []
    for i in range(chunk.shape[1]):
        processed_actions.append(postprocessor(chunk[:, i, :]))
    return torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu()


def aggregate_actions(
    queue: dict[int, torch.Tensor],
    incoming_actions: list[TimedAction],
    latest_action: int,
    aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[int, torch.Tensor]:
    next_queue = dict(queue)
    for new_action in incoming_actions:
        timestep = new_action.get_timestep()
        if timestep <= latest_action:
            continue
        if timestep in next_queue:
            next_queue[timestep] = aggregate_fn(next_queue[timestep], new_action.get_action())
        else:
            next_queue[timestep] = new_action.get_action()
    return next_queue


def clip_prediction(
    prediction: torch.Tensor,
    state: torch.Tensor,
    action_keys: list[str],
    fps: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    action = {key: float(prediction[i]) for i, key in enumerate(action_keys)}
    observation = {key: float(state[i]) for i, key in enumerate(action_keys)}
    clipped, safety_info = clip_piper_async_action(action, observation, fps=fps)
    return torch.tensor([clipped[key] for key in action_keys], dtype=torch.float32), safety_info


def evaluate_episode(
    dataset: LeRobotDataset,
    episode: int,
    policy,
    preprocessor,
    postprocessor,
    action_keys: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> EpisodeResult:
    indices = episode_indices(dataset, episode)
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    if not indices:
        raise RuntimeError(f"Episode {episode} has no frames.")

    aggregate_fn = get_aggregate_function(args.aggregate_fn_name)
    threshold_size = max(1, int(args.actions_per_chunk * args.chunk_size_threshold))
    queue: dict[int, torch.Tensor] = {}
    latest_action = -1
    rows: list[dict[str, float | int | str]] = []
    abs_errors: list[float] = []

    for local_t, dataset_idx in enumerate(indices):
        frame = dataset[dataset_idx]
        if len(queue) <= threshold_size:
            chunk = predict_action_chunk(policy, preprocessor, postprocessor, frame, args.actions_per_chunk)
            incoming = [
                TimedAction(
                    timestamp=float(frame["timestamp"]) + i / args.fps,
                    timestep=local_t + i,
                    action=chunk[i].cpu(),
                )
                for i in range(chunk.shape[0])
            ]
            queue = aggregate_actions(queue, incoming, latest_action, aggregate_fn)

        if local_t not in queue:
            chunk = predict_action_chunk(policy, preprocessor, postprocessor, frame, args.actions_per_chunk)
            incoming = [
                TimedAction(
                    timestamp=float(frame["timestamp"]) + i / args.fps,
                    timestep=local_t + i,
                    action=chunk[i].cpu(),
                )
                for i in range(chunk.shape[0])
            ]
            queue = aggregate_actions(queue, incoming, latest_action, aggregate_fn)

        pred_raw = queue.pop(local_t)
        pred_async, safety_info = clip_prediction(pred_raw, frame[OBS_STATE], action_keys, args.fps)
        latest_action = local_t

        gt_action = frame[ACTION].detach().cpu()
        gt_state = frame[OBS_STATE].detach().cpu()
        frame_abs_errors = torch.abs(pred_async - gt_action).tolist()
        abs_errors.extend(float(x) for x in frame_abs_errors)

        row: dict[str, float | int | str] = {
            "episode": episode,
            "local_timestep": local_t,
            "dataset_index": int(frame["index"]),
            "frame_index": int(frame["frame_index"]),
            "timestamp": float(frame["timestamp"]),
            "task": frame["task"],
            "num_abs_clipped": int(safety_info["num_abs_clipped"]),
            "num_speed_clipped": int(safety_info["num_speed_clipped"]),
            "max_abs_delta_before_clip": float(safety_info["max_abs_delta_before_clip"]),
            "max_abs_delta_after_clip": float(safety_info["max_abs_delta_after_clip"]),
        }
        for i, key in enumerate(action_keys):
            row[f"{key}.action"] = float(gt_action[i])
            row[f"{key}.observation_state"] = float(gt_state[i])
            row[f"{key}.pred_raw"] = float(pred_raw[i])
            row[f"{key}.pred_async"] = float(pred_async[i])
            row[f"{key}.abs_error"] = abs(float(pred_async[i]) - float(gt_action[i]))
        rows.append(row)

    csv_path = output_dir / f"episode_{episode:03d}_async_open_loop.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sorted_errors = sorted(abs_errors)
    p90 = sorted_errors[min(len(sorted_errors) - 1, int(0.9 * len(sorted_errors)))]
    return EpisodeResult(
        episode=episode,
        csv_path=csv_path,
        frame_count=len(rows),
        mae=sum(abs_errors) / len(abs_errors),
        p90=p90,
        max_error=max(abs_errors),
    )


def downsample_indices(length: int, max_points: int) -> list[int]:
    if length <= max_points:
        return list(range(length))
    return sorted(set(round(i * (length - 1) / (max_points - 1)) for i in range(max_points)))


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) / (src_max - src_min) * (dst_max - dst_min)


def svg_polyline(xs: list[float], ys: list[float], cls: str) -> str:
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=True))
    return f"<polyline class='{cls}' points='{points}' />"


def render_joint_svg(rows: list[dict[str, str]], key: str, max_points: int) -> str:
    width, height = 720, 330
    left, right, top, bottom = 58, 20, 28, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    indices = downsample_indices(len(rows), max_points)
    times = [float(rows[i]["local_timestep"]) / 30.0 for i in indices]
    series = {
        "action": [float(rows[i][f"{key}.action"]) for i in indices],
        "observation_state": [float(rows[i][f"{key}.observation_state"]) for i in indices],
        "pred_async": [float(rows[i][f"{key}.pred_async"]) for i in indices],
    }
    values = [v for vals in series.values() for v in vals]
    y_min, y_max = min(values), max(values)
    pad = max(5.0, (y_max - y_min) * 0.08)
    y_min -= pad
    y_max += pad
    x_min, x_max = min(times), max(times) if max(times) > min(times) else min(times) + 1

    x_pts = [scale(t, x_min, x_max, left, left + plot_w) for t in times]
    lines = []
    for name, vals in series.items():
        y_pts = [scale(v, y_min, y_max, top + plot_h, top) for v in vals]
        lines.append(svg_polyline(x_pts, y_pts, name))

    grid = []
    for i in range(5):
        y = top + plot_h * i / 4
        value = y_max - (y_max - y_min) * i / 4
        grid.append(f"<line class='grid' x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' />")
        grid.append(f"<text class='tick' x='{left - 10}' y='{y + 4:.2f}' text-anchor='end'>{value:.1f}</text>")
    for i in range(5):
        x = left + plot_w * i / 4
        value = x_min + (x_max - x_min) * i / 4
        grid.append(f"<line class='grid' x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{top + plot_h}' />")
        grid.append(f"<text class='tick' x='{x:.2f}' y='{height - 16}' text-anchor='middle'>{value:.1f}s</text>")

    return f"""
<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(key)}'>
  <text class='chart-title' x='18' y='22'>{html.escape(key)}</text>
  {''.join(grid)}
  <line class='axis' x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' />
  <line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' />
  {''.join(lines)}
</svg>
"""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_html(output_path: Path, results: list[EpisodeResult], action_keys: list[str], args: argparse.Namespace):
    cards = []
    for result in results:
        rows = read_csv_rows(result.csv_path)
        charts = "\n".join(render_joint_svg(rows, key, args.plot_max_points) for key in action_keys)
        cards.append(
            f"""
<section class='episode'>
  <h2>Episode {result.episode} <span>{result.frame_count} frames, MAE {result.mae:.2f}, P90 {result.p90:.2f}</span></h2>
  <div class='charts'>{charts}</div>
</section>
"""
        )

    output_path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>PI05 Async Open-Loop Evaluation</title>
<style>
body {{ margin: 0; background: #0b0f19; color: #e5e7eb; font-family: Inter, Arial, sans-serif; }}
header {{ position: sticky; top: 0; z-index: 2; background: #111827; border-bottom: 1px solid #253044; padding: 18px 28px; }}
h1 {{ margin: 0 0 8px; font-size: 24px; }}
.meta {{ color: #aab3c5; font-size: 14px; line-height: 1.5; }}
.legend {{ display: flex; gap: 22px; margin-top: 10px; color: #d4d9e5; }}
.swatch {{ display: inline-block; width: 22px; height: 4px; margin-right: 8px; vertical-align: middle; border-radius: 999px; }}
.action-s {{ background: #f97316; }}
.state-s {{ background: #60a5fa; }}
.pred-s {{ background: #22c55e; }}
.episode {{ padding: 22px 28px 34px; }}
h2 {{ font-size: 20px; margin: 0 0 16px; }}
h2 span {{ color: #aab3c5; font-weight: 400; margin-left: 12px; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
svg {{ width: 100%; background: #101422; border: 1px solid #2d3650; border-radius: 8px; }}
.grid {{ stroke: #29334d; stroke-width: 1; stroke-dasharray: 5 6; }}
.axis {{ stroke: #94a3b8; stroke-width: 1.4; }}
.tick {{ fill: #aab3c5; font-size: 12px; }}
.chart-title {{ fill: #dbe3f1; font-weight: 700; font-size: 15px; }}
polyline {{ fill: none; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }}
.action {{ stroke: #f97316; }}
.observation_state {{ stroke: #60a5fa; stroke-dasharray: 7 6; }}
.pred_async {{ stroke: #22c55e; }}
</style>
</head>
<body>
<header>
  <h1>PI05 Async Open-Loop Evaluation</h1>
  <div class='meta'>
    policy: {html.escape(args.policy_path)}<br>
    dataset: {html.escape(args.dataset_root)}<br>
    actions_per_chunk={args.actions_per_chunk}, chunk_size_threshold={args.chunk_size_threshold}, aggregate={html.escape(args.aggregate_fn_name)}
  </div>
  <div class='legend'>
    <span><i class='swatch action-s'></i>dataset action</span>
    <span><i class='swatch state-s'></i>observation.state</span>
    <span><i class='swatch pred-s'></i>async clipped prediction</span>
  </div>
</header>
{''.join(cards)}
</body>
</html>
""",
        encoding="utf-8",
    )


def write_summary(output_path: Path, results: list[EpisodeResult], selected_episodes: list[int], args: argparse.Namespace):
    lines = [
        "PI05 async open-loop evaluation",
        f"policy_path: {args.policy_path}",
        f"dataset_root: {args.dataset_root}",
        f"episodes: {selected_episodes}",
        f"actions_per_chunk: {args.actions_per_chunk}",
        f"chunk_size_threshold: {args.chunk_size_threshold}",
        f"aggregate_fn_name: {args.aggregate_fn_name}",
        "",
        "episode, frames, mae, p90, max_error, csv",
    ]
    for result in results:
        lines.append(
            f"{result.episode}, {result.frame_count}, {result.mae:.4f}, {result.p90:.4f}, "
            f"{result.max_error:.4f}, {result.csv_path}"
        )
    lines.append("")
    if results:
        lines.append(f"mean_episode_mae: {statistics.mean(r.mae for r in results):.4f}")
        lines.append(f"mean_episode_p90: {statistics.mean(r.p90 for r in results):.4f}")
        lines.append(f"max_episode_error: {max(r.max_error for r in results):.4f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    selected_episodes = choose_episodes(dataset, args)
    action_keys = list(dataset.features[ACTION]["names"])

    print(f"Selected episodes: {selected_episodes}", flush=True)
    print(f"Loading policy: {args.policy_path}", flush=True)
    start = time.perf_counter()
    policy, preprocessor, postprocessor = load_policy_and_processors(args.policy_path, args.device)
    print(f"Loaded policy in {time.perf_counter() - start:.1f}s", flush=True)

    results = []
    for episode in selected_episodes:
        print(f"Evaluating episode {episode}...", flush=True)
        result = evaluate_episode(
            dataset,
            episode,
            policy,
            preprocessor,
            postprocessor,
            action_keys,
            args,
            output_dir,
        )
        results.append(result)
        print(
            f"Episode {episode}: frames={result.frame_count}, "
            f"MAE={result.mae:.3f}, P90={result.p90:.3f}, max={result.max_error:.3f}",
            flush=True,
        )

    summary_path = output_dir / "summary.txt"
    html_path = output_dir / "async_open_loop_curves.html"
    write_summary(summary_path, results, selected_episodes, args)
    write_html(html_path, results, action_keys, args)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved curves: {html_path}", flush=True)


if __name__ == "__main__":
    main()
