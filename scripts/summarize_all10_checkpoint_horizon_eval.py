#!/usr/bin/env python3
"""Generate live CSV/Markdown reports for the all-checkpoint horizon sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TASKS = (
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
DEFAULT_HORIZONS = (10, 50)
DEFAULT_PROTOCOL_REVISION = "all10_h64_v2_bounded_texture_pool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-final-step", type=int, default=99_999)
    parser.add_argument("--checkpoint-interval", type=int, default=2_500)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--protocol-revision", default=DEFAULT_PROTOCOL_REVISION)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-selection-json", type=Path)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            file.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def atomic_json_write(path: Path, payload: dict) -> None:
    atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def available_checkpoints(root: Path) -> list[int]:
    result = []
    if not root.is_dir():
        return result
    for path in root.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        if (
            (path / "_CHECKPOINT_METADATA").is_file()
            and (path / "params" / "manifest.ocdbt").is_file()
            and next((path / "assets").rglob("norm_stats.json"), None) is not None
        ):
            result.append(int(path.name))
    return sorted(result)


def portable_artifact_path(root: Path, result_path: Path, value: object) -> object:
    """Rebase host-local artifact paths to the shared results-root namespace."""
    if not isinstance(value, str) or not value:
        return value
    normalized = value.replace("\\", "/")
    marker = "/attempts/"
    if marker not in normalized:
        return Path(normalized).as_posix() if not Path(normalized).is_absolute() else value
    attempt_suffix = normalized.split(marker, 1)[1]
    job_relative = result_path.parent.relative_to(root)
    return (job_relative / "attempts" / attempt_suffix).as_posix()


def load_results(
    root: Path, expected_episodes: int, protocol_revision: str
) -> tuple[dict[tuple[int, int, str], dict], list[str]]:
    results: dict[tuple[int, int, str], dict] = {}
    errors: list[str] = []
    for path in root.glob("runs/checkpoint_*/h_*/*/job_result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_revision") != protocol_revision:
                raise ValueError(
                    f"protocol_revision={payload.get('protocol_revision')!r}, "
                    f"expected={protocol_revision!r}"
                )
            key = (
                int(payload["checkpoint"]),
                int(payload["execution_horizon"]),
                str(payload["task"]),
            )
            if int(payload["episode_count"]) != expected_episodes:
                raise ValueError(f"episode_count={payload['episode_count']}")
            if int(payload["video_count"]) != expected_episodes:
                raise ValueError(f"video_count={payload['video_count']}")
            for field in ("metrics_path", "video_dir"):
                payload[field] = portable_artifact_path(root, path, payload.get(field))
            results[key] = payload
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return results, errors


def rate(successes: int, episodes: int) -> str:
    return "—" if episodes == 0 else f"{successes}/{episodes} ({successes / episodes:.1%})"


def mean_metric(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return None if len(values) != len(items) or not values else sum(values) / len(values)


def group_summary(checkpoint: int | None, horizon: int | None, items: list[dict]) -> dict:
    success_count = sum(int(item["success_count"]) for item in items)
    episode_count = sum(int(item["episode_count"]) for item in items)
    return {
        "checkpoint": checkpoint,
        "execution_horizon": horizon,
        "task_count": len(items),
        "success_count": success_count,
        "episode_count": episode_count,
        "success_rate": success_count / episode_count if episode_count else None,
        "video_count": sum(int(item["video_count"]) for item in items),
        "mean_task_action_steps_ratio": mean_metric(items, "average_action_steps_ratio"),
        "mean_task_inference_time_seconds": mean_metric(
            items, "average_inference_time_seconds"
        ),
    }


def ranking_key(item: dict) -> tuple:
    """Success first; efficient motion/latency break exact success ties."""
    action_ratio = item.get("mean_task_action_steps_ratio")
    latency = item.get("mean_task_inference_time_seconds")
    checkpoint = item.get("checkpoint")
    horizon = item.get("execution_horizon")
    return (
        -int(item["success_count"]),
        math.inf if action_ratio is None else float(action_ratio),
        math.inf if latency is None else float(latency),
        math.inf if checkpoint is None else int(checkpoint),
        math.inf if horizon is None else int(horizon),
    )


def build_selection(
    checkpoints: list[int],
    expected_checkpoints: list[int],
    results: dict[tuple[int, int, str], dict],
    expected_episodes: int,
    protocol_revision: str,
) -> dict:
    expected_tasks = set(DEFAULT_TASKS)
    candidates: list[dict] = []
    by_checkpoint_horizon: dict[tuple[int, int], dict] = {}
    for checkpoint in checkpoints:
        for horizon in DEFAULT_HORIZONS:
            items_by_task = {
                task: results[(checkpoint, horizon, task)]
                for task in DEFAULT_TASKS
                if (checkpoint, horizon, task) in results
            }
            if set(items_by_task) != expected_tasks:
                continue
            items = [items_by_task[task] for task in DEFAULT_TASKS]
            if any(
                int(item["episode_count"]) != expected_episodes
                or int(item["video_count"]) != expected_episodes
                for item in items
            ):
                continue
            summary = group_summary(checkpoint, horizon, items)
            summary["eligible"] = True
            candidates.append(summary)
            by_checkpoint_horizon[(checkpoint, horizon)] = summary
    candidates.sort(key=ranking_key)

    fully_complete_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if all((checkpoint, horizon) in by_checkpoint_horizon for horizon in DEFAULT_HORIZONS)
    ]

    checkpoint_rankings: list[dict] = []
    for checkpoint in fully_complete_checkpoints:
        horizon_summaries = [
            by_checkpoint_horizon[(checkpoint, horizon)] for horizon in DEFAULT_HORIZONS
        ]
        best_for_checkpoint = min(horizon_summaries, key=ranking_key)
        checkpoint_rankings.append(
            {
                "checkpoint": checkpoint,
                "best_execution_horizon": best_for_checkpoint["execution_horizon"],
                "best_horizon_summary": best_for_checkpoint,
                "all_horizons_complete": True,
            }
        )
    checkpoint_rankings.sort(key=lambda item: ranking_key(item["best_horizon_summary"]))

    horizon_rankings: list[dict] = []
    if fully_complete_checkpoints:
        for horizon in DEFAULT_HORIZONS:
            items = []
            for checkpoint in fully_complete_checkpoints:
                items.extend(
                    results[(checkpoint, horizon, task)] for task in DEFAULT_TASKS
                )
            summary = group_summary(None, horizon, items)
            summary["fully_complete_checkpoint_count"] = len(fully_complete_checkpoints)
            summary["checkpoints"] = fully_complete_checkpoints
            horizon_rankings.append(summary)
        horizon_rankings.sort(key=ranking_key)

    expected_keys = {
        (checkpoint, horizon, task)
        for checkpoint in expected_checkpoints
        for horizon in DEFAULT_HORIZONS
        for task in DEFAULT_TASKS
    }
    completed_expected_keys = expected_keys.intersection(results)
    sweep_complete = len(completed_expected_keys) == len(expected_keys)
    status = "final" if sweep_complete else ("provisional" if candidates else "pending")

    return {
        "schema_version": 1,
        "protocol_revision": protocol_revision,
        "status": status,
        "selection_is_final": sweep_complete,
        "eligibility": {
            "tasks_required": list(DEFAULT_TASKS),
            "task_count_required": len(DEFAULT_TASKS),
            "episodes_per_task_required": expected_episodes,
            "videos_per_task_required": expected_episodes,
            "horizons_required_for_checkpoint_ranking": list(DEFAULT_HORIZONS),
            "rule": "A checkpoint x H candidate requires all 10 tasks. A checkpoint weight requires both selected H values.",
        },
        "progress": {
            "expected_job_count": len(expected_keys),
            "completed_expected_job_count": len(completed_expected_keys),
            "eligible_checkpoint_horizon_count": len(candidates),
            "fully_complete_checkpoint_count": len(fully_complete_checkpoints),
            "fully_complete_checkpoints": fully_complete_checkpoints,
        },
        "tie_break_order": [
            "higher_success_count",
            "lower_mean_task_action_steps_ratio",
            "lower_mean_task_inference_time_seconds",
            "lower_checkpoint_step",
            "lower_execution_horizon",
        ],
        "best_joint_checkpoint_horizon": candidates[0] if candidates else None,
        "best_checkpoint_weight": checkpoint_rankings[0] if checkpoint_rankings else None,
        "best_execution_horizon": horizon_rankings[0] if horizon_rankings else None,
        "checkpoint_horizon_rankings": candidates,
        "checkpoint_weight_rankings": checkpoint_rankings,
        "execution_horizon_rankings": horizon_rankings,
    }


def format_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def format_latency(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.1f} ms"


def render_selection_markdown(selection: dict) -> list[str]:
    lines = ["## 最佳 H 与最佳权重", ""]
    status = selection["status"]
    progress = selection["progress"]
    lines += [
        f"- 选择状态：**{status}**。只有完成全部 10 个任务各 20 局的 checkpoint×H 才进入候选榜。",
        f"- 可比较 checkpoint×H：{progress['eligible_checkpoint_horizon_count']}；两个 H 全完成的权重：{progress['fully_complete_checkpoint_count']}。",
        f"- 总任务组合进度：{progress['completed_expected_job_count']}/{progress['expected_job_count']}。",
    ]
    if status != "final":
        lines.append("- 当前结果仅为暂定；所有 checkpoint 全部完成前不会宣布最终最佳。")

    best_joint = selection["best_joint_checkpoint_horizon"]
    if best_joint is None:
        lines += [
            "- 暂无合格 checkpoint×H：当前还没有一个组合完成全部 10 个任务。",
            "- 最佳权重 / 最佳 H：**待定**。",
            "",
        ]
        return lines

    lines.append(
        "- 当前最佳 checkpoint×H：checkpoint **{}** / H=**{}**，{}，平均任务归一化动作步数 {}，平均单次推理延迟 {}。".format(
            best_joint["checkpoint"],
            best_joint["execution_horizon"],
            rate(best_joint["success_count"], best_joint["episode_count"]),
            format_ratio(best_joint["mean_task_action_steps_ratio"]),
            format_latency(best_joint["mean_task_inference_time_seconds"]),
        )
    )
    best_checkpoint = selection["best_checkpoint_weight"]
    best_horizon = selection["best_execution_horizon"]
    lines.append(
        "- 最佳权重：**待定（尚无权重完成两个 H）**。"
        if best_checkpoint is None
        else "- 当前最佳权重：checkpoint **{}**，其最佳 H={}。".format(
            best_checkpoint["checkpoint"], best_checkpoint["best_execution_horizon"]
        )
    )
    lines.append(
        "- 最佳 H：**待定（尚无同一批完整权重可公平比较两个 H）**。"
        if best_horizon is None
        else "- 当前跨完整权重最佳 H：**{}**，{}。".format(
            best_horizon["execution_horizon"],
            rate(best_horizon["success_count"], best_horizon["episode_count"]),
        )
    )
    lines += [
        "",
        "### 合格 checkpoint×H 排名",
        "",
        "| rank | checkpoint | H | success | 动作步数比 | 单次推理延迟 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(selection["checkpoint_horizon_rankings"][:20], 1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                rank,
                item["checkpoint"],
                item["execution_horizon"],
                rate(item["success_count"], item["episode_count"]),
                format_ratio(item["mean_task_action_steps_ratio"]),
                format_latency(item["mean_task_inference_time_seconds"]),
            )
        )
    lines.append("")
    return lines


def render_csv(
    checkpoints: list[int], results: dict[tuple[int, int, str], dict]
) -> str:
    from io import StringIO

    output = StringIO()
    fields = [
        "checkpoint",
        "execution_horizon",
        "task",
        "status",
        "protocol_revision",
        "success_count",
        "episode_count",
        "success_rate",
        "video_count",
        "seed",
        "average_action_steps",
        "average_action_steps_ratio",
        "inference_call_count",
        "average_inference_time_seconds",
        "average_inference_time_per_episode_seconds",
        "metrics_path",
        "video_dir",
        "validated_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for checkpoint in checkpoints:
        for horizon in DEFAULT_HORIZONS:
            for task in DEFAULT_TASKS:
                payload = results.get((checkpoint, horizon, task))
                if payload is None:
                    writer.writerow(
                        {
                            "checkpoint": checkpoint,
                            "execution_horizon": horizon,
                            "task": task,
                            "status": "pending",
                        }
                    )
                    continue
                writer.writerow(
                    {
                        "checkpoint": checkpoint,
                        "execution_horizon": horizon,
                        "task": task,
                        "status": "complete",
                        **{field: payload.get(field) for field in fields if field in payload},
                    }
                )
    return output.getvalue()


def render_markdown(
    checkpoints: list[int],
    expected_checkpoints: list[int],
    results: dict[tuple[int, int, str], dict],
    errors: list[str],
    expected_episodes: int,
    protocol_revision: str,
    selection: dict,
) -> str:
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    available_jobs = len(checkpoints) * len(DEFAULT_HORIZONS) * len(DEFAULT_TASKS)
    final_jobs = len(expected_checkpoints) * len(DEFAULT_HORIZONS) * len(DEFAULT_TASKS)
    completed_available = sum(
        key[0] in checkpoints for key in results
    )
    completed_final = sum(key[0] in expected_checkpoints for key in results)
    videos = sum(int(item["video_count"]) for item in results.values())
    episodes = sum(int(item["episode_count"]) for item in results.values())

    lines = [
        "# π0.5 All-10：全 checkpoint × 执行 horizon 成功率报告",
        "",
        f"生成时间：`{generated}`",
        "",
        "## 评估协议",
        "",
        f"- 协议版本：`{protocol_revision}`。",
        f"- 任务：{len(DEFAULT_TASKS)} 个，每个组合 {expected_episodes} 局。",
        "- 执行 horizon：`10、50`；模型动作 horizon 固定为 64。",
        "- 固定配对 seed：同一任务在所有 checkpoint/horizon 下使用相同 20 个场景。",
        "- 视频：每局一个 2560×480 四视角横向拼接 MP4，依次为左腕、右腕、顶部、全局第三视角。",
        "- setting：`random_3p`；成功由任务正式评估器判定。",
        "",
    ]
    lines += render_selection_markdown(selection)
    lines += [
        "## 当前进度",
        "",
        f"- 已发现 checkpoint：{len(checkpoints)} 个（{', '.join(map(str, checkpoints)) or '无'}）。",
        f"- 当前可运行组合：{available_jobs}；已完成 {completed_available}（{completed_available / available_jobs:.1%}）。"
        if available_jobs
        else "- 当前尚未发现完整 checkpoint。",
        f"- 最终计划：{final_jobs} 个任务组合、{final_jobs * expected_episodes} 局视频；已完成 {completed_final} 个组合。",
        f"- 已验证视频：{videos}；已评估 episodes：{episodes}。",
        "",
        "## Checkpoint × horizon 汇总",
        "",
        "| checkpoint | H=10 | H=50 | 完整任务数 |",
        "|---:|---:|---:|---:|",
    ]

    for checkpoint in checkpoints:
        cells = []
        task_jobs = 0
        for horizon in DEFAULT_HORIZONS:
            items = [
                results[(checkpoint, horizon, task)]
                for task in DEFAULT_TASKS
                if (checkpoint, horizon, task) in results
            ]
            success = sum(int(item["success_count"]) for item in items)
            count = sum(int(item["episode_count"]) for item in items)
            task_jobs += len(items)
            cells.append(rate(success, count))
        lines.append(
            f"| {checkpoint} | {' | '.join(cells)} | "
            f"{task_jobs}/{len(DEFAULT_HORIZONS) * len(DEFAULT_TASKS)} |"
        )

    for checkpoint in checkpoints:
        if not any(key[0] == checkpoint for key in results):
            continue
        lines += ["", f"## Checkpoint {checkpoint} 分任务结果", ""]
        header = "| task | H=10 | H=50 |"
        lines += [header, "|---|---:|---:|"]
        for task in DEFAULT_TASKS:
            cells = []
            for horizon in DEFAULT_HORIZONS:
                item = results.get((checkpoint, horizon, task))
                cells.append(
                    "—"
                    if item is None
                    else rate(int(item["success_count"]), int(item["episode_count"]))
                )
            lines.append(f"| {task} | {' | '.join(cells)} |")

    if errors:
        lines += ["", "## 无效结果记录", ""]
        lines.extend(f"- `{error}`" for error in errors[-50:])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_md = args.output_md or args.results_root / "SUCCESS_RATE_REPORT.md"
    output_csv = args.output_csv or args.results_root / "success_rates.csv"
    output_selection_json = (
        args.output_selection_json or args.results_root / "best_selection.json"
    )
    checkpoints = available_checkpoints(args.checkpoint_root)
    expected_checkpoints = list(
        range(args.checkpoint_interval, args.expected_final_step, args.checkpoint_interval)
    )
    expected_checkpoints.append(args.expected_final_step)
    results, errors = load_results(
        args.results_root, args.expected_episodes, args.protocol_revision
    )
    # Preserve legacy H=30/H=64 artifacts on disk while excluding them from
    # the narrowed H=10/H=50 sweep, progress, CSV, rankings, and final gate.
    results = {
        key: payload for key, payload in results.items() if key[1] in DEFAULT_HORIZONS
    }
    selection = build_selection(
        checkpoints,
        expected_checkpoints,
        results,
        args.expected_episodes,
        args.protocol_revision,
    )
    atomic_write(output_csv, render_csv(checkpoints, results))
    atomic_json_write(output_selection_json, selection)
    atomic_write(
        output_md,
        render_markdown(
            checkpoints,
            expected_checkpoints,
            results,
            errors,
            args.expected_episodes,
            args.protocol_revision,
            selection,
        ),
    )
    print(
        f"report checkpoints={len(checkpoints)} completed_jobs={len(results)} "
        f"markdown={output_md} csv={output_csv} selection={output_selection_json}"
    )


if __name__ == "__main__":
    main()
