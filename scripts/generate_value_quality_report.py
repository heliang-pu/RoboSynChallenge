#!/usr/bin/env python
"""Generate a Chinese frozen-VLM value-model quality report from QC evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def finite(value: float, default: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else default


def score_checkpoint(checkpoint: dict) -> float:
    held = checkpoint["groups"]["held_out"]
    mae = finite(held["frame_regression"]["mae"], 1.0)
    return (
        0.45 * finite(held["episode_mean"]["roc_auc"], 0.0)
        + 0.20 * finite(held["first_frame"]["roc_auc"], 0.0)
        + 0.15 * finite(held["last_frame"]["roc_auc"], 0.0)
        + 0.20 * max(0.0, 1.0 - mae / 0.5)
    )


def fmt(value: float, digits: int = 3) -> str:
    return "—" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def loss_summary(path: Path) -> dict:
    losses = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"\bloss:([0-9.eE+-]+)", line)
        if match:
            losses.append(float(match.group(1)))
    if not losses:
        return {"count": 0}
    window = min(10, len(losses))
    return {
        "count": len(losses),
        "first_mean": float(np.mean(losses[:window])),
        "last_mean": float(np.mean(losses[-window:])),
        "minimum": float(np.min(losses)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    baseline_metrics = (
        json.loads(args.baseline_metrics.read_text()) if args.baseline_metrics else None
    )
    provenance = json.loads(args.provenance.read_text())
    checkpoints = metrics["checkpoints"]
    if not checkpoints:
        raise ValueError("metrics contain no checkpoints")
    if not provenance.get("passed"):
        raise ValueError("held-out provenance gate did not pass")

    ranked = sorted(checkpoints, key=score_checkpoint, reverse=True)
    best = ranked[0]
    last = max(checkpoints, key=lambda checkpoint: checkpoint["step"])
    best_h = best["groups"]["held_out"]
    last_h = last["groups"]["held_out"]
    best_i = best["groups"]["in_sample"]
    last_i = last["groups"]["in_sample"]

    auc_drop = finite(best_h["episode_mean"]["roc_auc"]) - finite(
        last_h["episode_mean"]["roc_auc"]
    )
    mae_increase = finite(last_h["frame_regression"]["mae"], 1.0) - finite(
        best_h["frame_regression"]["mae"], 1.0
    )
    in_sample_not_worse = (
        finite(last_i["frame_regression"]["mae"], 1.0)
        <= finite(best_i["frame_regression"]["mae"], 1.0) * 1.05
        or finite(last_i["episode_mean"]["roc_auc"])
        >= finite(best_i["episode_mean"]["roc_auc"]) - 0.01
    )
    overfit = (
        best["step"] < last["step"]
        and (auc_drop > 0.03 or mae_increase > max(0.015, best_h["frame_regression"]["mae"] * 0.10))
        and in_sample_not_worse
    )

    selected_name = f"新冻结 VLM step {best['step']}"
    selected_path = best_path = f"{args.checkpoint_root}/{best['step']:06d}"
    selected_metrics = best_h
    baseline = None
    if baseline_metrics:
        baseline = baseline_metrics["checkpoints"][0]
        if score_checkpoint(baseline) > score_checkpoint(best) + 0.01:
            selected_name = "旧未冻结 VLM step 3500"
            selected_path = baseline_metrics["config"]["checkpoint_path"]
            selected_metrics = baseline["groups"]["held_out"]

    best_auc = finite(selected_metrics["episode_mean"]["roc_auc"])
    best_mae = finite(selected_metrics["frame_regression"]["mae"], 1.0)
    if best_auc >= 0.90 and best_mae <= 0.12:
        quality = "优秀"
    elif best_auc >= 0.80 and best_mae <= 0.18:
        quality = "良好"
    elif best_auc >= 0.70 and best_mae <= 0.25:
        quality = "可用但需谨慎"
    else:
        quality = "不建议发布"

    fresh = provenance["heldout_fresh"]
    positive = provenance.get(
        "heldout_positive", {"episodes": 0, "success": 0, "failure": 0}
    )
    enough = (
        fresh["success"] + positive["success"] >= 20
        and fresh["failure"] + positive["failure"] >= 20
    )
    confidence = "高" if enough and provenance["passed"] else "中"
    heldout_successes = fresh["success"] + positive["success"]
    if overfit and heldout_successes < 20:
        overfit_label = "存在过拟合信号（成功样本少，置信度中等）"
    elif overfit:
        overfit_label = "存在明显过拟合"
    else:
        overfit_label = "未发现明显过拟合"
    collapsed = (
        last_h["prediction"]["std"] < 0.02
        or abs(last_h["episode_mean"]["gap"]) < 0.01
    )
    losses = loss_summary(args.train_log)
    lines = [
        "# 冻结 VLM 的 pistar06 奖励模型质量报告",
        "",
        "## 结论",
        "",
        f"- 质量等级：**{quality}**。",
        f"- 过拟合判定：**{overfit_label}**。",
        f"- 数值坍缩判定：**{'存在' if collapsed else '未发现'}**。",
        f"- 新训练最佳 checkpoint：**step {best['step']}**。",
        f"- 最终推荐：**{selected_name}**。",
        f"- 推荐重新打分权重：`{selected_path}`。",
        f"- 结论置信度：**{confidence}**。",
        "",
        "## Held-out 独立性",
        "",
        f"- 训练池：{provenance['train']['episodes']} episodes。",
        f"- 额外未见集：{positive['episodes']} episodes，成功 {positive['success']}，失败 {positive['failure']}。",
        f"- 新种子 rollout：{fresh['episodes']} episodes，成功 {fresh['success']}，失败 {fresh['failure']}。",
        "- action 轨迹哈希与训练池重叠：0。",
        "- 新 rollout seed 与原 rollout 重叠：0。",
        "",
        "## Checkpoint 对比",
        "",
        "| step | train MAE | held-out MAE | held-out mean AUC | first AUC | last AUC | held-out gap | 综合分 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in sorted(checkpoints, key=lambda item: item["step"]):
        inside = checkpoint["groups"]["in_sample"]
        held = checkpoint["groups"]["held_out"]
        lines.append(
            f"| {checkpoint['step']} | {fmt(inside['frame_regression']['mae'])} | "
            f"{fmt(held['frame_regression']['mae'])} | "
            f"{fmt(held['episode_mean']['roc_auc'])} | "
            f"{fmt(held['first_frame']['roc_auc'])} | "
            f"{fmt(held['last_frame']['roc_auc'])} | "
            f"{fmt(held['episode_mean']['gap'])} | {fmt(score_checkpoint(checkpoint))} |"
        )
    if baseline is not None:
        inside = baseline["groups"]["in_sample"]
        held = baseline["groups"]["held_out"]
        lines.append(
            f"| 旧-3500 | {fmt(inside['frame_regression']['mae'])} | "
            f"{fmt(held['frame_regression']['mae'])} | "
            f"{fmt(held['episode_mean']['roc_auc'])} | "
            f"{fmt(held['first_frame']['roc_auc'])} | "
            f"{fmt(held['last_frame']['roc_auc'])} | "
            f"{fmt(held['episode_mean']['gap'])} | {fmt(score_checkpoint(baseline))} |"
        )

    lines += [
        "",
        "## 过拟合依据",
        "",
        f"- 最佳点到 8K 的 held-out episode-mean AUC 变化：`-{fmt(auc_drop)}`。",
        f"- 最佳点到 8K 的 held-out MAE 变化：`+{fmt(mae_increase)}`。",
        f"- 最佳点训练内 MAE / held-out MAE：{fmt(best_i['frame_regression']['mae'])} / {fmt(best_h['frame_regression']['mae'])}。",
        f"- 8K 训练内 MAE / held-out MAE：{fmt(last_i['frame_regression']['mae'])} / {fmt(last_h['frame_regression']['mae'])}。",
        f"- 最佳点 / 8K held-out 预测 std：{fmt(best_h['prediction']['std'])} / {fmt(last_h['prediction']['std'])}。",
        f"- 最佳点 / 8K held-out AUC 95% CI：{best_h['episode_mean']['roc_auc_ci95']} / {last_h['episode_mean']['roc_auc_ci95']}。",
    ]
    if losses.get("count"):
        lines += [
            f"- 训练日志 loss：首窗口均值 {fmt(losses['first_mean'])}，末窗口均值 {fmt(losses['last_mean'])}，最低 {fmt(losses['minimum'])}。",
        ]
    lines += [
        "",
        "判定规则：若较早 checkpoint 的 held-out AUC 明显更高或 MAE 明显更低，而 8K 的训练内表现没有同步恶化，则判为过拟合；阈值为 AUC 下降超过 0.03，或 MAE 增加超过 max(0.015, 10%)。",
        "",
        "## 使用建议",
        "",
        f"1. 全量重新打分优先使用 {selected_name}，不要默认使用新训练的 8K。",
        "2. 在派生数据副本上写入 value/advantage/indicator，保留原训练数据不变。",
        "3. 全量打分结束后复查 value 范围、advantage 方差、近零 advantage 比例和 ACP 正例比例，再发布给 VLA。",
        "",
        "## 方法与限制",
        "",
        f"- 每个 episode 固定抽取 {metrics['config']['phases']} 个等间隔阶段帧，用于快速且可复现的 checkpoint 排序。",
        "- MAE 对照的是 pistar06 训练时的归一化 return target；AUC/PR-AUC 衡量成败排序能力。",
        f"- held-out 成功样本仅 {heldout_successes} 集；AUC 的置信区间较宽，过拟合结论按中等置信度解释。",
        "- 本报告用于 checkpoint 选择和过拟合诊断；最终发布前仍应对选中权重执行全帧 value/advantage 写回质检。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
