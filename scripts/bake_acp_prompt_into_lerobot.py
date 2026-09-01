#!/usr/bin/env python3
"""Bake a frame-level ACP indicator into LeRobot task prompts.

Run this only on a derived dataset copy. For each original task, the script
creates negative and positive task variants and rewrites every frame's
``task_index`` according to its ACP indicator.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"missing parquet column: {name}")
    return table.set_column(index, table.schema.field(index), values)


def atomic_write_table(table: pa.Table, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".baking.tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)


def scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    values = np.asarray(values, dtype=np.int64)
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
        "q01": [float(np.quantile(values, 0.01))],
        "q10": [float(np.quantile(values, 0.10))],
        "q50": [float(np.quantile(values, 0.50))],
        "q90": [float(np.quantile(values, 0.90))],
        "q99": [float(np.quantile(values, 0.99))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--indicator-field",
        default="complementary_info.acp_indicator_round1",
    )
    args = parser.parse_args()
    root = args.dataset.resolve()
    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.parquet"
    marker = root / "ACP_PROMPT_BAKED.json"
    if marker.exists():
        raise RuntimeError(f"dataset is already marked as baked: {marker}")

    info = json.loads(info_path.read_text())
    tasks = pq.read_table(tasks_path).to_pandas()
    text_columns = [column for column in tasks.columns if column != "task_index"]
    if not text_columns and not isinstance(tasks.index, pd.RangeIndex):
        original_prompts = {
            int(task_index): str(prompt)
            for prompt, task_index in tasks["task_index"].items()
        }
    elif len(text_columns) == 1:
        text_column = text_columns[0]
        original_prompts = {
            int(row["task_index"]): str(row[text_column])
            for _, row in tasks.iterrows()
        }
    else:
        raise RuntimeError(
            f"cannot identify task text in columns={tasks.columns.tolist()}, index={tasks.index.name}"
        )

    task_mapping: dict[tuple[int, int], int] = {}
    baked_prompts: dict[int, str] = {}
    for ordinal, old_index in enumerate(sorted(original_prompts)):
        base = original_prompts[old_index]
        for indicator, tag in ((0, "negative"), (1, "positive")):
            new_index = 2 * ordinal + indicator
            task_mapping[(old_index, indicator)] = new_index
            baked_prompts[new_index] = f"{base}\nAdvantage: {tag}" if base else f"Advantage: {tag}"

    all_new_task_indices: list[np.ndarray] = []
    episode_task_indices: dict[int, list[np.ndarray]] = defaultdict(list)
    data_files = sorted((root / "data").glob("**/*.parquet"))
    if not data_files:
        raise RuntimeError(f"no data parquet files under {root / 'data'}")

    for path in data_files:
        table = pq.read_table(path)
        old_indices = np.asarray(table["task_index"], dtype=np.int64)
        indicators = np.asarray(table[args.indicator_field], dtype=np.int64)
        episodes = np.asarray(table["episode_index"], dtype=np.int64)
        if not set(np.unique(indicators)).issubset({0, 1}):
            raise RuntimeError(f"invalid indicator values in {path}: {np.unique(indicators)}")

        new_indices = np.empty_like(old_indices)
        covered = np.zeros(old_indices.shape, dtype=bool)
        for key, new_index in task_mapping.items():
            mask = (old_indices == key[0]) & (indicators == key[1])
            new_indices[mask] = new_index
            covered |= mask
        if not covered.all():
            bad = np.flatnonzero(~covered)[:10].tolist()
            raise RuntimeError(f"unmapped task/indicator pairs in {path}, rows={bad}")

        table = replace_column(table, "task_index", pa.array(new_indices, type=pa.int64()))
        atomic_write_table(table, path)
        all_new_task_indices.append(new_indices)
        for episode in np.unique(episodes):
            episode_task_indices[int(episode)].append(new_indices[episodes == episode])

    baked_task_frame = pd.DataFrame(
        {"task_index": sorted(baked_prompts)},
        index=[baked_prompts[index] for index in sorted(baked_prompts)],
    )
    baked_task_frame.to_parquet(tasks_path)

    episode_files = sorted((root / "meta/episodes").glob("**/*.parquet"))
    for path in episode_files:
        table = pq.read_table(path)
        episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
        rows = table.to_pylist()
        for row, episode in zip(rows, episode_indices, strict=True):
            values = np.concatenate(episode_task_indices[int(episode)])
            present = sorted(np.unique(values).tolist())
            row["tasks"] = [baked_prompts[index] for index in present]
            stats = scalar_stats(values)
            for stat_name, stat_value in stats.items():
                key = f"stats/task_index/{stat_name}"
                if key in row:
                    row[key] = stat_value
        atomic_write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    all_indices = np.concatenate(all_new_task_indices)
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats["task_index"] = scalar_stats(all_indices)
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    info["total_tasks"] = len(baked_prompts)
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    qc = {
        "dataset": str(root),
        "indicator_field": args.indicator_field,
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(all_indices.size),
        "total_tasks": len(baked_prompts),
        "negative_frames": int(np.sum(all_indices % 2 == 0)),
        "positive_frames": int(np.sum(all_indices % 2 == 1)),
        "positive_ratio": float(np.mean(all_indices % 2 == 1)),
        "prompts": [baked_prompts[index] for index in sorted(baked_prompts)],
    }
    marker.write_text(json.dumps(qc, indent=2) + "\n")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
