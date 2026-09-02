#!/usr/bin/env python
"""Prove that derived value-model QC episodes do not overlap training data."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def action_hashes(root: Path) -> dict[int, str]:
    per_episode: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for parquet_file in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)):
        table = pq.read_table(parquet_file, columns=["episode_index", "frame_index", "action"])
        episode = np.asarray(table["episode_index"], dtype=np.int64)
        frame = np.asarray(table["frame_index"], dtype=np.int64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        for episode_index in np.unique(episode):
            mask = episode == episode_index
            per_episode.setdefault(int(episode_index), []).append((frame[mask], action[mask]))

    result = {}
    for episode_index, parts in per_episode.items():
        frames = np.concatenate([part[0] for part in parts])
        actions = np.concatenate([part[1] for part in parts])
        order = np.argsort(frames)
        payload = np.ascontiguousarray(actions[order]).tobytes()
        result[episode_index] = hashlib.sha256(payload).hexdigest()
    return result


def episode_labels(root: Path) -> dict[int, bool]:
    result = {}
    for parquet_file in sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True)):
        schema = pq.read_schema(parquet_file)
        if "episode_success" not in schema.names:
            continue
        table = pq.read_table(parquet_file, columns=["episode_index", "episode_success"])
        for index, label in zip(
            table["episode_index"].to_pylist(),
            table["episode_success"].to_pylist(),
            strict=True,
        ):
            normalized = str(label).lower()
            if normalized not in {"success", "failure"}:
                raise ValueError(f"bad episode_success={label!r} in {parquet_file}")
            result[int(index)] = normalized == "success"
    if result:
        return result
    sidecar = root / "episode_success.json"
    if not sidecar.is_file():
        raise KeyError(f"{root} has neither episode_success metadata nor sidecar")
    records = json.loads(sidecar.read_text()).get("episodes", [])
    return {int(record["episode_index"]): bool(record["success"]) for record in records}


def sidecar_seeds(root: Path) -> set[int]:
    path = root / "episode_success.json"
    if not path.is_file():
        return set()
    records = json.loads(path.read_text()).get("episodes", [])
    return {int(record["seed"]) for record in records if record.get("seed") is not None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--positive-source", type=Path)
    parser.add_argument("--positive-episodes", type=int, nargs="+")
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_hash = action_hashes(args.train)
    if bool(args.positive_source) != bool(args.positive_episodes):
        raise ValueError("--positive-source and --positive-episodes must be provided together")
    positive_hash: dict[int, str] = {}
    positive_labels: dict[int, bool] = {}
    if args.positive_source:
        positive_all = action_hashes(args.positive_source)
        missing = sorted(set(args.positive_episodes) - set(positive_all))
        if missing:
            raise ValueError(f"positive episodes missing from source: {missing}")
        positive_hash = {index: positive_all[index] for index in args.positive_episodes}
        positive_labels_all = episode_labels(args.positive_source)
        positive_labels = {index: positive_labels_all[index] for index in args.positive_episodes}
    fresh_hash = action_hashes(args.fresh)

    train_values = set(train_hash.values())
    positive_values = set(positive_hash.values())
    fresh_values = set(fresh_hash.values())
    positive_train_overlap = sorted(
        index for index, digest in positive_hash.items() if digest in train_values
    )
    fresh_train_overlap = sorted(index for index, digest in fresh_hash.items() if digest in train_values)
    fresh_positive_overlap = sorted(
        index for index, digest in fresh_hash.items() if digest in positive_values
    )

    fresh_labels = episode_labels(args.fresh)
    source_seeds = sidecar_seeds(args.positive_source) if args.positive_source else set()
    fresh_seeds = sidecar_seeds(args.fresh)
    seed_overlap = sorted(source_seeds & fresh_seeds)

    result = {
        "train": {"episodes": len(train_hash), "unique_action_hashes": len(train_values)},
        "heldout_positive": {
            "episodes": len(positive_hash),
            "success": sum(positive_labels.values()),
            "failure": sum(not value for value in positive_labels.values()),
            "action_hash_overlap_with_train": positive_train_overlap,
        },
        "heldout_fresh": {
            "episodes": len(fresh_hash),
            "success": sum(fresh_labels.values()),
            "failure": sum(not value for value in fresh_labels.values()),
            "unique_seeds": len(fresh_seeds),
            "action_hash_overlap_with_train": fresh_train_overlap,
            "action_hash_overlap_with_heldout_positive": fresh_positive_overlap,
            "seed_overlap_with_original_rollout": seed_overlap,
        },
        "passed": not (
            positive_train_overlap or fresh_train_overlap or fresh_positive_overlap or seed_overlap
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("held-out independence verification failed")


if __name__ == "__main__":
    main()
