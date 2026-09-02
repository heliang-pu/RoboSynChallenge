#!/usr/bin/env python3
"""Monte Carlo safety audit for constrained sample_loading collection configs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_CONFIG_ROOT = Path("configs/sample_loading")
DEFAULT_SUMMARY = Path("report/sample_loading_random_coverage/coverage_summary.json")
DEFAULT_OUTPUT = Path(
    "report/sample_loading_random_coverage/coverage_config_safety.csv"
)


def rotation_x(degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(degrees)
    result = np.zeros((len(degrees), 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1.0
    result[:, 1, 1] = np.cos(radians)
    result[:, 1, 2] = -np.sin(radians)
    result[:, 2, 1] = np.sin(radians)
    result[:, 2, 2] = np.cos(radians)
    return result


def rotation_y(degrees: float, count: int) -> np.ndarray:
    radians = np.deg2rad(degrees)
    value = np.array(
        [
            [np.cos(radians), 0.0, np.sin(radians)],
            [0.0, 1.0, 0.0],
            [-np.sin(radians), 0.0, np.cos(radians)],
        ],
        dtype=np.float64,
    )
    return np.repeat(value[None], count, axis=0)


def rotation_z(degrees: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(degrees)
    result = np.zeros((len(degrees), 3, 3), dtype=np.float64)
    result[:, 0, 0] = np.cos(radians)
    result[:, 0, 1] = -np.sin(radians)
    result[:, 1, 0] = np.sin(radians)
    result[:, 1, 1] = np.cos(radians)
    result[:, 2, 2] = 1.0
    return result


def projected_obb_separation_2d(
    first_position: np.ndarray,
    second_position: np.ndarray,
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
    first_half_extents: np.ndarray,
    second_half_extents: np.ndarray,
) -> np.ndarray:
    first_generators = first_rotation[:, :2, :] * first_half_extents[None, None]
    second_generators = second_rotation[:, :2, :] * second_half_extents[None, None]
    generators = np.concatenate((first_generators, second_generators), axis=2)
    axes = np.stack((-generators[:, 1, :], generators[:, 0, :]), axis=-1)
    norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    valid = norms[..., 0] > 1e-9
    axes = axes / np.maximum(norms, 1e-9)

    first_radius = np.sum(
        np.abs(np.einsum("bai,bij->baj", axes, first_rotation[:, :2, :]))
        * first_half_extents[None, None],
        axis=-1,
    )
    second_radius = np.sum(
        np.abs(np.einsum("bai,bij->baj", axes, second_rotation[:, :2, :]))
        * second_half_extents[None, None],
        axis=-1,
    )
    delta = second_position[:, :2] - first_position[:, :2]
    center_projection = np.abs(np.einsum("bai,bi->ba", axes, delta))
    gaps = center_projection - first_radius - second_radius
    gaps[~valid] = -np.inf
    return np.max(gaps, axis=1)


def uniform(rng: np.random.Generator, limits: list, count: int) -> np.ndarray:
    low, high = np.asarray(limits[0], dtype=float), np.asarray(limits[1], dtype=float)
    return rng.uniform(low, high, size=(count, 3))


def audit_config(path: Path, count: int, seed: int) -> dict:
    config = json.loads(path.read_text())
    params = config["env"]["events"]["randomize_tube_rack_pose_constrained"]["params"]
    rng = np.random.default_rng(seed)
    first_position = uniform(rng, params["first_position_range"], count)
    second_position = uniform(rng, params["second_position_range"], count)
    first_euler = uniform(rng, params["first_rotation_range"], count)
    second_euler = uniform(rng, params["second_rotation_range"], count)

    # Config uses relative rotation. Initial tube rotation is Ry(90 deg) and
    # initial rack rotation is Rz(80 deg).
    first_rotation = rotation_y(90.0, count) @ rotation_x(first_euler[:, 0])
    second_rotation = rotation_z(80.0 + second_euler[:, 2])
    clearance = projected_obb_separation_2d(
        first_position,
        second_position,
        first_rotation,
        second_rotation,
        np.asarray(params["first_half_extents"]),
        np.asarray(params["second_half_extents"]),
    )
    center_distance = np.linalg.norm(
        first_position[:, :2] - second_position[:, :2], axis=1
    )
    valid = (clearance >= params["min_xy_clearance"]) & (
        center_distance <= params["max_xy_center_distance"]
    )
    accepted_clearance = clearance[valid]
    accepted_distance = center_distance[valid]
    if not np.any(valid):
        raise RuntimeError(f"No valid candidates for {path}")

    acceptance_rate = float(valid.mean())
    failure_probability = float(
        (1.0 - acceptance_rate) ** params["max_resample_attempts"]
    )
    return {
        "coverage_group": path.parent.name,
        "candidates": count,
        "accepted": int(valid.sum()),
        "acceptance_rate": acceptance_rate,
        "min_accepted_clearance_m": float(accepted_clearance.min()),
        "max_accepted_center_distance_m": float(accepted_distance.max()),
        "estimated_256_attempt_failure_probability": failure_probability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.candidates < 100_000:
        raise ValueError("Use at least 100,000 candidates per coverage group")

    summary = json.loads(args.summary.read_text())
    rows = []
    for index, recommendation in enumerate(summary["recommendations"]):
        path = args.config_root / recommendation["name"] / "gym_config.json"
        row = audit_config(path, args.candidates, args.seed + index)
        if row["acceptance_rate"] < 0.01:
            raise RuntimeError(
                f"Acceptance rate is too low for {row['coverage_group']}: "
                f"{row['acceptance_rate']:.3%}"
            )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
