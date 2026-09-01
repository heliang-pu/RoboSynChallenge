#!/usr/bin/env python3
"""Reconstruct and audit sample_loading spatial randomization coverage."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import trimesh
from shapely.geometry import MultiPoint

DEFAULT_RAW = Path(
    os.environ.get(
        "ROBOSYN_RAW_SAMPLE_LOADING",
        str(Path.home() / "workspace/dataset/cobotmagic_Sim_sample_loading"),
    )
)
DEFAULT_KEPT = Path(
    os.environ.get(
        "ROBOSYN_DATASET_ROOT",
        str(Path.home() / "FermiBotNas/dataset/RoboSynChallenge/Sim_clean_filtered_pruned"),
    )
) / "cobotmagic_Sim_sample_loading"
DEFAULT_CONFIG = Path("configs/sample_loading/random/gym_config.json")
DEFAULT_OUT = Path("report/sample_loading_random_coverage")
BIN_COUNT = 5


def initial_pose(path: Path, key: str) -> np.ndarray:
    table = pq.read_table(path, columns=[key]).slice(0, 1)
    return np.asarray(table[key][0].as_py(), dtype=np.float64)


def yaw_deg(axis_xy: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(axis_xy[1], axis_xy[0])))


def wrap_degrees(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def footprint(vertices: np.ndarray, pose: np.ndarray):
    world = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
    return MultiPoint(world[:, :2]).convex_hull


def read_rows(dataset: Path, tube_vertices: np.ndarray, rack_vertices: np.ndarray):
    rows = []
    signatures = {}
    for path in sorted((dataset / "data").glob("**/*.parquet")):
        episode = int(path.stem.rsplit("_", 1)[1])
        tube = initial_pose(path, "cube_pose")
        rack = initial_pose(path, "rack_pose")
        tube_shape = footprint(tube_vertices, tube)
        rack_shape = footprint(rack_vertices, rack)
        tube_yaw = yaw_deg(tube[:2, 2])
        rack_yaw = yaw_deg(rack[:2, 0])
        row = {
            "episode_index": episode,
            "tube_x": float(tube[0, 3]),
            "tube_y": float(tube[1, 3]),
            "tube_yaw_deg": tube_yaw,
            "rack_x": float(rack[0, 3]),
            "rack_y": float(rack[1, 3]),
            "rack_yaw_delta_deg": wrap_degrees(rack_yaw - 80.0),
            "center_distance_m": float(np.linalg.norm(tube[:2, 3] - rack[:2, 3])),
            "footprint_clearance_m": float(tube_shape.distance(rack_shape)),
            "footprints_intersect": bool(tube_shape.intersects(rack_shape)),
        }
        signature = tuple(np.round(np.r_[tube.ravel(), rack.ravel()], 6))
        signatures[signature] = episode
        rows.append(row)
    return rows, signatures


def add_normalized_values(rows, bounds):
    for row in rows:
        in_bounds = True
        for name, (low, high) in bounds.items():
            normalized = (row[name] - low) / (high - low)
            row[f"{name}_normalized"] = normalized
            in_bounds &= -0.03 <= normalized <= 1.03
        row["within_config_bounds"] = bool(in_bounds)
        row["safe_2cm_clearance"] = bool(
            not row["footprints_intersect"]
            and row["footprint_clearance_m"] >= 0.02
            and row["center_distance_m"] <= 0.38
        )


def write_csv(path: Path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def marginal_coverage(raw_rows, kept_rows, bounds):
    output = []
    for name, (low, high) in bounds.items():
        edges = np.linspace(low, high, BIN_COUNT + 1)
        raw = np.histogram([row[name] for row in raw_rows], edges)[0]
        kept = np.histogram([row[name] for row in kept_rows], edges)[0]
        for index in range(BIN_COUNT):
            output.append(
                {
                    "parameter": name,
                    "bin_index": index,
                    "low": edges[index],
                    "high": edges[index + 1],
                    "raw_count": int(raw[index]),
                    "kept_count": int(kept[index]),
                    "retention": float(kept[index] / raw[index]) if raw[index] else 0.0,
                    "expected_raw_uniform": len(raw_rows) / BIN_COUNT,
                    "expected_kept_uniform": len(kept_rows) / BIN_COUNT,
                }
            )
    return output


def pairwise_coverage(raw_rows, kept_rows, bounds):
    factor_pairs = [
        ("tube_x", "tube_y"),
        ("rack_x", "rack_y"),
        ("tube_yaw_deg", "rack_yaw_delta_deg"),
        ("tube_y", "rack_y"),
        ("tube_x", "rack_x"),
    ]
    output = []
    for first, second in factor_pairs:
        first_edges = np.linspace(*bounds[first], BIN_COUNT + 1)
        second_edges = np.linspace(*bounds[second], BIN_COUNT + 1)
        raw = np.histogram2d(
            [row[first] for row in raw_rows],
            [row[second] for row in raw_rows],
            [first_edges, second_edges],
        )[0]
        kept = np.histogram2d(
            [row[first] for row in kept_rows],
            [row[second] for row in kept_rows],
            [first_edges, second_edges],
        )[0]
        for i in range(BIN_COUNT):
            for j in range(BIN_COUNT):
                output.append(
                    {
                        "first_parameter": first,
                        "second_parameter": second,
                        "first_bin": i,
                        "second_bin": j,
                        "first_low": first_edges[i],
                        "first_high": first_edges[i + 1],
                        "second_low": second_edges[j],
                        "second_high": second_edges[j + 1],
                        "raw_count": int(raw[i, j]),
                        "kept_count": int(kept[i, j]),
                        "retention": (
                            float(kept[i, j] / raw[i, j]) if raw[i, j] else 0.0
                        ),
                    }
                )
    return output


def plot_overview(path: Path, raw_rows, kept_rows, bounds):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, (name, (low, high)) in zip(axes.flat, bounds.items()):
        edges = np.linspace(low, high, BIN_COUNT + 1)
        axis.hist(
            [row[name] for row in raw_rows],
            bins=edges,
            alpha=0.55,
            label="raw success 1000",
        )
        axis.hist(
            [row[name] for row in kept_rows],
            bins=edges,
            alpha=0.55,
            label="strict kept 756",
        )
        axis.axhline(len(raw_rows) / BIN_COUNT, color="tab:blue", linestyle="--")
        axis.axhline(len(kept_rows) / BIN_COUNT, color="tab:orange", linestyle="--")
        axis.set_title(name)
        axis.set_xlim(low, high)
    axes[0, 0].legend()
    fig.suptitle("sample_loading reconstructed spatial randomization coverage")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def recommendations():
    return [
        {
            "name": "coverage_rack_upper_feasible",
            "episodes": 120,
            "reason": "rack y=0.06-0.09 is under-covered but still has successful examples; x stays in the reachable high-x band",
            "tube_position_range": [[0.45, -0.28, 0.86], [0.60, -0.10, 0.86]],
            "tube_rotation_range": [[-20, 0, 0], [20, 0, 0]],
            "rack_position_range": [[0.686, 0.06, 0.865], [0.70, 0.09, 0.865]],
            "rack_rotation_range": [[0, 0, 0], [0, 0, 90]],
        },
        {
            "name": "coverage_tube_right_lower_y",
            "episodes": 120,
            "reason": "tube high-x is under-covered; lower-y is used because high-x/near-zero-y is collision-prone and lacks success evidence",
            "tube_position_range": [[0.63, -0.28, 0.86], [0.68, -0.17, 0.86]],
            "tube_rotation_range": [[-20, 0, 0], [20, 0, 0]],
            "rack_position_range": [[0.67, 0.0, 0.865], [0.70, 0.06, 0.865]],
            "rack_rotation_range": [[0, 0, 0], [0, 0, 90]],
        },
        {
            "name": "coverage_yaw_low_tube_high_rack",
            "episodes": 80,
            "reason": "tube yaw -20..-12 with rack yaw delta 72..90 has only 13 strict episodes",
            "tube_position_range": [[0.47, -0.25, 0.86], [0.60, -0.10, 0.86]],
            "tube_rotation_range": [[-20, 0, 0], [-12, 0, 0]],
            "rack_position_range": [[0.68, 0.02, 0.865], [0.70, 0.07, 0.865]],
            "rack_rotation_range": [[0, 0, 72], [0, 0, 90]],
        },
        {
            "name": "coverage_yaw_high_tube_high_rack",
            "episodes": 80,
            "reason": "tube yaw 12..20 with rack yaw delta 72..90 has only 13 strict episodes",
            "tube_position_range": [[0.47, -0.25, 0.86], [0.60, -0.10, 0.86]],
            "tube_rotation_range": [[12, 0, 0], [20, 0, 0]],
            "rack_position_range": [[0.68, 0.02, 0.865], [0.70, 0.07, 0.865]],
            "rack_rotation_range": [[0, 0, 72], [0, 0, 90]],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--kept", type=Path, default=DEFAULT_KEPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text())
    events = config["env"]["events"]
    tube_event = events["random_cube_pose"]["params"]
    rack_event = events["random_rack_pose"]["params"]
    bounds = {
        "tube_x": (
            tube_event["position_range"][0][0],
            tube_event["position_range"][1][0],
        ),
        "tube_y": (
            tube_event["position_range"][0][1],
            tube_event["position_range"][1][1],
        ),
        "tube_yaw_deg": (
            tube_event["rotation_range"][0][0],
            tube_event["rotation_range"][1][0],
        ),
        "rack_x": (
            rack_event["position_range"][0][0],
            rack_event["position_range"][1][0],
        ),
        "rack_y": (
            rack_event["position_range"][0][1],
            rack_event["position_range"][1][1],
        ),
        "rack_yaw_delta_deg": (
            rack_event["rotation_range"][0][2],
            rack_event["rotation_range"][1][2],
        ),
    }

    tube_mesh = trimesh.load("assets/test_tube_standard.ply", process=False)
    rack_mesh = trimesh.load("assets/test_tube_rack_standard.ply", process=False)
    tube_vertices = np.asarray(tube_mesh.vertices) * np.asarray([1.1, 1.1, 1.3])
    rack_vertices = np.asarray(rack_mesh.vertices) * np.asarray([1.1, 1.1, 1.1])
    raw_rows, raw_signatures = read_rows(args.raw, tube_vertices, rack_vertices)
    kept_rows, _ = read_rows(args.kept, tube_vertices, rack_vertices)
    add_normalized_values(raw_rows, bounds)
    add_normalized_values(kept_rows, bounds)

    kept_raw_indices = set()
    for path in sorted((args.kept / "data").glob("**/*.parquet")):
        tube = initial_pose(path, "cube_pose")
        rack = initial_pose(path, "rack_pose")
        signature = tuple(np.round(np.r_[tube.ravel(), rack.ravel()], 6))
        if signature not in raw_signatures:
            raise RuntimeError(f"Cannot map kept episode {path} to raw dataset")
        kept_raw_indices.add(raw_signatures[signature])
    for row in raw_rows:
        row["retained_in_strict_dataset"] = row["episode_index"] in kept_raw_indices

    marginal = marginal_coverage(raw_rows, kept_rows, bounds)
    pairwise = pairwise_coverage(raw_rows, kept_rows, bounds)
    write_csv(args.out / "reconstructed_raw_parameters.csv", raw_rows)
    write_csv(args.out / "reconstructed_kept_parameters.csv", kept_rows)
    write_csv(args.out / "marginal_coverage.csv", marginal)
    write_csv(args.out / "pairwise_coverage.csv", pairwise)
    plot_overview(args.out / "coverage_overview.png", raw_rows, kept_rows, bounds)

    plan = recommendations()
    summary = {
        "raw_episodes": len(raw_rows),
        "kept_episodes": len(kept_rows),
        "mapped_kept_episodes": len(kept_raw_indices),
        "random_parameter_bounds": bounds,
        "raw_footprint_intersections": sum(
            row["footprints_intersect"] for row in raw_rows
        ),
        "kept_footprint_intersections": sum(
            row["footprints_intersect"] for row in kept_rows
        ),
        "raw_below_2cm_clearance": sum(
            not row["safe_2cm_clearance"] for row in raw_rows
        ),
        "kept_below_2cm_clearance": sum(
            not row["safe_2cm_clearance"] for row in kept_rows
        ),
        "recommended_collection_episodes": sum(item["episodes"] for item in plan),
        "recommendations": plan,
        "unrecoverable_from_current_parquet": [
            "light position/color/intensity",
            "camera intrinsic/extrinsic offsets",
            "material/texture selections",
            "distractor asset identities and poses",
        ],
    }
    (args.out / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
