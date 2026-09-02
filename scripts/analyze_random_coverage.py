#!/usr/bin/env python3
"""Generic randomization-coverage audit for RoboSynChallenge tasks.

Reconstructs per-episode scene layouts (xy + yaw of the key objects) from the
first frame of each episode parquet, compares the empirical distribution with
the official ``configs/<task>/random/gym_config.json`` ranges, and emits:

- ``coverage_summary.json``: per-dim 8-bin histograms, per-object xy grids,
  lowest-density bins and ready-to-build recollection recommendations
  (ranges are expressed in config units so the builder can inject them).
- ``marginals.png`` / ``grids_2d.png`` histograms.
- ``episodes.csv`` cache of the reconstructed per-episode values.

For tasks without object pose columns use ``--stratify-only``: no data is
read; the official ranges are split into blocks (fine 8-way per-dim listing
plus a coarse cross-product used for the generated configs).

Example:
    python scripts/analyze_random_coverage.py --task drawer_open_place \
        --pose-cols duck_pose,drawer_pose --out report/coverage/drawer_open_place
    python scripts/analyze_random_coverage.py --task click_bell --stratify-only
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _coverage_common import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_REPORT_ROOT,
    REPO_ROOT,
    STRATIFY_ONLY_TASKS,
    STRATIFY_TASK_UIDS,
    TASK_POSE_COLUMNS,
    DimSpec,
    band_label,
    allocate_episodes,
    dim_value,
    dims_from_event,
    entity_entry,
    entity_init,
    event_sampling_spec,
    find_pose_event,
    half_extents_from_vertices,
    is_zero_pose,
    load_json,
    mc_pair_acceptance,
    mesh_scaled_vertices,
    pose_matrix_from_value,
    sat_clearance_2d,
)

MARGINAL_DEFICIT_FACTOR = 0.6  # bin is "deficient" below this * uniform expectation
GRID_DEFICIT_FACTOR = 0.5
MAX_MERGED_BINS = 3
MAX_GAP_RECS = 4
GAP_MIN_EPISODES = 40
EPISODE_STEP = 5


# --------------------------------------------------------------------------
# Object bookkeeping
# --------------------------------------------------------------------------
class ObjectSpec:
    def __init__(self, config: dict, uid: str):
        self.uid = uid
        self.event_name, self.event = find_pose_event(config, uid)
        self.dims: list[DimSpec] = dims_from_event(uid, self.event)
        self.init_pos, self.init_rot, self.section = entity_init(config, uid)
        _, entry = entity_entry(config, uid)
        self.entry = entry
        vertices, mesh_path = mesh_scaled_vertices(entry)
        self.mesh_path = mesh_path
        self.half_extents = (
            half_extents_from_vertices(vertices) if vertices is not None else None
        )
        self.observed = False  # set during the data scan

    @property
    def pos_dims(self) -> list[DimSpec]:
        return [d for d in self.dims if d.kind == "pos"]

    @property
    def rot_dim(self) -> DimSpec | None:
        rots = [d for d in self.dims if d.kind == "rot"]
        return rots[0] if rots else None

    def describe(self) -> dict:
        return {
            "event": self.event_name,
            "func": self.event.get("func"),
            "section": self.section,
            "init_pos": [round(float(v), 5) for v in self.init_pos],
            "dims": [d.to_dict() for d in self.dims],
            "half_extents": self.half_extents,
            "mesh_path": self.mesh_path,
        }


def narrowed_ranges(obj: ObjectSpec, narrow: dict[str, tuple[float, float]]) -> dict:
    """Full position/rotation ranges (config units) with selected dims narrowed."""
    params = obj.event.get("params", {})
    override: dict = {}
    position_range = params.get("position_range")
    if position_range is not None:
        pr = [list(map(float, position_range[0])), list(map(float, position_range[1]))]
        for dim in obj.pos_dims:
            if dim.key in narrow:
                lo, hi = narrow[dim.key]
                pr[0][dim.comp], pr[1][dim.comp] = round(lo, 5), round(hi, 5)
        override["position_range"] = pr
    rotation_range = params.get("rotation_range")
    if rotation_range is not None:
        rr = [list(map(float, rotation_range[0])), list(map(float, rotation_range[1]))]
        rot = obj.rot_dim
        if rot is not None and rot.key in narrow:
            lo, hi = narrow[rot.key]
            rr[0][rot.comp], rr[1][rot.comp] = round(lo, 3), round(hi, 3)
        override["rotation_range"] = rr
    return override


# --------------------------------------------------------------------------
# Dataset scan
# --------------------------------------------------------------------------
def resolve_dataset_dir(dataset_root: Path, task: str) -> Path:
    cand = dataset_root / f"cobotmagic_Sim_{task}"
    if (cand / "data").exists():
        return cand
    if (dataset_root / "data").exists():
        return dataset_root
    raise FileNotFoundError(
        f"No dataset found for {task} under {dataset_root} "
        f"(expected cobotmagic_Sim_{task}/data)"
    )


def scan_first_frames(dataset_dir: Path, pose_cols: list[str], limit: int | None):
    import pyarrow.parquet as pq

    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_dir}/data")
    rows = []
    for path in files:
        try:
            episode = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            episode = len(rows)
        table = pq.read_table(path, columns=pose_cols).slice(0, 1)
        poses = {}
        for col in pose_cols:
            pose = pose_matrix_from_value(table[col][0].as_py())
            poses[col] = None if is_zero_pose(pose) else pose
        rows.append({"episode_index": episode, "poses": poses})
    return rows


# --------------------------------------------------------------------------
# Histograms / gap mining
# --------------------------------------------------------------------------
def marginal_stats(values: np.ndarray, dim: DimSpec, bins: int) -> dict:
    edges = np.linspace(dim.low, dim.high, bins + 1)
    in_range = values[(values >= dim.low) & (values <= dim.high)]
    counts, _ = np.histogram(in_range, edges)
    pad = 10.0 if dim.kind == "rot" else max(0.01, 0.05 * (dim.high - dim.low))
    inside = np.mean((values >= dim.low - pad) & (values <= dim.high + pad))
    # Rotation values can be silently corrupted by physics settling (objects
    # rolling after the drop), so a poor in-range fraction disqualifies the
    # dim. Position extraction is exact: out-of-range positions are evidence
    # of a dataset-vs-config distribution shift, which is precisely what the
    # coverage collection should compensate, so pos dims stay eligible.
    return {
        "bin_edges": [round(float(v), 5) for v in edges],
        "counts": [int(c) for c in counts],
        "expected_uniform": round(len(in_range) / bins, 2),
        "n_values": int(values.size),
        "n_in_range": int(in_range.size),
        "out_of_range": int(values.size - in_range.size),
        "inside_padded_fraction": round(float(inside), 4),
        "reliable": bool(dim.kind != "rot" or inside >= 0.90),
        "distribution_shift": bool(dim.kind == "pos" and inside < 0.95),
        "min": round(float(values.min()), 5),
        "max": round(float(values.max()), 5),
        "mean": round(float(values.mean()), 5),
    }


def deficient_intervals(stats: dict, dim: DimSpec) -> list[dict]:
    counts = np.asarray(stats["counts"], dtype=float)
    expected = stats["expected_uniform"]
    edges = stats["bin_edges"]
    flags = counts < MARGINAL_DEFICIT_FACTOR * expected
    intervals = []
    i = 0
    while i < len(counts):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(counts) and flags[j + 1] and (j + 1 - i) < MAX_MERGED_BINS:
            j += 1
        deficit = float(np.sum(np.maximum(0.0, expected - counts[i : j + 1])))
        intervals.append(
            {
                "dim": dim.key,
                "uid": dim.uid,
                "low": edges[i],
                "high": edges[j + 1],
                "bins": [i, j],
                "count": int(counts[i : j + 1].sum()),
                "expected": round(expected * (j + 1 - i), 1),
                "deficit": round(deficit, 1),
            }
        )
        i = j + 1
    return intervals


def grid_stats(xv: np.ndarray, yv: np.ndarray, dx: DimSpec, dy: DimSpec, grid: int):
    x_edges = np.linspace(dx.low, dx.high, grid + 1)
    y_edges = np.linspace(dy.low, dy.high, grid + 1)
    counts, _, _ = np.histogram2d(xv, yv, [x_edges, y_edges])
    mask = (xv >= dx.low) & (xv <= dx.high) & (yv >= dy.low) & (yv <= dy.high)
    expected = float(mask.sum()) / (grid * grid)
    return {
        "x_dim": dx.key,
        "y_dim": dy.key,
        "x_edges": [round(float(v), 5) for v in x_edges],
        "y_edges": [round(float(v), 5) for v in y_edges],
        "counts": counts.astype(int).tolist(),
        "expected_uniform": round(expected, 2),
    }


def deficient_grid_windows(
    grid: dict, window: int = 2, max_windows: int = 2
) -> list[dict]:
    """Top disjoint window x window blocks with the largest coverage deficit.

    Sliding windows keep the recommended regions tight; a connected-component
    bounding box can degenerate to (almost) the full official range when the
    data is strongly concentrated.
    """
    counts = np.asarray(grid["counts"], dtype=float)
    expected = grid["expected_uniform"]
    deficit = np.maximum(0.0, expected - counts)
    n, m = deficit.shape
    window = min(window, n, m)
    used = np.zeros_like(deficit, dtype=bool)
    windows: list[dict] = []
    for _ in range(max_windows):
        best, bi, bj = 0.0, -1, -1
        for i in range(n - window + 1):
            for j in range(m - window + 1):
                if used[i : i + window, j : j + window].any():
                    continue
                total = float(deficit[i : i + window, j : j + window].sum())
                if total > best:
                    best, bi, bj = total, i, j
        if bi < 0 or best <= 0:
            break
        block_count = float(counts[bi : bi + window, bj : bj + window].sum())
        block_expected = expected * window * window
        if block_count >= GRID_DEFICIT_FACTOR * block_expected:
            break
        if windows and best < 0.6 * windows[0]["deficit"]:
            break
        used[bi : bi + window, bj : bj + window] = True
        windows.append(
            {
                "x_dim": grid["x_dim"],
                "y_dim": grid["y_dim"],
                "cells": window * window,
                "x_low": grid["x_edges"][bi],
                "x_high": grid["x_edges"][bi + window],
                "y_low": grid["y_edges"][bj],
                "y_high": grid["y_edges"][bj + window],
                "count": int(block_count),
                "expected": round(block_expected, 1),
                "deficit": round(best, 1),
            }
        )
    return windows


def deficient_grid_components(grid: dict) -> list[dict]:
    counts = np.asarray(grid["counts"], dtype=float)
    expected = grid["expected_uniform"]
    flags = counts < GRID_DEFICIT_FACTOR * expected
    seen = np.zeros_like(flags, dtype=bool)
    comps = []
    n, m = flags.shape
    for si in range(n):
        for sj in range(m):
            if not flags[si, sj] or seen[si, sj]:
                continue
            stack, cells = [(si, sj)], []
            seen[si, sj] = True
            while stack:
                ci, cj = stack.pop()
                cells.append((ci, cj))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < n and 0 <= nj < m and flags[ni, nj] and not seen[ni, nj]:
                        seen[ni, nj] = True
                        stack.append((ni, nj))
            xi = [c[0] for c in cells]
            yj = [c[1] for c in cells]
            i0, i1, j0, j1 = min(xi), max(xi), min(yj), max(yj)
            deficit = float(
                sum(max(0.0, expected - counts[ci, cj]) for ci, cj in cells)
            )
            comps.append(
                {
                    "x_dim": grid["x_dim"],
                    "y_dim": grid["y_dim"],
                    "cells": len(cells),
                    "x_low": grid["x_edges"][i0],
                    "x_high": grid["x_edges"][i1 + 1],
                    "y_low": grid["y_edges"][j0],
                    "y_high": grid["y_edges"][j1 + 1],
                    "count": int(sum(counts[ci, cj] for ci, cj in cells)),
                    "expected": round(expected * len(cells), 1),
                    "deficit": round(deficit, 1),
                }
            )
    return comps


def lowest_bins(dim_stats: dict[str, dict], top: int = 10) -> list[dict]:
    entries = []
    for key, stats in dim_stats.items():
        for i, c in enumerate(stats["counts"]):
            entries.append(
                {
                    "dim": key,
                    "bin": i,
                    "low": stats["bin_edges"][i],
                    "high": stats["bin_edges"][i + 1],
                    "count": int(c),
                    "expected": stats["expected_uniform"],
                }
            )
    entries.sort(key=lambda e: (e["count"], -e["expected"]))
    return entries[:top]


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
def build_gap_recommendations(
    objects: dict[str, ObjectSpec],
    dim_stats: dict[str, dict],
    dim_lookup: dict[str, DimSpec],
    xy_grids: dict[str, dict],
    yaw_pair: dict | None,
    budget: int,
) -> list[dict]:
    candidates = []
    for key, stats in dim_stats.items():
        dim = dim_lookup[key]
        if not stats["reliable"]:
            continue
        for iv in deficient_intervals(stats, dim):
            candidates.append(
                {
                    "kind": "marginal",
                    "deficit": iv["deficit"],
                    "narrow": {dim.uid: {dim.key: (iv["low"], iv["high"])}},
                    "detail": iv,
                }
            )
    for uid, grid in xy_grids.items():
        obj = objects[uid]
        xdim = next(d for d in obj.pos_dims if d.comp == 0)
        ydim = next(d for d in obj.pos_dims if d.comp == 1)
        for comp in deficient_grid_windows(grid):
            candidates.append(
                {
                    "kind": "xy",
                    "deficit": comp["deficit"],
                    "narrow": {
                        uid: {
                            xdim.key: (comp["x_low"], comp["x_high"]),
                            ydim.key: (comp["y_low"], comp["y_high"]),
                        }
                    },
                    "detail": comp,
                }
            )
    if yaw_pair is not None:
        uid_a, uid_b = yaw_pair["uid_a"], yaw_pair["uid_b"]
        dim_a = objects[uid_a].rot_dim
        dim_b = objects[uid_b].rot_dim
        for comp in deficient_grid_windows(yaw_pair["grid"]):
            candidates.append(
                {
                    "kind": "yawpair",
                    "deficit": comp["deficit"],
                    "narrow": {
                        uid_a: {dim_a.key: (comp["x_low"], comp["x_high"])},
                        uid_b: {dim_b.key: (comp["y_low"], comp["y_high"])},
                    },
                    "detail": comp,
                }
            )

    candidates.sort(key=lambda c: c["deficit"], reverse=True)
    picked: list[dict] = []
    for cand in candidates:
        if len(picked) >= MAX_GAP_RECS:
            break
        if cand["deficit"] <= 0:
            continue
        redundant = False
        for prev in picked:
            shared = set()
            for uid, dims in cand["narrow"].items():
                for key, (lo, hi) in dims.items():
                    if uid in prev["narrow"] and key in prev["narrow"][uid]:
                        plo, phi = prev["narrow"][uid][key]
                        overlap = min(hi, phi) - max(lo, plo)
                        if overlap > 0.5 * min(hi - lo, phi - plo):
                            shared.add(key)
            prev_keys = {k for dims in prev["narrow"].values() for k in dims}
            cand_keys = {k for dims in cand["narrow"].values() for k in dims}
            if shared and shared == prev_keys == cand_keys:
                redundant = True
                break
        if not redundant:
            picked.append(cand)
    if not picked:
        return [], budget

    episodes = allocate_episodes(
        [c["deficit"] for c in picked], budget, minimum=GAP_MIN_EPISODES, step=EPISODE_STEP
    )
    # Cap each recommendation at ~3x its deficit so a single small gap cannot
    # swallow the whole gap budget; the surplus flows to the stratified part.
    episodes = [
        min(
            eps,
            max(
                GAP_MIN_EPISODES,
                int(round(3.0 * cand["deficit"] / EPISODE_STEP)) * EPISODE_STEP,
            ),
        )
        for cand, eps in zip(picked, episodes)
    ]
    recs = []
    used_names = set()
    for cand, eps in zip(picked, episodes):
        detail = cand["detail"]
        if cand["kind"] == "marginal":
            dim = dim_lookup[detail["dim"]]
            band = band_label(detail["low"], detail["high"], dim.low, dim.high)
            name = f"gap_{detail['dim']}_{band}"
            reason = (
                f"{detail['dim']} in [{detail['low']:.3f}, {detail['high']:.3f}] holds "
                f"{detail['count']} kept episodes vs {detail['expected']} expected "
                f"(8-bin uniform)"
            )
        elif cand["kind"] == "xy":
            uid = next(iter(cand["narrow"]))
            xdim = dim_lookup[detail["x_dim"]]
            ydim = dim_lookup[detail["y_dim"]]
            xband = band_label(detail["x_low"], detail["x_high"], xdim.low, xdim.high)
            yband = band_label(detail["y_low"], detail["y_high"], ydim.low, ydim.high)
            name = f"gap_{uid}_xy_x{xband}_y{yband}"
            reason = (
                f"{uid} xy block x[{detail['x_low']:.3f},{detail['x_high']:.3f}] "
                f"y[{detail['y_low']:.3f},{detail['y_high']:.3f}] holds "
                f"{detail['count']} kept episodes vs {detail['expected']} expected "
                f"({detail['cells']} grid cells)"
            )
        else:
            uids = list(cand["narrow"].keys())
            xdim = dim_lookup[detail["x_dim"]]
            ydim = dim_lookup[detail["y_dim"]]
            aband = band_label(detail["x_low"], detail["x_high"], xdim.low, xdim.high)
            bband = band_label(detail["y_low"], detail["y_high"], ydim.low, ydim.high)
            name = f"gap_yawpair_{uids[0]}{aband}_{uids[1]}{bband}"
            reason = (
                f"yaw pair {detail['x_dim']}[{detail['x_low']:.1f},{detail['x_high']:.1f}] x "
                f"{detail['y_dim']}[{detail['y_low']:.1f},{detail['y_high']:.1f}] holds "
                f"{detail['count']} kept episodes vs {detail['expected']} expected"
            )
        while name in used_names:
            name += "_b"
        used_names.add(name)
        shifted = sorted(
            {
                key
                for dims in cand["narrow"].values()
                for key in dims
                if dim_stats.get(key, {}).get("distribution_shift")
            }
        )
        if shifted:
            reason += (
                f"; note: {'/'.join(shifted)} of the expert data is shifted vs the "
                "current official range, so this region lacks on-support data"
            )
        overrides = {}
        for uid, dims in cand["narrow"].items():
            overrides[uid] = narrowed_ranges(
                objects[uid], {k: v for k, v in dims.items()}
            )
        recs.append(
            {
                "name": name,
                "kind": cand["kind"],
                "episodes": int(eps),
                "reason": reason,
                "source_stats": detail,
                "overrides": overrides,
            }
        )
    return recs, budget - sum(r["episodes"] for r in recs)


def make_block_grid(dims: list[DimSpec], splits: list[int]):
    """Cartesian product of per-dim equal splits -> list of {key: (lo,hi), idx}."""
    per_dim = []
    for dim, k in zip(dims, splits):
        edges = np.linspace(dim.low, dim.high, k + 1)
        per_dim.append([(float(edges[i]), float(edges[i + 1])) for i in range(k)])
    blocks = [({}, [])]
    for dim, intervals in zip(dims, per_dim):
        blocks = [
            ({**narrow, dim.key: iv}, idx + [i])
            for narrow, idx in blocks
            for i, iv in enumerate(intervals)
        ]
    return blocks


def build_strat_recommendations(
    objects: dict[str, ObjectSpec],
    targets: list[tuple[str, list[int]]],
    budget: int,
    reason_prefix: str,
) -> list[dict]:
    """targets: list of (uid, splits) stratified jointly via cross product."""
    joint = [({}, "")]
    for uid, splits in targets:
        obj = objects[uid]
        dims = obj.pos_dims[:2]
        if not dims:
            continue
        use = splits[: len(dims)]
        blocks = make_block_grid(dims, use)
        joint = [
            ({**narrow, **blk}, f"{tag}_{uid}{''.join(map(str, idx))}".strip("_"))
            for narrow, tag in joint
            for blk, idx in blocks
        ]
    if len(joint) <= 1:
        return []
    episodes = allocate_episodes([1.0] * len(joint), budget, minimum=0, step=1)
    recs = []
    for (narrow, tag), eps in zip(joint, episodes):
        per_uid: dict[str, dict] = {}
        for key, iv in narrow.items():
            uid = next(u for u, o in objects.items() if any(d.key == key for d in o.dims))
            per_uid.setdefault(uid, {})[key] = iv
        overrides = {
            uid: narrowed_ranges(objects[uid], dims) for uid, dims in per_uid.items()
        }
        parts = [
            f"{key}[{iv[0]:.3f},{iv[1]:.3f}]" for key, iv in sorted(narrow.items())
        ]
        recs.append(
            {
                "name": f"strat_{tag}",
                "kind": "strat",
                "episodes": int(eps),
                "reason": f"{reason_prefix}: {' '.join(parts)}",
                "source_stats": {"block": tag},
                "overrides": overrides,
            }
        )
    return recs


# --------------------------------------------------------------------------
# Builder hints (pair-constrained vs keep-original)
# --------------------------------------------------------------------------
def floor_to(value: float, step: float) -> float:
    return math.floor(value / step) * step


def decide_builder_hints(
    config: dict,
    objects: dict[str, ObjectSpec],
    ordered_uids: list[str],
    clearances: np.ndarray | None,
    mc_samples: int,
) -> dict:
    hints: dict = {"mode": "keep", "keep_events": [], "pair": None, "notes": []}
    hints["keep_events"] = [objects[u].event_name for u in ordered_uids]
    if len(ordered_uids) != 2:
        hints["notes"].append("not a two-object task; keep original events")
        return hints
    a, b = ordered_uids
    pairable = all(
        objects[u].event.get("func") == "randomize_rigid_object_pose"
        and objects[u].section == "rigid_object"
        and objects[u].half_extents is not None
        for u in (a, b)
    )
    if not pairable:
        hints["notes"].append(
            "at least one object is an articulation/group event or has no mesh; "
            "keeping original events"
        )
        return hints

    spec_a = event_sampling_spec(objects[a].event, objects[a].init_pos, objects[a].init_rot)
    spec_b = event_sampling_spec(objects[b].event, objects[b].init_pos, objects[b].init_rot)
    half_a = np.asarray(objects[a].half_extents)
    half_b = np.asarray(objects[b].half_extents)

    min_clear = 0.03
    if clearances is not None and clearances.size:
        p02 = float(np.percentile(clearances, 2))
        min_clear = min(0.03, max(0.005, floor_to(p02, 0.005)))
        violate = float(np.mean(clearances < min_clear))
        hints["notes"].append(
            f"observed OBB clearance p02={p02:.4f}; chose min_xy_clearance={min_clear}"
        )
        if violate > 0.10:
            hints["notes"].append(
                f"{violate:.0%} of kept episodes violate clearance {min_clear}; "
                "keeping original events to avoid cutting official support"
            )
            return hints
    acceptance = mc_pair_acceptance(
        spec_a, spec_b, half_a, half_b, min_clear, None, samples=mc_samples
    )
    if clearances is None:
        if acceptance < 0.97:
            retry = mc_pair_acceptance(
                spec_a, spec_b, half_a, half_b, 0.01, None, samples=mc_samples
            )
            if retry >= 0.97:
                min_clear, acceptance = 0.01, retry
                hints["notes"].append(
                    "no data; clearance relaxed to 0.01 to keep official support"
                )
            else:
                hints["notes"].append(
                    f"no data and official ranges violate the pair constraint too "
                    f"often (acceptance {acceptance:.2%} @0.03, {retry:.2%} @0.01); "
                    "keeping original events"
                )
                return hints
    hints["mode"] = "pair"
    hints["pair"] = {
        "first_uid": a,
        "second_uid": b,
        "first_event": objects[a].event_name,
        "second_event": objects[b].event_name,
        "first_half_extents": objects[a].half_extents,
        "second_half_extents": objects[b].half_extents,
        "min_xy_clearance": round(min_clear, 3),
        "mc_acceptance_official_ranges": round(acceptance, 4),
    }
    return hints


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_marginals(path: Path, dim_stats: dict, extra_hists: dict):
    panels = list(dim_stats.items()) + list(extra_hists.items())
    if not panels:
        return
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, (key, stats) in zip(axes, panels):
        edges = np.asarray(stats["bin_edges"], dtype=float)
        counts = np.asarray(stats["counts"], dtype=float)
        axis.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color="tab:blue",
            alpha=0.75,
            edgecolor="white",
        )
        expected = stats.get("expected_uniform")
        if expected:
            axis.axhline(expected, color="tab:orange", linestyle="--", linewidth=1.2)
        axis.set_title(key, fontsize=10)
    for axis in axes[len(panels):]:
        axis.axis("off")
    fig.suptitle("kept-episode coverage vs uniform expectation (dashed)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grids(path: Path, xy_grids: dict, yaw_pair: dict | None):
    panels = [(f"{uid} xy", g) for uid, g in xy_grids.items()]
    if yaw_pair is not None:
        panels.append((f"yaw {yaw_pair['uid_a']} x {yaw_pair['uid_b']}", yaw_pair["grid"]))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4))
    axes = np.atleast_1d(axes).ravel()
    for axis, (title, grid) in zip(axes, panels):
        counts = np.asarray(grid["counts"], dtype=float)
        im = axis.imshow(
            counts.T,
            origin="lower",
            aspect="auto",
            extent=(
                grid["x_edges"][0],
                grid["x_edges"][-1],
                grid["y_edges"][0],
                grid["y_edges"][-1],
            ),
            cmap="viridis",
        )
        axis.set_title(f"{title} (expected {grid['expected_uniform']}/cell)", fontsize=10)
        axis.set_xlabel(grid["x_dim"])
        axis.set_ylabel(grid["y_dim"])
        fig.colorbar(im, ax=axis, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--pose-cols",
        default=None,
        help="Comma separated pose columns, e.g. duck_pose,drawer_pose",
    )
    parser.add_argument("--uids", default=None, help="Override object uids (comma sep)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--yaw-grid", type=int, default=4)
    parser.add_argument("--episode-budget", type=int, default=500)
    parser.add_argument("--gap-fraction", type=float, default=0.7)
    parser.add_argument("--stratify-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap parquet scan")
    parser.add_argument("--use-cache", action="store_true", help="Reuse episodes.csv")
    parser.add_argument("--mc-samples", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = args.task
    config_path = args.config or (REPO_ROOT / "configs" / task / "random" / "gym_config.json")
    out_dir = args.out or (DEFAULT_REPORT_ROOT / task)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(config_path)

    stratify_only = args.stratify_only or (
        task in STRATIFY_ONLY_TASKS and args.pose_cols is None
    )
    if args.uids:
        uids = [u.strip() for u in args.uids.split(",") if u.strip()]
        pose_cols = [f"{u}_pose" for u in uids]
    elif stratify_only:
        uids = STRATIFY_TASK_UIDS.get(task)
        if uids is None:
            raise SystemExit(f"--stratify-only for {task} requires --uids")
        pose_cols = []
    else:
        pose_cols = [
            c.strip()
            for c in (args.pose_cols or ",".join(TASK_POSE_COLUMNS[task])).split(",")
            if c.strip()
        ]
        uids = [c[:-5] if c.endswith("_pose") else c for c in pose_cols]

    objects = {uid: ObjectSpec(config, uid) for uid in uids}
    dim_lookup = {d.key: d for obj in objects.values() for d in obj.dims}
    summary: dict = {
        "task": task,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "stratify_only": bool(stratify_only),
        "episode_budget": int(args.episode_budget),
        "objects": {uid: obj.describe() for uid, obj in objects.items()},
        "notes": [],
    }

    if stratify_only:
        summary["dataset"] = None
        summary["n_episodes"] = None
        # fine 8-way per-dim listing (informational)
        fine = {}
        for obj in objects.values():
            for dim in obj.dims:
                edges = np.linspace(dim.low, dim.high, args.bins + 1)
                fine[dim.key] = {
                    "blocks": [
                        [round(float(edges[i]), 5), round(float(edges[i + 1]), 5)]
                        for i in range(args.bins)
                    ],
                    "episodes_per_block": round(args.episode_budget / args.bins, 1),
                }
        summary["stratification_blocks_fine"] = fine
        if len(uids) == 1:
            targets = [(uids[0], [3, 3])]
        else:
            targets = [(uid, [2, 2]) for uid in uids]
        recs = build_strat_recommendations(
            objects,
            targets,
            args.episode_budget,
            "stratified block over the official random range (no pose columns in "
            "the dataset)",
        )
        summary["recommendations"] = recs
        hints = decide_builder_hints(config, objects, uids, None, args.mc_samples)
        summary["builder_hints"] = hints
        summary["gap_budget"] = 0
        summary["strat_budget"] = int(args.episode_budget)
    else:
        dataset_dir = resolve_dataset_dir(args.dataset_root, task)
        summary["dataset"] = str(dataset_dir)
        cache_path = out_dir / "episodes.csv"
        dim_values: dict[str, np.ndarray] = {}
        per_episode: list[dict] = []
        if args.use_cache and cache_path.exists():
            with cache_path.open() as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    per_episode.append(
                        {
                            k: (float(v) if v not in ("", "None") else None)
                            for k, v in row.items()
                        }
                    )
            summary["notes"].append("loaded per-episode values from episodes.csv cache")
            for uid, obj in objects.items():
                obj.observed = any(
                    row.get(f"{obj.dims[0].key}") is not None
                    for row in per_episode
                ) if obj.dims else False
        else:
            rows = scan_first_frames(dataset_dir, pose_cols, args.limit)
            zero_frac = {
                col: np.mean([r["poses"][col] is None for r in rows])
                for col in pose_cols
            }
            for uid, col in zip(uids, pose_cols):
                objects[uid].observed = zero_frac[col] < 0.2
                if not objects[uid].observed:
                    summary["notes"].append(
                        f"pose column {col} is empty/zero in "
                        f"{zero_frac[col]:.0%} of episodes; {uid} treated as "
                        "unobserved (stratified replenishment instead)"
                    )
            observed_uids = [u for u in uids if objects[u].observed]
            for row in rows:
                rec = {"episode_index": row["episode_index"]}
                for uid, col in zip(uids, pose_cols):
                    pose = row["poses"][col]
                    if pose is None or not objects[uid].observed:
                        for dim in objects[uid].dims:
                            rec[dim.key] = None
                        continue
                    for dim in objects[uid].dims:
                        rec[dim.key] = dim_value(
                            dim, pose, objects[uid].init_pos, objects[uid].init_rot
                        )
                    rec[f"{uid}_z"] = float(pose[2, 3])
                if len(observed_uids) == 2:
                    pa = row["poses"][pose_cols[uids.index(observed_uids[0])]]
                    pb = row["poses"][pose_cols[uids.index(observed_uids[1])]]
                    if pa is not None and pb is not None:
                        rec["center_distance"] = float(
                            np.linalg.norm(pa[:2, 3] - pb[:2, 3])
                        )
                        rec["bearing_deg"] = float(
                            np.degrees(
                                np.arctan2(pb[1, 3] - pa[1, 3], pb[0, 3] - pa[0, 3])
                            )
                        )
                        ha = objects[observed_uids[0]].half_extents
                        hb = objects[observed_uids[1]].half_extents
                        if ha is not None and hb is not None:
                            rec["obb_clearance"] = float(
                                sat_clearance_2d(
                                    pa[None, ...], pb[None, ...], ha, hb
                                )[0]
                            )
                per_episode.append(rec)
            fields = sorted({k for r in per_episode for k in r})
            fields.remove("episode_index")
            fields = ["episode_index"] + fields
            with cache_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(per_episode)

        n_episodes = len(per_episode)
        summary["n_episodes"] = n_episodes
        for key, dim in dim_lookup.items():
            vals = np.asarray(
                [r[key] for r in per_episode if r.get(key) is not None], dtype=float
            )
            if vals.size:
                dim_values[key] = vals
        for uid, obj in objects.items():
            obj.observed = any(d.key in dim_values for d in obj.dims)

        dim_stats = {
            key: marginal_stats(vals, dim_lookup[key], args.bins)
            for key, vals in dim_values.items()
        }
        xy_grids = {}
        for uid, obj in objects.items():
            if not obj.observed:
                continue
            xdims = [d for d in obj.pos_dims if d.comp == 0 and d.key in dim_values]
            ydims = [d for d in obj.pos_dims if d.comp == 1 and d.key in dim_values]
            if xdims and ydims:
                xy_grids[uid] = grid_stats(
                    dim_values[xdims[0].key],
                    dim_values[ydims[0].key],
                    xdims[0],
                    ydims[0],
                    args.grid,
                )
        yaw_pair = None
        rot_objs = [
            uid
            for uid in uids
            if objects[uid].observed
            and objects[uid].rot_dim is not None
            and objects[uid].rot_dim.key in dim_stats
            and dim_stats[objects[uid].rot_dim.key]["reliable"]
        ]
        if len(rot_objs) == 2:
            da = objects[rot_objs[0]].rot_dim
            db = objects[rot_objs[1]].rot_dim
            mask = np.asarray(
                [
                    (r.get(da.key) is not None) and (r.get(db.key) is not None)
                    for r in per_episode
                ]
            )
            va = np.asarray([r[da.key] for r in per_episode if r.get(da.key) is not None and r.get(db.key) is not None])
            vb = np.asarray([r[db.key] for r in per_episode if r.get(da.key) is not None and r.get(db.key) is not None])
            yaw_pair = {
                "uid_a": rot_objs[0],
                "uid_b": rot_objs[1],
                "grid": grid_stats(va, vb, da, db, args.yaw_grid),
            }

        rel = {}
        for key in ("center_distance", "bearing_deg", "obb_clearance"):
            vals = np.asarray(
                [r[key] for r in per_episode if r.get(key) is not None], dtype=float
            )
            if vals.size:
                edges = np.linspace(vals.min(), vals.max(), args.bins + 1)
                counts, _ = np.histogram(vals, edges)
                rel[key] = {
                    "bin_edges": [round(float(v), 5) for v in edges],
                    "counts": [int(c) for c in counts],
                    "expected_uniform": round(vals.size / args.bins, 2),
                    "min": round(float(vals.min()), 5),
                    "max": round(float(vals.max()), 5),
                    "mean": round(float(vals.mean()), 5),
                    "p02": round(float(np.percentile(vals, 2)), 5),
                    "p50": round(float(np.percentile(vals, 50)), 5),
                }
        summary["relative_metrics"] = rel or None
        summary["dim_stats"] = dim_stats
        summary["xy_grids"] = xy_grids
        summary["yaw_pair_grid"] = yaw_pair
        summary["lowest_density_bins"] = lowest_bins(dim_stats)

        gap_budget = int(round(args.episode_budget * args.gap_fraction))
        strat_budget = args.episode_budget - gap_budget
        gap_recs, gap_leftover = build_gap_recommendations(
            objects, dim_stats, dim_lookup, xy_grids, yaw_pair, gap_budget
        )
        if not gap_recs:
            summary["notes"].append(
                "no significant marginal/2D gaps found; whole budget goes to "
                "stratified replenishment"
            )
        elif gap_leftover > 0:
            summary["notes"].append(
                f"gap recommendations capped at ~3x their deficit; {gap_leftover} "
                "episodes moved to the stratified part"
            )
        strat_budget += gap_leftover
        gap_budget -= gap_leftover
        unobserved = [u for u in uids if not objects[u].observed and objects[u].pos_dims]
        if unobserved:
            targets = [(unobserved[0], [2, 2])]
            strat_reason = (
                f"stratified block over the official range of unobserved object "
                f"{unobserved[0]} (pose column always zero in the dataset)"
            )
        else:
            def xy_area(uid: str) -> float:
                dims = objects[uid].pos_dims
                area = 1.0
                for d in dims[:2]:
                    area *= max(d.high - d.low, 1e-9)
                return area

            primary = max(
                [u for u in uids if objects[u].observed], key=xy_area, default=None
            )
            targets = [(primary, [2, 2])] if primary else []
            strat_reason = (
                f"stratified xy quadrant of {primary} for uniform replenishment "
                "over the full official range"
            )
        strat_recs = (
            build_strat_recommendations(objects, targets, strat_budget, strat_reason)
            if targets and strat_budget > 0
            else []
        )
        summary["gap_budget"] = gap_budget
        summary["strat_budget"] = strat_budget
        summary["recommendations"] = gap_recs + strat_recs

        clearances = np.asarray(
            [r["obb_clearance"] for r in per_episode if r.get("obb_clearance") is not None],
            dtype=float,
        )
        hints = decide_builder_hints(
            config,
            objects,
            uids,
            clearances if clearances.size else None,
            args.mc_samples,
        )
        summary["builder_hints"] = hints

        plot_marginals(out_dir / "marginals.png", dim_stats, rel)
        plot_grids(out_dir / "grids_2d.png", xy_grids, yaw_pair)

    total = sum(r["episodes"] for r in summary["recommendations"])
    summary["recommended_total_episodes"] = int(total)
    (out_dir / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"[{task}] wrote {out_dir / 'coverage_summary.json'}")
    for rec in summary["recommendations"]:
        print(f"  {rec['name']:<44s} {rec['episodes']:>4d}  {rec['kind']}")
    print(f"  total recommended episodes: {total}")
    if summary["notes"]:
        for note in summary["notes"]:
            print(f"  note: {note}")


if __name__ == "__main__":
    main()
