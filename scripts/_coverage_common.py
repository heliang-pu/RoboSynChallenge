#!/usr/bin/env python3
"""Shared helpers for the generic randomization-coverage audit tooling.

Used by scripts/analyze_random_coverage.py and scripts/build_coverage_configs.py.
All conventions mirror the official randomization implementation:

- ``matrix_from_euler`` is intrinsic XYZ, i.e. ``R = Rx(a) @ Ry(b) @ Rz(c)``.
- ``relative_rotation=True`` composes ``R_world = R_init @ R_sample``.
- ``relative_position=True`` adds the sampled offset to ``init_pos``.
- Official defaults (randomize_rigid_object_pose / randomize_articulation_root_pose
  / randomize_entity_root_pose_group): relative_position=True, relative_rotation
  defaults False for the single-object funcs and True for the group func; every
  task config we handle sets the flags explicitly, so the defaults are fallback
  only.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# 本地 NAS 路径因机器而异，用环境变量覆盖。
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "ROBOSYN_DATASET_ROOT",
        str(Path.home() / "FermiBotNas/dataset/RoboSynChallenge/Sim_clean_filtered_pruned"),
    )
)
DEFAULT_SAVE_ROOT = Path(
    os.environ.get(
        "ROBOSYN_SAVE_ROOT",
        str(Path.home() / "FermiBotNas/dataset/RoboSynChallenge/Syn"),
    )
)
DEFAULT_REPORT_ROOT = REPO_ROOT / "report" / "coverage"

# Default first-frame pose columns per task (Sim_clean_filtered_pruned v2.1).
TASK_POSE_COLUMNS = {
    "drawer_open_place": ["duck_pose", "drawer_pose"],
    "item_assembly": ["guijiao1_pose", "guijiao2_pose"],
    "items_handover": ["pen_pose", "holder_pose"],
    "manipulate_pipette": ["beaker1_pose", "pipette_pose"],
    "mixer_operating": ["beaker_pose", "beaker_mixer_pose"],
    "water_pouring": ["bottle_pose", "cup_pose"],
    "sample_loading": ["cube_pose", "rack_pose"],
}

# Tasks whose datasets carry no object pose columns -> stratify from config only.
STRATIFY_ONLY_TASKS = {"click_bell", "handle_basket", "table_rearrangement"}
STRATIFY_TASK_UIDS = {
    "click_bell": ["button"],
    "handle_basket": ["milk", "basket"],
    "table_rearrangement": ["fork", "spoon"],
}

# Event functions that place the key objects.
POSE_EVENT_FUNCS = {
    "randomize_rigid_object_pose",
    "randomize_articulation_root_pose",
    "randomize_entity_root_pose_group",
}

PAIR_EVENT_FUNC = "randomize_rigid_object_pair_pose_constrained"


# --------------------------------------------------------------------------
# Rotation helpers (numpy mirrors of embodichain.utils.math)
# --------------------------------------------------------------------------
def _axis_rot(axis: int, angle_rad: np.ndarray) -> np.ndarray:
    """Batched single-axis rotation matrices; angle_rad shape (N,)."""
    angle_rad = np.atleast_1d(np.asarray(angle_rad, dtype=np.float64))
    n = angle_rad.shape[0]
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    out = np.tile(np.eye(3), (n, 1, 1))
    if axis == 0:
        out[:, 1, 1], out[:, 1, 2] = c, -s
        out[:, 2, 1], out[:, 2, 2] = s, c
    elif axis == 1:
        out[:, 0, 0], out[:, 0, 2] = c, s
        out[:, 2, 0], out[:, 2, 2] = -s, c
    else:
        out[:, 0, 0], out[:, 0, 1] = c, -s
        out[:, 1, 0], out[:, 1, 1] = s, c
    return out


def matrix_from_euler_deg(euler_deg) -> np.ndarray:
    """Intrinsic XYZ euler (degrees) to rotation matrices, batched or single."""
    e = np.asarray(euler_deg, dtype=np.float64)
    single = e.ndim == 1
    e = np.atleast_2d(e)
    rad = np.radians(e)
    mats = _axis_rot(0, rad[:, 0]) @ _axis_rot(1, rad[:, 1]) @ _axis_rot(2, rad[:, 2])
    return mats[0] if single else mats


def extract_component_angle_deg(rot_delta: np.ndarray, comp: int) -> float:
    """Angle of a (near) pure single-axis rotation about x/y/z (comp 0/1/2)."""
    r = rot_delta
    if comp == 0:
        return float(np.degrees(np.arctan2(r[2, 1], r[1, 1])))
    if comp == 1:
        return float(np.degrees(np.arctan2(r[0, 2], r[0, 0])))
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


def wrap_deg(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def pose_matrix_from_value(value) -> np.ndarray:
    """Parquet pose cell -> 4x4 matrix. Supports (4,4), flat 16, xyz+quat(7)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    flat = arr.reshape(-1)
    if flat.size == 16:
        return flat.reshape(4, 4)
    if flat.size == 7:
        xyz, quat = flat[:3], flat[3:]
        # heuristic: wxyz if first element has the largest magnitude on average
        w, x, y, z = (
            (quat[0], quat[1], quat[2], quat[3])
            if abs(quat[0]) >= abs(quat[3])
            else (quat[3], quat[0], quat[1], quat[2])
        )
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = xyz
        return pose
    raise ValueError(f"Unsupported pose payload with {flat.size} values")


def is_zero_pose(pose: np.ndarray) -> bool:
    return bool(np.abs(pose).sum() < 1e-9)


# --------------------------------------------------------------------------
# Config introspection
# --------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def config_events(config: dict) -> dict:
    return config["env"]["events"]


def event_entity_uids(event: dict) -> list[str]:
    params = event.get("params", {})
    uids: list[str] = []
    entity_cfg = params.get("entity_cfg")
    if isinstance(entity_cfg, dict) and entity_cfg.get("uid"):
        uids.append(entity_cfg["uid"])
    for cfg in params.get("entity_cfgs") or []:
        if isinstance(cfg, dict) and cfg.get("uid"):
            uids.append(cfg["uid"])
    for value in params.get("entity_uids") or []:
        if isinstance(value, str):
            uids.append(value)
    return uids


def find_pose_event(config: dict, uid: str) -> tuple[str, dict]:
    """Locate the randomization event that places ``uid``."""
    for name, event in config_events(config).items():
        if event.get("func") in POSE_EVENT_FUNCS and uid in event_entity_uids(event):
            return name, event
    raise KeyError(f"No pose randomization event found for uid {uid!r}")


def entity_entry(config: dict, uid: str) -> tuple[str, dict]:
    for section in ("rigid_object", "articulation", "background"):
        entries = config.get(section, [])
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if entry.get("uid") == uid:
                return section, entry
    raise KeyError(f"Entity {uid!r} not found in config")


def entity_init(config: dict, uid: str) -> tuple[np.ndarray, np.ndarray, str]:
    section, entry = entity_entry(config, uid)
    init_pos = np.asarray(entry.get("init_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    init_rot = matrix_from_euler_deg(entry.get("init_rot", [0.0, 0.0, 0.0]))
    return init_pos, init_rot, section


@dataclass
class DimSpec:
    """A single randomized scalar dimension expressed in config units."""

    key: str  # e.g. "duck_x", "drawer_dx", "pen_rotz"
    uid: str
    kind: str  # "pos" | "rot"
    comp: int  # 0/1/2 (x/y/z axis or euler component)
    low: float
    high: float
    relative: bool

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "uid": self.uid,
            "kind": self.kind,
            "comp": self.comp,
            "low": self.low,
            "high": self.high,
            "relative": self.relative,
        }


def _default_relative(event: dict, which: str) -> bool:
    func = event.get("func", "")
    if which == "position":
        return True
    return func == "randomize_entity_root_pose_group"


def dims_from_event(
    uid: str,
    event: dict,
    min_pos_width: float = 1e-6,
    min_rot_width_deg: float = 1.0,
) -> list[DimSpec]:
    """Extract the randomized x/y position dims and the dominant euler dim."""
    params = event.get("params", {})
    dims: list[DimSpec] = []
    position_range = params.get("position_range")
    relative_position = bool(
        params.get("relative_position", _default_relative(event, "position"))
    )
    if position_range:
        for comp, axis in enumerate("xy"):
            low = float(position_range[0][comp])
            high = float(position_range[1][comp])
            if high - low > min_pos_width:
                key = f"{uid}_d{axis}" if relative_position else f"{uid}_{axis}"
                dims.append(
                    DimSpec(key, uid, "pos", comp, low, high, relative_position)
                )
    rotation_range = params.get("rotation_range")
    relative_rotation = bool(
        params.get("relative_rotation", _default_relative(event, "rotation"))
    )
    if rotation_range:
        widths = [
            float(rotation_range[1][i]) - float(rotation_range[0][i]) for i in range(3)
        ]
        comp = int(np.argmax(widths))
        if widths[comp] > min_rot_width_deg:
            axis = "xyz"[comp]
            dims.append(
                DimSpec(
                    f"{uid}_rot{axis}",
                    uid,
                    "rot",
                    comp,
                    float(rotation_range[0][comp]),
                    float(rotation_range[1][comp]),
                    relative_rotation,
                )
            )
    return dims


def dim_value(
    dim: DimSpec, pose: np.ndarray, init_pos: np.ndarray, init_rot: np.ndarray
) -> float:
    """Recover the sampled randomization value (config units) from a world pose."""
    if dim.kind == "pos":
        value = float(pose[dim.comp, 3])
        return value - float(init_pos[dim.comp]) if dim.relative else value
    rot = pose[:3, :3]
    rot_delta = init_rot.T @ rot if dim.relative else rot
    return extract_component_angle_deg(rot_delta, dim.comp)


# --------------------------------------------------------------------------
# Assets / half extents
# --------------------------------------------------------------------------
def resolve_asset_path(fpath: str) -> Path | None:
    candidates = [REPO_ROOT / fpath, Path(fpath)]
    for cand in candidates:
        if cand.exists():
            return cand
    try:
        from embodichain.data import get_data_path

        cand = Path(get_data_path(fpath))
        if cand.exists():
            return cand
    except Exception:
        pass
    name = Path(fpath).name
    for base in (REPO_ROOT / "robosynchallenge", REPO_ROOT / "assets"):
        hits = sorted(base.glob(f"**/{name}"))
        if hits:
            return hits[0]
    return None


def mesh_scaled_vertices(entry: dict) -> tuple[np.ndarray | None, str | None]:
    """Local-frame vertices of an entity mesh, scaled by body_scale."""
    shape = entry.get("shape", {})
    fpath = shape.get("fpath")
    if not fpath:
        return None, None
    resolved = resolve_asset_path(fpath)
    if resolved is None:
        return None, None
    try:
        import trimesh

        mesh = trimesh.load(resolved, process=False, force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
    except Exception:
        return None, str(resolved)
    scale = np.asarray(entry.get("body_scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    return vertices * scale, str(resolved)


def half_extents_from_vertices(vertices: np.ndarray, inflate: float = 1.05) -> list:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extents = np.maximum(np.abs(low), np.abs(high)) * inflate
    return [round(float(v), 5) for v in extents]


# --------------------------------------------------------------------------
# 2D separating-axis clearance (numpy port of _projected_obb_separation_2d)
# --------------------------------------------------------------------------
def sat_clearance_2d(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    half_a: np.ndarray,
    half_b: np.ndarray,
) -> np.ndarray:
    half_a = np.asarray(half_a, dtype=np.float64)
    half_b = np.asarray(half_b, dtype=np.float64)
    rot_a, rot_b = pose_a[:, :3, :3], pose_b[:, :3, :3]
    gen_a = rot_a[:, :2, :] * half_a[None, None, :]
    gen_b = rot_b[:, :2, :] * half_b[None, None, :]
    gens = np.concatenate([gen_a, gen_b], axis=2)
    axes = np.stack([-gens[:, 1, :], gens[:, 0, :]], axis=-1)
    norm = np.linalg.norm(axes, axis=-1, keepdims=True)
    valid = norm[..., 0] > 1e-7
    axes = axes / np.clip(norm, 1e-7, None)
    radius_a = (
        np.abs(np.einsum("bai,bij->baj", axes, rot_a[:, :2, :])) * half_a[None, None, :]
    ).sum(-1)
    radius_b = (
        np.abs(np.einsum("bai,bij->baj", axes, rot_b[:, :2, :])) * half_b[None, None, :]
    ).sum(-1)
    delta = pose_b[:, :2, 3] - pose_a[:, :2, 3]
    projection = np.abs(np.einsum("bai,bi->ba", axes, delta))
    gaps = projection - radius_a - radius_b
    gaps = np.where(valid, gaps, -np.inf)
    return gaps.max(axis=1)


def sample_event_poses(
    rng: np.random.Generator,
    count: int,
    position_range,
    rotation_range,
    relative_position: bool,
    relative_rotation: bool,
    init_pos: np.ndarray,
    init_rot: np.ndarray,
) -> np.ndarray:
    """Offline mirror of get_random_pose / _sample_rigid_pose_from_ranges."""
    poses = np.tile(np.eye(4), (count, 1, 1))
    if position_range:
        low = np.asarray(position_range[0], dtype=np.float64)
        high = np.asarray(position_range[1], dtype=np.float64)
        pos = rng.uniform(size=(count, 3)) * (high - low) + low
        if relative_position:
            pos = pos + init_pos[None, :]
        poses[:, :3, 3] = pos
    else:
        poses[:, :3, 3] = init_pos[None, :]
    if rotation_range:
        low = np.asarray(rotation_range[0], dtype=np.float64)
        high = np.asarray(rotation_range[1], dtype=np.float64)
        euler = rng.uniform(size=(count, 3)) * (high - low) + low
        rot = matrix_from_euler_deg(euler)
        if relative_rotation:
            rot = init_rot[None, :, :] @ rot
        poses[:, :3, :3] = rot
    else:
        poses[:, :3, :3] = init_rot[None, :, :]
    return poses


def mc_pair_acceptance(
    first_spec: dict,
    second_spec: dict,
    half_a,
    half_b,
    min_xy_clearance: float,
    max_xy_center_distance: float | None = None,
    samples: int = 20000,
    seed: int = 0,
) -> float:
    """Fraction of joint samples that satisfy the pair constraint.

    Each spec: {position_range, rotation_range, relative_position,
    relative_rotation, init_pos, init_rot}.
    """
    rng = np.random.default_rng(seed)
    pose_a = sample_event_poses(rng, samples, **first_spec)
    pose_b = sample_event_poses(rng, samples, **second_spec)
    clear = sat_clearance_2d(pose_a, pose_b, half_a, half_b)
    ok = clear >= float(min_xy_clearance)
    if max_xy_center_distance is not None:
        dist = np.linalg.norm(pose_a[:, :2, 3] - pose_b[:, :2, 3], axis=-1)
        ok &= dist <= float(max_xy_center_distance)
    return float(np.mean(ok))


def event_sampling_spec(event: dict, init_pos: np.ndarray, init_rot: np.ndarray) -> dict:
    params = event.get("params", {})
    return {
        "position_range": params.get("position_range"),
        "rotation_range": params.get("rotation_range"),
        "relative_position": bool(
            params.get("relative_position", _default_relative(event, "position"))
        ),
        "relative_rotation": bool(
            params.get("relative_rotation", _default_relative(event, "rotation"))
        ),
        "init_pos": init_pos,
        "init_rot": init_rot,
    }


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def band_label(low: float, high: float, full_low: float, full_high: float) -> str:
    span = full_high - full_low
    if span <= 0:
        return "mid"
    center = ((low + high) / 2.0 - full_low) / span
    if center < 1.0 / 3.0:
        return "lo"
    if center < 2.0 / 3.0:
        return "mid"
    return "hi"


def allocate_episodes(
    weights: list[float], total: int, minimum: int = 0, step: int = 1
) -> list[int]:
    """Largest-remainder allocation with a per-item minimum, exact total."""
    n = len(weights)
    if n == 0:
        return []
    weights = [max(float(w), 1e-9) for w in weights]
    scale = sum(weights)
    raw = [total * w / scale for w in weights]
    alloc = [max(minimum, int(v // step) * step) for v in raw]
    diff = total - sum(alloc)
    order = sorted(range(n), key=lambda i: raw[i] - alloc[i], reverse=diff > 0)
    idx = 0
    guard = 0
    while diff != 0 and guard < 100000:
        i = order[idx % n]
        if diff > 0:
            alloc[i] += min(step, diff)
            diff = total - sum(alloc)
        else:
            if alloc[i] - step >= minimum:
                alloc[i] -= step
                diff = total - sum(alloc)
        idx += 1
        guard += 1
    return alloc


def round_range(values, digits: int = 5):
    return [[round(float(v), digits) for v in row] for row in values]
