"""Validated task artifacts used by robot skills."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RecordedPose:
    """A validated joint-space demonstration keyframe."""

    path: Path
    side: str
    joints_deg: tuple[float, ...]
    gripper_pos: float | None
    payload: Mapping[str, Any]

    def joints_array(self) -> np.ndarray:
        return np.asarray(self.joints_deg, dtype=float)


@dataclass(frozen=True)
class RigidTransform:
    """A named homogeneous transform loaded from a calibration artifact."""

    path: Path
    key: str
    matrix: np.ndarray

    def inverse(self) -> RigidTransform:
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]
        result = np.eye(4, dtype=float)
        result[:3, :3] = rotation.T
        result[:3, 3] = -(rotation.T @ translation)
        return RigidTransform(path=self.path, key=f"inverse({self.key})", matrix=result)


def _finite_vector(values: Sequence[float], *, count: int, label: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != count:
        raise ValueError(f"{label} must contain {count} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return tuple(float(value) for value in array)


def load_recorded_pose(path: str | Path, *, side: str = "right", joint_count: int = 6) -> RecordedPose:
    selected = Path(path).expanduser().resolve()
    payload = json.loads(selected.read_text(encoding="utf-8"))
    key = f"{side}_joints_deg"
    if key not in payload:
        raise ValueError(f"pose file does not contain {key}: {selected}")
    joints = _finite_vector(payload[key], count=joint_count, label=key)
    raw_gripper = payload.get(f"{side}_gripper_pos")
    gripper = None if raw_gripper is None else float(raw_gripper)
    if gripper is not None and not np.isfinite(gripper):
        raise ValueError(f"{side}_gripper_pos is not finite")
    return RecordedPose(
        path=selected,
        side=str(side),
        joints_deg=joints,
        gripper_pos=gripper,
        payload=payload,
    )


def validate_rigid_transform(matrix: np.ndarray, *, atol: float = 1e-5) -> np.ndarray:
    transform = np.asarray(matrix, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(transform)):
        raise ValueError("transform contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError(f"invalid homogeneous transform last row: {transform[3].tolist()}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("transform rotation determinant is not +1")
    return transform.copy()


def load_rigid_transform(path: str | Path, *, key: str) -> RigidTransform:
    selected = Path(path).expanduser().resolve()
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if key not in payload:
        raise ValueError(f"calibration file does not contain {key}: {selected}")
    return RigidTransform(path=selected, key=key, matrix=validate_rigid_transform(payload[key]))
