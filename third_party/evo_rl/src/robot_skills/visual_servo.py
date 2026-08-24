"""Task-neutral pieces used by image-feature servo controllers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureEstimate:
    value: float
    confidence: float
    valid: bool = True
    reject_reason: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


def piecewise_linear_compensation(
    value: float,
    *,
    negative_reference: float,
    center_reference: float,
    positive_reference: float,
    negative_compensation_m: float,
    positive_compensation_m: float,
    deadband: float = 0.0,
) -> float:
    """Map a scalar visual feature to a bounded signed Cartesian offset."""
    low = float(negative_reference)
    center = float(center_reference)
    high = float(positive_reference)
    if not low < center < high:
        raise ValueError("references must satisfy negative < center < positive")
    error = float(value) - center
    if abs(error) <= float(deadband):
        return 0.0
    if error < 0.0:
        alpha = float(np.clip((center - float(value)) / (center - low), 0.0, 1.0))
        return alpha * float(negative_compensation_m)
    alpha = float(np.clip((float(value) - center) / (high - center), 0.0, 1.0))
    return alpha * float(positive_compensation_m)


def compensated_pose_along_axis(
    base_pose: np.ndarray,
    *,
    axis_in_base: Sequence[float],
    compensation_m: float,
    preserve_z: bool = False,
) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=float).reshape(4, 4).copy()
    axis = np.asarray(axis_in_base, dtype=float).reshape(3)
    if preserve_z:
        axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("axis_in_base must be non-zero after constraints")
    pose[:3, 3] += axis / norm * float(compensation_m)
    return pose
