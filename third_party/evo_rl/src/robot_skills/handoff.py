"""Observation-based policy-to-skill handoff detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HandoffStabilityConfig:
    enabled: bool
    fps: int
    min_runtime_s: float
    stable_s: float
    stable_joint_delta_deg: float
    gripper_key: str
    min_gripper_pos: float
    joint_prefix: str = "right_joint_"
    joint_suffix: str = ".pos"

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.min_runtime_s < 0 or self.stable_s < 0:
            raise ValueError("handoff timing values must be non-negative")
        if self.stable_joint_delta_deg < 0:
            raise ValueError("stable_joint_delta_deg must be non-negative")


def extract_handoff_joints(
    observation: Mapping[str, Any],
    *,
    joint_prefix: str,
    joint_suffix: str,
) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in observation.items()
        if str(key).startswith(joint_prefix) and str(key).endswith(joint_suffix)
    }


class HandoffStabilityDetector:
    """Detect a held object and a stable arm for a configurable duration."""

    def __init__(self, config: HandoffStabilityConfig, *, start_time_s: float):
        self.config = config
        self.start_time_s = float(start_time_s)
        self.previous_joints: dict[str, float] | None = None
        self.stable_count = 0
        self.last_max_delta_deg: float | None = None
        self.last_reject_reason = "not_started"

    @property
    def stable_required(self) -> int:
        return max(int(float(self.config.stable_s) * int(self.config.fps)), 1)

    def reset(self, *, reason: str = "reset") -> None:
        self.previous_joints = None
        self.stable_count = 0
        self.last_max_delta_deg = None
        self.last_reject_reason = reason

    def update(self, observation: Mapping[str, Any], *, now_s: float) -> bool:
        if not self.config.enabled:
            self.reset(reason="disabled")
            return False
        if float(now_s) - self.start_time_s < float(self.config.min_runtime_s):
            self.reset(reason="minimum_runtime")
            return False
        gripper_value = observation.get(self.config.gripper_key)
        if gripper_value is None or float(gripper_value) < float(self.config.min_gripper_pos):
            self.reset(reason="gripper_condition")
            return False
        joints = extract_handoff_joints(
            observation,
            joint_prefix=self.config.joint_prefix,
            joint_suffix=self.config.joint_suffix,
        )
        if not joints:
            self.reset(reason="missing_joints")
            return False
        if self.previous_joints is None:
            self.previous_joints = joints
            self.stable_count = 0
            self.last_reject_reason = "first_sample"
            return False
        common_keys = set(joints) & set(self.previous_joints)
        if not common_keys:
            self.previous_joints = joints
            self.stable_count = 0
            self.last_reject_reason = "no_common_joints"
            return False
        max_delta = max(abs(joints[key] - self.previous_joints[key]) for key in common_keys)
        self.last_max_delta_deg = float(max_delta)
        self.previous_joints = joints
        if max_delta <= float(self.config.stable_joint_delta_deg):
            self.stable_count += 1
            self.last_reject_reason = ""
        else:
            self.stable_count = 0
            self.last_reject_reason = "arm_moving"
        return self.stable_count >= self.stable_required
