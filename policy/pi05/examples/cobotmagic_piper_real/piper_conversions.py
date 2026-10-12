"""Unit conversions and safety filtering between the policy and dual PiPER arms.

The policy speaks the RoboSynChallenge CobotMagic convention:

    index  0-5   left arm joints 1-6, radians
    index  6     left gripper, normalized opening in [0, 1]
    index  7-12  right arm joints 1-6, radians
    index  13    right gripper, normalized opening in [0, 1]

The gripper value is normalized by ``policy_gripper_full_mm``. With the
``pi05_robosynchallenge_sim2real`` training config this is the 70 mm PiPER
gripper, so ``1.0`` means a 70 mm opening on the real robot.

The pyAgxArm SDK reads and writes joint angles in radians and the AgxGripper
opening width in meters. Everything in this module is pure numpy so that it
can be checked without hardware.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

STATE_DIM = 14
LEFT_ARM = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_ARM = slice(7, 13)
RIGHT_GRIPPER = 13
GRIPPER_DIMS = (LEFT_GRIPPER, RIGHT_GRIPPER)
ARM_DIMS = tuple(range(6)) + tuple(range(7, 13))

# PiPER joint limits in radians, as in agx_arm_urdf piper_description.urdf and the pyAgxArm defaults.
PIPER_JOINT_LIMITS_RAD = np.array(
    [
        [-2.6179938, 2.6179938],
        [0.0, 3.1415926],
        [-2.9670597, 0.0],
        [-1.7453292, 1.7453292],
        [-1.2217304, 1.2217304],
        [-2.0943951, 2.0943951],
    ],
    dtype=np.float64,
)

MM_PER_M = 1000.0


@dataclasses.dataclass(frozen=True)
class ArmCalibration:
    """Per-joint mapping from policy joint angles to PiPER joint angles.

    ``real = sign * policy + offset``. With a matching zero pose the offsets
    stay at zero. Flip a sign only if a single-joint jog moves the real joint
    the opposite way from the simulated one.
    """

    signs: tuple[float, ...] = (1.0,) * 6
    offsets_rad: tuple[float, ...] = (0.0,) * 6

    def __post_init__(self):
        if len(self.signs) != 6 or len(self.offsets_rad) != 6:
            raise ValueError("ArmCalibration needs exactly 6 signs and 6 offsets.")
        if any(s not in (1.0, -1.0) for s in self.signs):
            raise ValueError(f"Joint signs must be +1 or -1, got {self.signs}.")

    def policy_to_real(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self.signs) * np.asarray(q, dtype=np.float64) + np.asarray(self.offsets_rad)

    def real_to_policy(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, dtype=np.float64) - np.asarray(self.offsets_rad)) * np.asarray(self.signs)


def gripper_norm_to_m(value: float, full_mm: float, hardware_max_mm: float) -> float:
    """Policy gripper value to the AgxGripper opening width in meters."""
    opening_mm = float(np.clip(value, 0.0, 1.0)) * full_mm
    return min(opening_mm, hardware_max_mm) / MM_PER_M


def gripper_m_to_norm(width_m: float, full_mm: float) -> float:
    """AgxGripper opening width in meters to the policy gripper value."""
    return float(np.clip(abs(float(width_m)) * MM_PER_M / full_mm, 0.0, 1.0))


@dataclasses.dataclass
class SafetyFilter:
    """Clamp commands to the PiPER joint limits and limit the per-step joint change.

    ``max_joint_step_rad`` is applied per control step. At 25 Hz the default
    0.1 rad corresponds to 2.5 rad/s, below the PiPER rated joint speeds.
    """

    max_joint_step_rad: float = 0.1
    joint_limit_margin_rad: float = 0.02
    joint_limits_rad: np.ndarray = dataclasses.field(default_factory=PIPER_JOINT_LIMITS_RAD.copy)

    def clamp_to_limits(self, real_arm_q: np.ndarray) -> np.ndarray:
        low = self.joint_limits_rad[:, 0] + self.joint_limit_margin_rad
        high = self.joint_limits_rad[:, 1] - self.joint_limit_margin_rad
        return np.clip(real_arm_q, low, high)

    def limit_step(self, target: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float]:
        """Move from ``current`` toward ``target`` by at most ``max_joint_step_rad`` per joint.

        Returns the limited target and the largest requested step, which is
        useful to detect a policy that asks for a jump.
        """
        delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
        largest = float(np.max(np.abs(delta))) if delta.size else 0.0
        limited = np.asarray(current) + np.clip(delta, -self.max_joint_step_rad, self.max_joint_step_rad)
        return limited, largest


def split_state(state: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] < STATE_DIM:
        raise ValueError(f"Expected at least {STATE_DIM} values, got {state.shape[0]}.")
    return state[LEFT_ARM], float(state[LEFT_GRIPPER]), state[RIGHT_ARM], float(state[RIGHT_GRIPPER])


def join_state(left_arm: np.ndarray, left_gripper: float, right_arm: np.ndarray, right_gripper: float) -> np.ndarray:
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[LEFT_ARM] = left_arm
    out[LEFT_GRIPPER] = left_gripper
    out[RIGHT_ARM] = right_arm
    out[RIGHT_GRIPPER] = right_gripper
    return out


def interpolate_path(start: np.ndarray, goal: np.ndarray, num_steps: int) -> np.ndarray:
    """Linear path from ``start`` to ``goal`` with ``num_steps`` waypoints, excluding ``start``."""
    num_steps = max(1, int(num_steps))
    alphas = np.linspace(0.0, 1.0, num_steps + 1)[1:, None]
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return start + alphas * (goal - start)


def warn_on_jump(largest_step: float, threshold: float, logger: logging.Logger) -> None:
    if largest_step > threshold:
        logger.warning(
            "Policy requested a %.3f rad joint step; limited to %.3f rad. "
            "Frequent warnings usually mean a state or unit mismatch.",
            largest_step,
            threshold,
        )
