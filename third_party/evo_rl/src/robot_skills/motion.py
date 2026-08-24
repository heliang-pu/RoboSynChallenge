"""Hardware-independent joint and Cartesian motion skills."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .base import CancellationToken, SkillCancelledError, SkillResult, SkillStatus

JointReader = Callable[[], Sequence[float]]
JointCommander = Callable[[np.ndarray], Mapping[str, object] | None]
PoseReader = Callable[[], np.ndarray]
PoseCommander = Callable[[np.ndarray], Mapping[str, object] | None]


@dataclass(frozen=True)
class JointTrajectoryPoint:
    time_from_start_s: float
    joints_deg: tuple[float, ...]


@dataclass(frozen=True)
class JointMoveConfig:
    max_step_deg: float = 1.0
    tolerance_deg: float = 0.5
    control_fps: float = 20.0
    timeout_s: float = 20.0

    def __post_init__(self) -> None:
        for label, value in (
            ("max_step_deg", self.max_step_deg),
            ("tolerance_deg", self.tolerance_deg),
            ("control_fps", self.control_fps),
            ("timeout_s", self.timeout_s),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")


@dataclass(frozen=True)
class LinearMoveConfig:
    distance_m: float
    step_m: float = 0.004
    control_fps: float = 20.0
    timeout_s: float = 10.0
    tolerance_m: float = 0.002

    def __post_init__(self) -> None:
        for label, value in (
            ("distance_m", self.distance_m),
            ("step_m", self.step_m),
            ("control_fps", self.control_fps),
            ("timeout_s", self.timeout_s),
            ("tolerance_m", self.tolerance_m),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")


def quintic_smoothstep(unit_t: float) -> float:
    u = float(np.clip(unit_t, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def quintic_joint_trajectory(
    start_joints_deg: Sequence[float],
    target_joints_deg: Sequence[float],
    *,
    duration_s: float,
    fps: float,
) -> list[JointTrajectoryPoint]:
    if duration_s <= 0 or fps <= 0:
        raise ValueError("duration_s and fps must be positive")
    start = np.asarray(start_joints_deg, dtype=float).reshape(-1)
    target = np.asarray(target_joints_deg, dtype=float).reshape(start.shape)
    steps = max(int(math.ceil(float(duration_s) * float(fps))), 1)
    return [
        JointTrajectoryPoint(
            time_from_start_s=float(duration_s) * index / steps,
            joints_deg=tuple(
                float(value) for value in start + (target - start) * quintic_smoothstep(index / steps)
            ),
        )
        for index in range(1, steps + 1)
    ]


def clipped_joint_step(
    current_joints_deg: Sequence[float],
    target_joints_deg: Sequence[float],
    *,
    max_step_deg: float,
) -> np.ndarray:
    if max_step_deg <= 0:
        raise ValueError("max_step_deg must be positive")
    current = np.asarray(current_joints_deg, dtype=float)
    target = np.asarray(target_joints_deg, dtype=float).reshape(current.shape)
    return current + np.clip(target - current, -float(max_step_deg), float(max_step_deg))


def max_joint_error_deg(current_joints_deg: Sequence[float], target_joints_deg: Sequence[float]) -> float:
    current = np.asarray(current_joints_deg, dtype=float)
    target = np.asarray(target_joints_deg, dtype=float).reshape(current.shape)
    return float(np.max(np.abs(target - current)))


def joint_target_reached(
    current_joints_deg: Sequence[float],
    target_joints_deg: Sequence[float],
    *,
    tolerance_deg: float,
) -> bool:
    return max_joint_error_deg(current_joints_deg, target_joints_deg) <= float(tolerance_deg)


def normalized_axis(axis: Sequence[float]) -> np.ndarray:
    vector = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("motion axis must be non-zero")
    return vector / norm


def translated_pose(pose: np.ndarray, *, axis: Sequence[float], distance_m: float) -> np.ndarray:
    result = np.asarray(pose, dtype=float).reshape(4, 4).copy()
    result[:3, 3] += normalized_axis(axis) * float(distance_m)
    return result


def linear_pose_waypoints(
    start_pose: np.ndarray,
    *,
    axis: Sequence[float],
    distance_m: float,
    step_m: float,
) -> list[np.ndarray]:
    if distance_m <= 0 or step_m <= 0:
        raise ValueError("distance_m and step_m must be positive")
    count = int(math.ceil(float(distance_m) / float(step_m)))
    return [
        translated_pose(
            start_pose,
            axis=axis,
            distance_m=min(index * float(step_m), float(distance_m)),
        )
        for index in range(1, count + 1)
    ]


def straight_insert_target_pose(preinsert_pose: np.ndarray, *, insert_distance_m: float) -> np.ndarray:
    return translated_pose(preinsert_pose, axis=(0.0, 0.0, -1.0), distance_m=insert_distance_m)


def make_straight_insert_waypoints(
    preinsert_pose: np.ndarray,
    *,
    insert_distance_m: float,
    step_m: float,
) -> list[np.ndarray]:
    return linear_pose_waypoints(
        preinsert_pose,
        axis=(0.0, 0.0, -1.0),
        distance_m=insert_distance_m,
        step_m=step_m,
    )


def monotonic_insert_step_pose(current_pose: np.ndarray, *, final_z_m: float, step_m: float) -> np.ndarray:
    current = np.asarray(current_pose, dtype=float).reshape(4, 4)
    target = current.copy()
    current_z = float(current[2, 3])
    target[2, 3] = min(current_z, max(float(final_z_m), current_z - float(step_m)))
    return target


def locked_insert_step_pose(
    *,
    current_pose: np.ndarray,
    locked_preinsert_pose: np.ndarray,
    final_z_m: float,
    step_m: float,
) -> np.ndarray:
    current = np.asarray(current_pose, dtype=float).reshape(4, 4)
    target = np.asarray(locked_preinsert_pose, dtype=float).reshape(4, 4).copy()
    current_z = float(current[2, 3])
    target[2, 3] = min(current_z, max(float(final_z_m), current_z - float(step_m)))
    return target


def insert_z_tolerance_m(*, step_m: float) -> float:
    return min(0.002, max(0.001, float(step_m) * 0.5))


def execute_joint_move(
    *,
    read_joints: JointReader,
    command_joints: JointCommander,
    target_joints_deg: Sequence[float],
    config: JointMoveConfig = JointMoveConfig(),
    cancellation: CancellationToken | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SkillResult:
    target = np.asarray(target_joints_deg, dtype=float).reshape(-1)
    started = clock()
    commands = 0
    while True:
        try:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
        except SkillCancelledError as exc:
            return SkillResult(SkillStatus.CANCELLED, str(exc), metrics={"commands": commands})
        current = np.asarray(read_joints(), dtype=float).reshape(target.shape)
        error = max_joint_error_deg(current, target)
        if error <= config.tolerance_deg:
            return SkillResult.success(
                "joint target reached",
                metrics={"commands": commands, "max_joint_error_deg": error, "elapsed_s": clock() - started},
                final_state={"joints_deg": tuple(float(value) for value in current)},
            )
        if clock() - started >= config.timeout_s:
            return SkillResult(
                SkillStatus.TIMEOUT,
                "joint move timed out",
                metrics={"commands": commands, "max_joint_error_deg": error, "elapsed_s": clock() - started},
            )
        command_joints(clipped_joint_step(current, target, max_step_deg=config.max_step_deg))
        commands += 1
        sleep(1.0 / config.control_fps)


class MoveToRecordedPoseSkill:
    def __init__(
        self,
        *,
        read_joints: JointReader,
        command_joints: JointCommander,
        target_joints_deg: Sequence[float],
        config: JointMoveConfig = JointMoveConfig(),
        name: str = "move_to_recorded_pose",
    ) -> None:
        self._name = name
        self.read_joints = read_joints
        self.command_joints = command_joints
        self.target_joints_deg = tuple(float(value) for value in target_joints_deg)
        self.config = config

    @property
    def name(self) -> str:
        return self._name

    def execute(self, *, cancellation: CancellationToken | None = None) -> SkillResult:
        return execute_joint_move(
            read_joints=self.read_joints,
            command_joints=self.command_joints,
            target_joints_deg=self.target_joints_deg,
            config=self.config,
            cancellation=cancellation,
        )


def execute_linear_move(
    *,
    read_pose: PoseReader,
    command_pose: PoseCommander,
    axis: Sequence[float],
    config: LinearMoveConfig,
    cancellation: CancellationToken | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SkillResult:
    """Execute a feedback-driven straight move with locked orientation.

    command_pose is the hardware adapter boundary. For Piper it can perform
    one DLS IK step; for MoveIt it can execute one LIN waypoint. The target pose
    always comes from the initial pose, so orthogonal translation and rotation
    cannot drift in the command sequence.
    """

    direction = normalized_axis(axis)
    start_pose = np.asarray(read_pose(), dtype=float).reshape(4, 4).copy()
    started = clock()
    commands = 0
    while True:
        try:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
        except SkillCancelledError as exc:
            return SkillResult(SkillStatus.CANCELLED, str(exc), metrics={"commands": commands})

        current_pose = np.asarray(read_pose(), dtype=float).reshape(4, 4)
        progress_m = float(np.dot(current_pose[:3, 3] - start_pose[:3, 3], direction))
        remaining_m = float(config.distance_m) - progress_m
        if remaining_m <= config.tolerance_m:
            return SkillResult.success(
                "linear target reached",
                metrics={
                    "commands": commands,
                    "progress_m": progress_m,
                    "remaining_m": max(0.0, remaining_m),
                    "elapsed_s": clock() - started,
                },
                final_state={"pose": current_pose.tolist()},
            )
        if clock() - started >= config.timeout_s:
            return SkillResult(
                SkillStatus.TIMEOUT,
                "linear move timed out",
                metrics={
                    "commands": commands,
                    "progress_m": progress_m,
                    "remaining_m": remaining_m,
                    "elapsed_s": clock() - started,
                },
            )

        next_progress_m = min(max(progress_m, 0.0) + config.step_m, config.distance_m)
        waypoint = translated_pose(start_pose, axis=direction, distance_m=next_progress_m)
        command_pose(waypoint)
        commands += 1
        sleep(1.0 / config.control_fps)


class LinearCartesianSkill:
    def __init__(
        self,
        *,
        read_pose: PoseReader,
        command_pose: PoseCommander,
        axis: Sequence[float],
        config: LinearMoveConfig,
        name: str = "linear_cartesian_move",
    ) -> None:
        self._name = name
        self.read_pose = read_pose
        self.command_pose = command_pose
        self.axis = tuple(float(value) for value in axis)
        self.config = config

    @property
    def name(self) -> str:
        return self._name

    def execute(self, *, cancellation: CancellationToken | None = None) -> SkillResult:
        return execute_linear_move(
            read_pose=self.read_pose,
            command_pose=self.command_pose,
            axis=self.axis,
            config=self.config,
            cancellation=cancellation,
        )
