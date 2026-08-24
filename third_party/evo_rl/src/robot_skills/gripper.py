"""Reusable gripper command skill."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .base import CancellationToken, SkillCancelledError, SkillResult, SkillStatus

ActionSender = Callable[[Mapping[str, float]], object]


@dataclass(frozen=True)
class GripperCommandConfig:
    action_key: str
    target_pos: float
    duration_s: float = 1.0
    control_fps: float = 20.0

    def __post_init__(self) -> None:
        if not self.action_key.strip():
            raise ValueError("action_key must not be empty")
        if self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if self.control_fps <= 0:
            raise ValueError("control_fps must be positive")


def gripper_action_key(*, robot_type: str, side: str) -> str:
    prefix = f"{side}_" if str(robot_type).startswith("bi_") else ""
    return f"{prefix}gripper.pos"


class GripperCommandSkill:
    def __init__(
        self,
        *,
        send_action: ActionSender,
        config: GripperCommandConfig,
        name: str = "gripper_command",
    ) -> None:
        self._name = name
        self.send_action = send_action
        self.config = config

    @property
    def name(self) -> str:
        return self._name

    def execute(self, *, cancellation: CancellationToken | None = None) -> SkillResult:
        steps = max(int(round(self.config.duration_s * self.config.control_fps)), 1)
        started = time.monotonic()
        for index in range(steps):
            try:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
            except SkillCancelledError as exc:
                return SkillResult(
                    SkillStatus.CANCELLED,
                    str(exc),
                    metrics={"commands": index, "elapsed_s": time.monotonic() - started},
                )
            self.send_action({self.config.action_key: float(self.config.target_pos)})
            if index + 1 < steps:
                time.sleep(1.0 / self.config.control_fps)
        return SkillResult.success(
            "gripper command completed",
            metrics={"commands": steps, "elapsed_s": time.monotonic() - started},
            final_state={self.config.action_key: float(self.config.target_pos)},
        )
