"""Common contracts for reusable robot skills.

The classes in this module intentionally do not depend on a concrete robot,
camera, policy, or task. Hardware adapters provide the callbacks used by a
skill while this module provides consistent lifecycle and result semantics.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class SkillStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SkillResult:
    """Structured outcome returned by every reusable skill."""

    status: SkillStatus
    message: str = ""
    metrics: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    final_state: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is SkillStatus.SUCCESS

    @classmethod
    def success(
        cls,
        message: str = "",
        *,
        metrics: Mapping[str, float | int | str | bool] | None = None,
        final_state: Mapping[str, Any] | None = None,
    ) -> SkillResult:
        return cls(
            status=SkillStatus.SUCCESS,
            message=message,
            metrics={} if metrics is None else dict(metrics),
            final_state={} if final_state is None else dict(final_state),
        )


class CancellationToken:
    """Thread-safe cooperative cancellation shared by an orchestration chain."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise SkillCancelledError("skill execution was cancelled")


class SkillError(RuntimeError):
    """Base exception for reusable skill execution failures."""


class SkillCancelledError(SkillError):
    pass


class AtomicSkill(Protocol):
    """Minimal interface understood by the sequential orchestrator."""

    @property
    def name(self) -> str: ...

    def execute(self, *, cancellation: CancellationToken | None = None) -> SkillResult: ...
