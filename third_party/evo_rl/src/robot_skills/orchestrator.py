"""Small deterministic orchestrator for composing atomic skills."""

from __future__ import annotations

from collections.abc import Sequence

from .base import AtomicSkill, CancellationToken, SkillResult, SkillStatus


class SequentialSkill:
    """Run skills in order and stop immediately on the first non-success result."""

    def __init__(self, steps: Sequence[AtomicSkill], *, name: str = "sequence") -> None:
        self.steps = tuple(steps)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, *, cancellation: CancellationToken | None = None) -> SkillResult:
        token = cancellation or CancellationToken()
        completed: list[str] = []
        for step in self.steps:
            if token.cancelled:
                return SkillResult(
                    SkillStatus.CANCELLED,
                    f"{self.name} cancelled before {step.name}",
                    final_state={"completed_steps": tuple(completed)},
                )
            result = step.execute(cancellation=token)
            if not result.ok:
                return SkillResult(
                    result.status,
                    f"{step.name}: {result.message}",
                    metrics=result.metrics,
                    final_state={"completed_steps": tuple(completed), "failed_step": step.name},
                )
            completed.append(step.name)
        return SkillResult.success(
            f"{self.name} completed",
            metrics={"step_count": len(completed)},
            final_state={"completed_steps": tuple(completed)},
        )
