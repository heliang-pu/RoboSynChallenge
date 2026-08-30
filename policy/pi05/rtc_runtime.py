"""Chunk scheduling for asynchronous inference with Real-Time Chunking.

Splits the bookkeeping out of the policy adapter so it can be unit-tested
without a simulator or a GPU.  The scheduler owns three things:

* **when** to launch an inference so the resulting chunk lands exactly when the
  currently-executing chunk hands over,
* **which** action to emit at each environment step, and
* **what** to hand RTC as its guidance target -- the previous chunk resampled
  onto the *new* chunk's timeline.

Timeline
--------
Chunk ``k`` becomes active at env step ``a_k = k * H`` (``H`` = execution
horizon) and is executed until ``a_{k+1} - 1``.  Because inference takes ``d``
env steps, it is launched at ``a_k - d`` from the observation available then, so
the chunk it produces is indexed from ``t0 = a_k - d``.  Covering steps ``a_k``
through ``a_{k+1} - 1`` therefore needs chunk indices ``d .. H + d - 1``, which
is why ``H + d`` must not exceed the model's action horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from openpi.models import rtc as _rtc  # noqa: E402

ASYNC_MODES = ("off", "sim", "real")


@dataclass(frozen=True)
class PhaseChunkSelection:
    """One phase selected for the current environment step."""

    name: str
    start_step: int
    end_step: int | None
    execution_horizon: int

    def step_budget(self, current_step: int) -> int:
        """Do not let one policy call cross into the next phase."""
        if self.end_step is None:
            return self.execution_horizon
        return min(self.execution_horizon, self.end_step - int(current_step))


class PhaseChunkSchedule:
    """Map semantic/time phases to different open-loop execution horizons.

    The policy still predicts its full model action horizon (50 actions for the
    current pi0.5 checkpoint).  This schedule changes how many actions are
    executed before replanning.  Phases are half-open intervals
    ``[start_step, end_step)`` and must cover the episode contiguously from zero.
    The final phase may omit ``end_step``.
    """

    def __init__(self, phases: list[dict]):
        if not isinstance(phases, list) or not phases:
            raise ValueError("phase_action_chunks must be a non-empty list")
        parsed = []
        for index, raw in enumerate(phases):
            if not isinstance(raw, dict):
                raise ValueError(f"phase_action_chunks[{index}] must be a mapping")
            name = str(raw.get("name", f"phase_{index}"))
            start = int(raw.get("start_step", 0 if index == 0 else -1))
            end_value = raw.get("end_step")
            end = None if end_value is None else int(end_value)
            horizon = int(raw.get("chunk", raw.get("execution_horizon", 0)))
            if start < 0:
                raise ValueError(f"phase {name!r} requires start_step")
            if end is not None and end <= start:
                raise ValueError(f"phase {name!r} end_step must exceed start_step")
            if horizon < 1:
                raise ValueError(f"phase {name!r} chunk must be >= 1")
            parsed.append(PhaseChunkSelection(name, start, end, horizon))

        if parsed[0].start_step != 0:
            raise ValueError("phase_action_chunks must start at step 0")
        for previous, current in zip(parsed, parsed[1:]):
            if previous.end_step is None:
                raise ValueError(f"open-ended phase {previous.name!r} must be last")
            if previous.end_step != current.start_step:
                raise ValueError(
                    "phase_action_chunks must be contiguous: "
                    f"{previous.name!r} ends at {previous.end_step}, "
                    f"{current.name!r} starts at {current.start_step}"
                )
        self.phases = tuple(parsed)

    def select(self, current_step: int) -> PhaseChunkSelection:
        step = int(current_step)
        if step < 0:
            raise ValueError(f"current_step must be >= 0, got {step}")
        for phase in self.phases:
            if step >= phase.start_step and (
                phase.end_step is None or step < phase.end_step
            ):
                return phase
        final = self.phases[-1]
        raise RuntimeError(
            f"phase_action_chunks does not cover step {step}; "
            f"final phase {final.name!r} ends at {final.end_step}"
        )

    def describe(self) -> list[dict]:
        return [
            {
                "name": phase.name,
                "start_step": phase.start_step,
                "end_step": phase.end_step,
                "chunk": phase.execution_horizon,
            }
            for phase in self.phases
        ]


class ChunkScheduler:
    """Tracks chunk timing and produces RTC guidance targets.

    Args:
        action_horizon: chunk length the model emits (50 for this pi0.5 checkpoint).
        execution_horizon: replan interval ``H`` -- how many env steps each chunk
            is executed for before the next one takes over.
        inference_delay: ``d``, how many env steps one inference occupies.  Zero
            means fully synchronous: the env is frozen while the model runs.
        rtc_enabled: whether to build guidance targets at all.
        prefix_attention_schedule: see `openpi.models.rtc.get_prefix_weights`.
    """

    def __init__(
        self,
        *,
        action_horizon: int,
        execution_horizon: int,
        inference_delay: int = 0,
        rtc_enabled: bool = False,
        prefix_attention_schedule: str = "exp",
    ):
        if execution_horizon < 1:
            raise ValueError(f"execution_horizon must be >= 1, got {execution_horizon}")
        if inference_delay < 0:
            raise ValueError(f"inference_delay must be >= 0, got {inference_delay}")

        self.action_horizon = int(action_horizon)
        self.inference_delay = int(inference_delay)
        self.rtc_enabled = bool(rtc_enabled)
        self.prefix_attention_schedule = prefix_attention_schedule

        self.execution_horizon = 0
        self.execution_horizon_requested = 0
        self.clamped_from = None
        self.set_execution_horizon(execution_horizon)

        self.reset()

    # ---------------------------------------------------------------- state

    def reset(self) -> None:
        self.step_index = 0          # env steps executed this episode
        self.chunk = None            # active chunk, absolute env-space actions (T, A)
        self.chunk_t0 = None         # env step that chunk[0] corresponds to
        self.pending = None          # chunk computed but not yet landed
        self.force_replan = False    # phase change requests immediate replacement
        self.launches = 0
        self.guided_launches = 0

    def set_execution_horizon(self, execution_horizon: int) -> bool:
        """Change the replan interval, returning whether the effective H changed."""
        requested = int(execution_horizon)
        if requested < 1:
            raise ValueError(f"execution_horizon must be >= 1, got {requested}")
        available = self.action_horizon - self.inference_delay
        if available < 1:
            raise ValueError(
                f"inference_delay={self.inference_delay} leaves no room inside "
                f"action_horizon={self.action_horizon}"
            )
        effective = min(requested, available)
        changed = effective != self.execution_horizon
        self.execution_horizon = effective
        self.execution_horizon_requested = requested
        self.clamped_from = requested if effective != requested else None
        return changed

    # ------------------------------------------------------------- schedule

    def next_land_step(self) -> int:
        """Env step at which the next chunk must take over."""
        if self.chunk_t0 is None:
            return 0
        return self.chunk_t0 + self.inference_delay + self.execution_horizon

    def should_launch(self) -> bool:
        """True when inference for the next chunk has to start now."""
        if self.pending is not None:
            return False
        if self.force_replan:
            return True
        if self.chunk is None:
            return True
        return self.step_index >= self.next_land_step() - self.inference_delay

    def request_replan(self) -> None:
        """Force the next scheduler tick to replace the active chunk."""
        if self.pending is not None:
            raise RuntimeError("cannot force a replan while a chunk is pending")
        self.force_replan = True

    def should_adopt(self) -> bool:
        return self.pending is not None and self.step_index >= self.pending["land_step"]

    # -------------------------------------------------------------- guidance

    def guidance(self, launch_step: int) -> dict | None:
        """RTC target for a chunk launched at ``launch_step``.

        The target is returned in *absolute environment* action space, not the
        model's. The model predicts deltas against the state of the inference
        that produced them, so a chunk planned earlier is only meaningful once
        it has been rebased onto the current state -- the caller does that
        conversion, which is why absolute actions are what gets stored here.

        Returns ``None`` when guidance is off or there is no overlap left to
        guide with (the first chunk of an episode, or ``H == action_horizon``
        with a synchronous sampler, where the old chunk is fully consumed).
        """
        if not self.rtc_enabled or self.chunk is None:
            return None

        offset = launch_step - self.chunk_t0
        overlap = self.action_horizon - offset
        if overlap <= 0:
            return None

        # Resample the old chunk onto the new chunk's timeline: new index i is
        # env step launch_step + i, which is old index offset + i.
        aligned = np.zeros_like(self.chunk)
        aligned[:overlap] = self.chunk[offset : offset + overlap]

        # Never let the soft region run past the actions we actually have; past
        # `overlap` the target is zero padding, which is not a real constraint.
        end = min(self.execution_horizon, overlap)
        start = min(self.inference_delay, end)
        weights = _rtc.get_prefix_weights(start, end, self.action_horizon, self.prefix_attention_schedule)
        if not weights.any():
            return None
        return {"prev_actions_env": aligned, "prefix_weights": weights, "overlap": overlap}

    # ---------------------------------------------------------------- chunks

    def stage(
        self, actions: np.ndarray, launch_step: int, land_step: int | None = None
    ) -> None:
        """Record a freshly sampled chunk.

        By default it lands ``inference_delay`` steps after launch, which is the
        deterministic (``async_mode="sim"``) timeline.  ``land_step`` overrides
        that for ``async_mode="real"``, where the chunk lands whenever the
        background inference actually finished.
        """
        self.pending = {
            "actions": np.asarray(actions),
            "t0": int(launch_step),
            "land_step": int(launch_step) + self.inference_delay if land_step is None else int(land_step),
        }
        self.force_replan = False
        self.launches += 1

    def adopt(self) -> None:
        assert self.pending is not None, "adopt() with nothing pending"
        self.chunk = self.pending["actions"]
        self.chunk_t0 = self.pending["t0"]
        self.pending = None

    def action(self) -> np.ndarray:
        """Action for the current env step."""
        if self.chunk is None:
            raise RuntimeError("no active chunk; launch an inference first")
        idx = self.step_index - self.chunk_t0
        if idx < 0:
            raise RuntimeError(f"chunk starts at {self.chunk_t0}, cannot serve step {self.step_index}")
        if idx >= len(self.chunk):
            # Should be unreachable given the H + d <= action_horizon invariant;
            # hold the last action rather than crash a long-running eval.
            idx = len(self.chunk) - 1
        return self.chunk[idx]

    def advance(self) -> None:
        self.step_index += 1

    # ----------------------------------------------------------------- stats

    def describe(self) -> dict:
        return {
            "action_horizon": self.action_horizon,
            "execution_horizon": self.execution_horizon,
            "execution_horizon_requested": self.execution_horizon_requested,
            "execution_horizon_clamped_from": self.clamped_from,
            "inference_delay": self.inference_delay,
            "rtc_enabled": self.rtc_enabled,
            "prefix_attention_schedule": self.prefix_attention_schedule,
            "launches": self.launches,
            "guided_launches": self.guided_launches,
        }
