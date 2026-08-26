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

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from openpi.models import rtc as _rtc  # noqa: E402

ASYNC_MODES = ("off", "sim", "real")


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

        # A chunk launched d steps early must still cover H steps of execution,
        # so H + d actions have to fit inside the model's action horizon.
        requested = int(execution_horizon)
        self.execution_horizon = min(requested, self.action_horizon - self.inference_delay)
        if self.execution_horizon < 1:
            raise ValueError(
                f"inference_delay={self.inference_delay} leaves no room inside "
                f"action_horizon={self.action_horizon}"
            )
        self.clamped_from = requested if self.execution_horizon != requested else None

        self.reset()

    # ---------------------------------------------------------------- state

    def reset(self) -> None:
        self.step_index = 0          # env steps executed this episode
        self.chunk = None            # active chunk, env-space actions (T, A)
        self.chunk_raw = None        # active chunk, model-space (T, A') for RTC
        self.chunk_t0 = None         # env step that chunk[0] corresponds to
        self.pending = None          # chunk computed but not yet landed
        self.launches = 0
        self.guided_launches = 0

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
        if self.chunk is None:
            return True
        return self.step_index >= self.next_land_step() - self.inference_delay

    def should_adopt(self) -> bool:
        return self.pending is not None and self.step_index >= self.pending["land_step"]

    # -------------------------------------------------------------- guidance

    def guidance(self, launch_step: int) -> dict | None:
        """RTC target for a chunk launched at ``launch_step``.

        Returns ``None`` when guidance is off or there is no overlap left to
        guide with (the first chunk of an episode, or ``H == action_horizon``
        with a synchronous sampler, where the old chunk is fully consumed).
        """
        if not self.rtc_enabled or self.chunk_raw is None:
            return None

        offset = launch_step - self.chunk_t0
        overlap = self.action_horizon - offset
        if overlap <= 0:
            return None

        # Resample the old chunk onto the new chunk's timeline: new index i is
        # env step launch_step + i, which is old index offset + i.
        aligned = np.zeros_like(self.chunk_raw)
        aligned[:overlap] = self.chunk_raw[offset : offset + overlap]

        # Never let the soft region run past the actions we actually have; past
        # `overlap` the target is zero padding, which is not a real constraint.
        end = min(self.execution_horizon, overlap)
        start = min(self.inference_delay, end)
        weights = _rtc.get_prefix_weights(start, end, self.action_horizon, self.prefix_attention_schedule)
        if not weights.any():
            return None
        return {"prev_chunk": aligned, "prefix_weights": weights, "overlap": overlap}

    # ---------------------------------------------------------------- chunks

    def stage(
        self, actions: np.ndarray, raw_actions: np.ndarray, launch_step: int, land_step: int | None = None
    ) -> None:
        """Record a freshly sampled chunk.

        By default it lands ``inference_delay`` steps after launch, which is the
        deterministic (``async_mode="sim"``) timeline.  ``land_step`` overrides
        that for ``async_mode="real"``, where the chunk lands whenever the
        background inference actually finished.
        """
        self.pending = {
            "actions": np.asarray(actions),
            "raw": np.asarray(raw_actions),
            "t0": int(launch_step),
            "land_step": int(launch_step) + self.inference_delay if land_step is None else int(land_step),
        }
        self.launches += 1

    def adopt(self) -> None:
        assert self.pending is not None, "adopt() with nothing pending"
        self.chunk = self.pending["actions"]
        self.chunk_raw = self.pending["raw"]
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
            "execution_horizon_requested": self.clamped_from,
            "inference_delay": self.inference_delay,
            "rtc_enabled": self.rtc_enabled,
            "prefix_attention_schedule": self.prefix_attention_schedule,
            "launches": self.launches,
            "guided_launches": self.guided_launches,
        }
