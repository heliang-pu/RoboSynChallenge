"""Real-Time Chunking (RTC) prefix guidance for flow-matching action experts.

Implements the inpainting-style guidance described in

    Kevin Black, Manuel Y. Galliker, Sergey Levine.
    "Real-Time Execution of Action Chunking Flow Policies."
    arXiv:2506.07339 (2025).

The idea: when a new action chunk is sampled while the robot is still executing
the previous one, the first ``inference_delay`` actions of the new chunk are
already spoken for -- they *will* have been executed by the time the new chunk
lands.  Rather than let the flow model pick an arbitrary mode and jump, RTC
treats those timesteps as an inpainting constraint: the sampler is guided so the
new chunk's prefix reproduces the previous chunk, hard for the delay region and
with a decaying weight over the rest of the execution horizon.

Time convention here matches ``openpi``'s sampler: ``t = 1`` is noise, ``t = 0``
is data, ``x_t = t * noise + (1 - t) * actions``, and the model predicts
``v_t = noise - actions``.  Hence ``x_t - t * v_t`` is the running estimate of
the clean chunk (``x1`` in the paper).
"""

import math

import jax.numpy as jnp
import numpy as np

SCHEDULES = ("zeros", "ones", "linear", "exp")


def _lin_weights(start: int, end: int, total: int) -> np.ndarray:
    """Interior of a 1 -> 0 ramp covering the soft region ``[start, end)``."""
    skip_at_end = max(total - end, 0)
    n = total - skip_at_end - start
    if end <= start or n <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.linspace(1.0, 0.0, n + 2, dtype=np.float32)[1:-1]


def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> np.ndarray:
    """Per-timestep guidance weights over a chunk of length ``total``.

    Args:
        start: ``inference_delay`` -- timesteps that are already committed and
            therefore pinned with weight 1.
        end: execution horizon -- beyond this the new chunk is free (weight 0).
        total: action horizon of the chunk.
        schedule: how weights decay across ``[start, end)``.  ``"zeros"`` pins
            only the delay region (RTC degenerates to hard inpainting),
            ``"ones"`` pins the whole execution horizon, ``"linear"`` and
            ``"exp"`` ramp down.

    Returns:
        ``float32`` array of shape ``(total,)`` with values in ``[0, 1]``.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    schedule = schedule.lower()
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown prefix_attention_schedule {schedule!r}, expected one of {SCHEDULES}")

    start = max(0, min(start, end))

    if schedule == "zeros":
        weights = np.zeros(total, dtype=np.float32)
        weights[: min(start, total)] = 1.0
        return weights
    if schedule == "ones":
        weights = np.ones(total, dtype=np.float32)
        if end < total:
            weights[end:] = 0.0
        return weights

    lin = _lin_weights(start, end, total)
    if schedule == "exp":
        # Same reshaping as the reference implementation: pushes weight mass
        # toward the pinned end so the hand-off is smoother than a pure ramp.
        lin = lin * np.expm1(lin) / (math.e - 1.0)

    leading = np.ones(min(start, total), dtype=np.float32)
    trailing = np.zeros(max(total - end, 0), dtype=np.float32)
    weights = np.concatenate([leading, lin, trailing]).astype(np.float32)
    assert weights.shape == (total,), f"built {weights.shape} weights for total={total}"
    return weights


def guidance_weight(time, max_guidance_weight: float):
    """Scalar guidance gain at flow time ``time``.

    Equals ``(t^2 + (1-t)^2) / (t * (1-t))`` clipped at ``max_guidance_weight``
    -- a U-shape that is strongest at both ends of the trajectory and weakest at
    ``t = 0.5``.  Written with a guarded denominator so the endpoints, where the
    exact expression diverges, saturate at the clip instead of producing NaN.
    """
    t = jnp.asarray(time, dtype=jnp.float32)
    denom = t * (1.0 - t)
    numer = t**2 + (1.0 - t) ** 2
    safe = jnp.where(denom > 0, denom, 1.0)
    weight = jnp.where(denom > 0, numer / safe, max_guidance_weight)
    return jnp.minimum(weight, max_guidance_weight)
