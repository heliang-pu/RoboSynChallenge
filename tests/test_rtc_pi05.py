"""Timeline and guidance-alignment tests for the pi0.5 RTC scheduler."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "policy" / "pi05"))
sys.path.insert(0, str(REPO_ROOT / "policy" / "pi05" / "src"))

from openpi.models import rtc  # noqa: E402
from rtc_runtime import ChunkScheduler  # noqa: E402

AH = 50  # action horizon of the pi0.5 checkpoint under test


def stamped_chunk(t0, dim=4):
    """Chunk whose every entry records the env step it belongs to."""
    return np.stack([np.full(dim, t0 + i, dtype=np.float32) for i in range(AH)])


def run_episode(sched, steps):
    """Drive the scheduler for `steps` env steps, returning the action stream."""
    emitted = []
    while sched.step_index < steps:
        if sched.should_launch():
            launch = sched.step_index
            guide = sched.guidance(launch)
            if guide is not None:
                sched.guided_launches += 1
                # The guidance target must agree with the old plan step-for-step.
                overlap = guide["overlap"]
                expected = np.arange(launch, launch + overlap, dtype=np.float32)
                np.testing.assert_allclose(guide["prev_chunk"][:overlap, 0], expected)
                np.testing.assert_allclose(guide["prev_chunk"][overlap:], 0.0)
            chunk = stamped_chunk(launch)
            sched.stage(chunk, chunk, launch)
            if sched.chunk is None:  # first chunk of the episode lands immediately
                sched.adopt()
        if sched.should_adopt():
            sched.adopt()
        emitted.append(sched.action()[0])
        sched.advance()
    return np.array(emitted)


@pytest.mark.parametrize("horizon", [10, 30, 50])
def test_sync_matches_open_loop(horizon):
    """With no delay the scheduler reproduces plain open-loop chunk execution."""
    sched = ChunkScheduler(action_horizon=AH, execution_horizon=horizon, inference_delay=0)
    emitted = run_episode(sched, 200)
    # Every action is the one the freshest plan intended for that exact step.
    np.testing.assert_allclose(emitted, np.arange(200, dtype=np.float32))
    assert sched.launches == int(np.ceil(200 / horizon))


@pytest.mark.parametrize("horizon", [10, 30, 50])
@pytest.mark.parametrize("delay", [1, 5])
def test_async_never_runs_out_of_actions(horizon, delay):
    """Chunks must cover every step they are responsible for, with no stalling."""
    sched = ChunkScheduler(
        action_horizon=AH, execution_horizon=horizon, inference_delay=delay, rtc_enabled=True
    )
    emitted = run_episode(sched, 300)
    # A stalled scheduler would repeat its last action; a correct one emits the
    # action planned for each step, so the stream is strictly increasing.
    assert np.all(np.diff(emitted) == 1), "scheduler stalled or jumped"
    assert emitted[0] == 0


def test_execution_horizon_clamped_to_fit_delay():
    """H + d must fit inside the action horizon; H is clamped, loudly."""
    sched = ChunkScheduler(action_horizon=AH, execution_horizon=50, inference_delay=5)
    assert sched.execution_horizon == 45
    assert sched.clamped_from == 50

    sched = ChunkScheduler(action_horizon=AH, execution_horizon=30, inference_delay=5)
    assert sched.execution_horizon == 30
    assert sched.clamped_from is None


def test_guidance_absent_without_overlap():
    """Synchronous H == action_horizon consumes the chunk fully: nothing to guide."""
    sched = ChunkScheduler(
        action_horizon=AH, execution_horizon=AH, inference_delay=0, rtc_enabled=True
    )
    chunk = stamped_chunk(0)
    sched.stage(chunk, chunk, 0)
    sched.adopt()
    assert sched.guidance(AH) is None  # old chunk exactly exhausted


def test_guidance_overlap_shrinks_with_horizon():
    """Larger execution horizons leave less of the old chunk to anchor against."""
    overlaps = {}
    for horizon in (10, 30, 50):
        sched = ChunkScheduler(
            action_horizon=AH, execution_horizon=horizon, inference_delay=5, rtc_enabled=True
        )
        chunk = stamped_chunk(0)
        sched.stage(chunk, chunk, 0)
        sched.adopt()
        guide = sched.guidance(sched.execution_horizon)
        overlaps[horizon] = guide["overlap"]
    assert overlaps == {10: 40, 30: 20, 50: 5}


def test_prefix_weights_pin_delay_region():
    """The delay region is hard-pinned; the soft region decays to zero by H."""
    for schedule in ("linear", "exp"):
        w = rtc.get_prefix_weights(5, 20, AH, schedule)
        np.testing.assert_allclose(w[:5], 1.0)
        assert np.all(w[20:] == 0.0)
        assert np.all(np.diff(w[5:20]) < 0), "soft region must decay monotonically"


def test_zeros_schedule_is_hard_inpainting_only():
    w = rtc.get_prefix_weights(5, 20, AH, "zeros")
    np.testing.assert_allclose(w[:5], 1.0)
    assert w[5:].sum() == 0.0


def test_guidance_weight_saturates_at_endpoints():
    assert float(rtc.guidance_weight(1.0, 10.0)) == 10.0
    assert float(rtc.guidance_weight(0.0, 10.0)) == 10.0
    assert float(rtc.guidance_weight(0.5, 10.0)) == pytest.approx(2.0)
