"""openpi runtime environment for the dual-arm CobotMagic PiPER robot.

Observations use the same keys as the RoboSynChallenge pi0 / pi0.5 training
data after repacking (see ``LeRobotEmbodiChainDataConfig``), so the policy
server receives exactly what it saw during training:

    observation/image              cam_high, RGB uint8
    observation/left_wrist_image   cam_left_wrist, RGB uint8
    observation/right_wrist_image  cam_right_wrist, RGB uint8
    observation/state              14-dim joint state, see ``piper_conversions``
    prompt                         language instruction
"""

from __future__ import annotations

import logging
import time

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.cobotmagic_piper_real import hardware as _hardware
from examples.cobotmagic_piper_real import piper_conversions as pc

logger = logging.getLogger(__name__)


class CobotMagicPiperEnvironment(_environment.Environment):
    def __init__(
        self,
        robot: _hardware.CobotMagicPiper,
        prompt: str,
        *,
        safety: pc.SafetyFilter,
        reset_state: np.ndarray | None,
        reset_duration_s: float = 3.0,
        control_hz: float = 25.0,
        image_size: int = 224,
    ) -> None:
        self._robot = robot
        self._prompt = prompt
        self._safety = safety
        self._reset_state = None if reset_state is None else np.asarray(reset_state, dtype=np.float64)
        self._reset_duration_s = reset_duration_s
        self._control_hz = control_hz
        self._image_size = image_size
        # The last commanded target. Step limiting is applied relative to it so that
        # the commanded trajectory stays smooth even if the measured state lags.
        self._last_target: np.ndarray | None = None

    @override
    def reset(self) -> None:
        state = self._robot.read_state().astype(np.float64)
        self._last_target = state
        if self._reset_state is None:
            logger.info("Starting from the current pose: %s", np.round(state, 3))
            return
        num_steps = max(1, round(self._reset_duration_s * self._control_hz))
        logger.info("Moving to the reset pose over %.1f s.", self._reset_duration_s)
        for waypoint in pc.interpolate_path(state, self._reset_state, num_steps):
            self._send(waypoint, warn=False)
            time.sleep(1.0 / self._control_hz)

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        images = self._robot.read_images()

        def prepare(image: np.ndarray) -> np.ndarray:
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, self._image_size, self._image_size))

        return {
            "observation/image": prepare(images["cam_high"]),
            "observation/left_wrist_image": prepare(images["cam_left_wrist"]),
            "observation/right_wrist_image": prepare(images["cam_right_wrist"]),
            "observation/state": self._robot.read_state(),
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        target = np.asarray(action["actions"], dtype=np.float64).reshape(-1)
        if target.shape[0] < pc.STATE_DIM:
            raise ValueError(f"Policy returned {target.shape[0]} action dims, expected {pc.STATE_DIM}.")
        self._send(target[: pc.STATE_DIM], warn=True)

    def _send(self, target: np.ndarray, *, warn: bool) -> None:
        target = np.array(target, dtype=np.float64, copy=True)
        if self._last_target is None:
            self._last_target = self._robot.read_state().astype(np.float64)
        arm = list(pc.ARM_DIMS)
        limited, largest = self._safety.limit_step(target[arm], self._last_target[arm])
        if warn:
            pc.warn_on_jump(largest, self._safety.max_joint_step_rad, logger)
        target[arm] = limited
        for dim in pc.GRIPPER_DIMS:
            target[dim] = float(np.clip(target[dim], 0.0, 1.0))
        self._robot.command(target, self._safety)
        self._last_target = target
