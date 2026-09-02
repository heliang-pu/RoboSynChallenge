#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

PIPER_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
PIPER_JOINT_ACTION_KEYS = tuple(f"{joint}.pos" for joint in PIPER_JOINT_NAMES)
PIPER_ACTION_KEYS = PIPER_JOINT_ACTION_KEYS + ("gripper.pos",)
PIPER_CTRL_MODE_TEACH = 0x02
PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT = 0x06


def milli_to_unit(value: float | int) -> float:
    return float(value) * 1e-3


def unit_to_milli(value: float | int) -> int:
    return int(round(float(value) * 1e3))


@lru_cache(maxsize=1)
def get_piper_sdk() -> tuple[type[Any], Any]:
    try:
        from piper_sdk import C_PiperInterface_V2, LogLevel

        return C_PiperInterface_V2, LogLevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `piper_sdk`. Install Evo-RL dependencies first (for example: `pip install -e .`)."
        ) from exc


def parse_piper_log_level(level_name: str) -> Any:
    _, log_level_enum = get_piper_sdk()
    normalized = level_name.upper()
    try:
        return getattr(log_level_enum, normalized)
    except AttributeError as exc:
        raise ValueError(
            f"Invalid Piper log level '{level_name}'. "
            "Expected one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT."
        ) from exc


def wait_enable_piper(arm: Any, timeout_s: float, retry_interval_s: float = 0.2) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    interval_s = max(0.01, retry_interval_s)
    while time.monotonic() < deadline:
        if bool(arm.EnablePiper()):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(interval_s, remaining_s))
    return False


def _piper_mode_to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "value") and isinstance(value.value, int):
        return int(value.value)
    if hasattr(value, "__int__"):
        return int(value)
    return None


def read_piper_ctrl_mode(arm: Any, timeout_s: float = 1.0, poll_s: float = 0.02) -> int | None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        status_msg = arm.GetArmStatus()
        if getattr(status_msg, "time_stamp", 0.0) > 0.0:
            arm_status = getattr(status_msg, "arm_status", None)
            if arm_status is not None:
                mode = _piper_mode_to_int(getattr(arm_status, "ctrl_mode", None))
                if mode is not None:
                    return mode
        time.sleep(max(0.005, poll_s))
    return None


def guard_piper_ctrl_mode_on_connect(
    arm: Any,
    *,
    interface_name: str,
    timeout_s: float = 0.5,
    poll_s: float = 0.02,
    settle_s: float = 0.05,
    allow_missing: bool = False,
) -> int | None:
    mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s)
    if mode is None:
        if allow_missing:
            return None
        raise RuntimeError(
            f"[{interface_name}] could not read arm ctrl_mode within {timeout_s:.2f}s. "
            "Check CAN wiring/power and rerun."
        )
    if mode in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
        arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
        recover_wait_s = max(settle_s, 0.5)
        if recover_wait_s > 0:
            time.sleep(recover_wait_s)
        mode_after = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s)
        if mode_after not in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
            return mode_after
        mode_after_str = f"0x{mode_after:02X}" if mode_after is not None else "unknown"
        raise RuntimeError(
            f"[{interface_name}] arm is in master/teaching role (ctrl_mode=0x{mode:02X}) "
            f"and did not switch to follower mode after 0xFC command (still {mode_after_str}). "
            "Power-cycle this arm, then retry."
        )
    return mode
