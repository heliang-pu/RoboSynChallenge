"""Hardware access for a dual-arm CobotMagic built from two PiPER arms and three D435 cameras.

The arms are driven through AgileX's pyAgxArm SDK
(https://github.com/agilexrobotics/pyAgxArm), which works with all PiPER
firmware versions. pyAgxArm and pyrealsense2 are imported lazily so that the
rest of the client, and its tests, run on machines without them.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

import numpy as np

from examples.cobotmagic_piper_real import piper_conversions as pc

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ArmConfig:
    # CAN channel, e.g. "can_left" or "can0" with interface "socketcan".
    channel: str
    calibration: pc.ArmCalibration = dataclasses.field(default_factory=pc.ArmCalibration)
    # pyAgxArm PiperFW profile: "default" (<= S-V1.8-2), "v183", "v188" or "v189" (>= S-V1.8-9).
    firmware: str = "default"
    interface: str = "socketcan"


@dataclasses.dataclass(frozen=True)
class GripperConfig:
    # Opening that the policy's gripper value 1.0 stands for.
    policy_full_mm: float = 70.0
    # Largest opening the gripper hardware accepts.
    hardware_max_mm: float = 70.0
    # Gripping force in N, range 0-3.
    force_n: float = 1.0


def _wait_for(read, what: str, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while True:
        value = read()
        if value is not None:
            return value
        if time.monotonic() > deadline:
            raise RuntimeError(f"Timed out waiting for {what}.")
        time.sleep(0.01)


class PiperArm:
    """One PiPER arm with its AgxGripper, addressed through its CAN channel."""

    def __init__(
        self,
        config: ArmConfig,
        gripper: GripperConfig,
        *,
        send_commands: bool,
        speed_percent: int = 100,
        sdk_config_overrides: dict | None = None,
        feedback_timeout_s: float = 3.0,
    ) -> None:
        from pyAgxArm import AgxArmFactory  # noqa: PLC0415 - optional hardware dependency
        from pyAgxArm import ArmModel  # noqa: PLC0415 - optional hardware dependency
        from pyAgxArm import create_agx_arm_config  # noqa: PLC0415 - optional hardware dependency

        self._config = config
        self._gripper_config = gripper
        self._send_commands = send_commands
        sdk_config = create_agx_arm_config(
            robot=ArmModel.PIPER,
            firmeware_version=config.firmware,
            interface=config.interface,
            channel=config.channel,
            **(sdk_config_overrides or {}),
        )
        self._robot = AgxArmFactory.create_arm(sdk_config)
        # The effector has to be created before connecting.
        self._gripper = self._robot.init_effector(self._robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._robot.connect()
        self._feedback_timeout_s = feedback_timeout_s

        firmware = self._robot.get_firmware()
        if firmware is not None:
            self._check_firmware(firmware.get("software_version"))

        if send_commands:
            _wait_for(lambda: True if self._robot.enable() else None, f"enabling the arm on {config.channel}", 5.0)
            self._robot.set_speed_percent(int(np.clip(speed_percent, 1, 100)))
        logger.info("Connected PiPER on %s (commands %s).", config.channel, "on" if send_commands else "off")

    def _check_firmware(self, software_version: str | None) -> None:
        if not software_version:
            return
        from pyAgxArm import ArmModel  # noqa: PLC0415 - optional hardware dependency
        from pyAgxArm import resolve_firmware_profile  # noqa: PLC0415 - optional hardware dependency

        try:
            expected = resolve_firmware_profile(ArmModel.PIPER, software_version)
        except Exception:  # Unknown versions are reported, not fatal.
            logger.warning("Could not map PiPER firmware %s to a pyAgxArm profile.", software_version)
            return
        if expected != self._config.firmware:
            logger.warning(
                "PiPER on %s runs firmware %s, which pyAgxArm maps to profile %r, but %r is configured. "
                "Set the firmware option to %r.",
                self._config.channel,
                software_version,
                expected,
                self._config.firmware,
                expected,
            )

    def read(self) -> tuple[np.ndarray, float]:
        """Return the arm joints in policy radians and the gripper in policy units."""
        joints = _wait_for(
            self._robot.get_joint_angles, f"joint feedback on {self._config.channel}", self._feedback_timeout_s
        )
        status = _wait_for(
            self._gripper.get_gripper_status, f"gripper feedback on {self._config.channel}", self._feedback_timeout_s
        )
        return (
            self._config.calibration.real_to_policy(np.asarray(joints.msg, dtype=np.float64)),
            pc.gripper_m_to_norm(status.msg.value, self._gripper_config.policy_full_mm),
        )

    def command(self, policy_q: np.ndarray, gripper: float, safety: pc.SafetyFilter) -> np.ndarray:
        """Send one joint and gripper target. Returns the real-robot joint target in radians."""
        real_q = safety.clamp_to_limits(self._config.calibration.policy_to_real(policy_q))
        if not self._send_commands:
            return real_q
        # Position-velocity joint control, smoothed by the arm controller.
        self._robot.move_j([float(v) for v in real_q])
        width_m = pc.gripper_norm_to_m(
            gripper, self._gripper_config.policy_full_mm, self._gripper_config.hardware_max_mm
        )
        self._gripper.move_gripper_m(value=width_m, force=self._gripper_config.force_n)
        return real_q


class RealSenseCamera:
    """Background reader that keeps the latest RGB frame of one RealSense camera."""

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30) -> None:
        import pyrealsense2 as rs  # noqa: PLC0415 - optional hardware dependency

        self._serial = serial
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipeline.start(config)
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while self.latest() is None:
            if time.monotonic() > deadline:
                raise RuntimeError(f"No frames from RealSense camera {serial}.")
            time.sleep(0.01)
        logger.info("RealSense %s streaming %dx%d RGB.", serial, width, height)

    def _loop(self) -> None:
        while not self._stop.is_set():
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data()).copy()
            with self._lock:
                self._latest = image

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._pipeline.stop()


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    left_arm: ArmConfig
    right_arm: ArmConfig
    gripper: GripperConfig = dataclasses.field(default_factory=GripperConfig)
    # RealSense serial numbers, keyed by the camera names used in the RoboSynChallenge datasets.
    camera_serials: dict[str, str] = dataclasses.field(default_factory=dict)
    image_width: int = 640
    image_height: int = 480
    speed_percent: int = 100
    # Extra keyword arguments for pyAgxArm's create_agx_arm_config, e.g. for testing.
    sdk_config_overrides: dict = dataclasses.field(default_factory=dict)


CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class CobotMagicPiper:
    """Dual-arm robot with the 14-dim RoboSynChallenge state and action layout."""

    def __init__(self, config: RobotConfig, *, send_commands: bool) -> None:
        missing = [name for name in CAMERA_NAMES if name not in config.camera_serials]
        if missing:
            raise ValueError(f"Missing RealSense serial numbers for {missing}.")
        self._left, self._right = (
            PiperArm(
                arm,
                config.gripper,
                send_commands=send_commands,
                speed_percent=config.speed_percent,
                sdk_config_overrides=config.sdk_config_overrides,
            )
            for arm in (config.left_arm, config.right_arm)
        )
        self._cameras = {
            name: RealSenseCamera(config.camera_serials[name], config.image_width, config.image_height)
            for name in CAMERA_NAMES
        }

    def read_state(self) -> np.ndarray:
        left_q, left_g = self._left.read()
        right_q, right_g = self._right.read()
        return pc.join_state(left_q, left_g, right_q, right_g)

    def read_images(self) -> dict[str, np.ndarray]:
        images = {}
        for name, camera in self._cameras.items():
            image = camera.latest()
            if image is None:
                raise RuntimeError(f"Camera {name} has no frame.")
            images[name] = image
        return images

    def command(self, target: np.ndarray, safety: pc.SafetyFilter) -> None:
        left_q, left_g, right_q, right_g = pc.split_state(target)
        self._left.command(left_q, left_g, safety)
        self._right.command(right_q, right_g, safety)

    def close(self) -> None:
        # The arms are left enabled on purpose: disabling a PiPER releases its motors
        # and raised joints fall under gravity. Park the arms before powering them off.
        for camera in self._cameras.values():
            camera.close()
