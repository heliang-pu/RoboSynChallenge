"""Run a pi0 / pi0.5 policy server on the dual-arm CobotMagic PiPER robot.

Start the policy server first, for example:

    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_robosynchallenge_sim2real \
        --policy.dir=checkpoints/pi05_robosynchallenge_sim2real/my_run/20000

Then run this client from ``policy/pi05`` on the robot computer:

    python -m examples.cobotmagic_piper_real.main --prompt "Click the bell" \
        --cam-high-serial 123 --cam-left-wrist-serial 456 --cam-right-wrist-serial 789 --dry-run

Drop ``--dry-run`` once the printed actions look right.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.cobotmagic_piper_real import env as _env
from examples.cobotmagic_piper_real import hardware as _hardware
from examples.cobotmagic_piper_real import piper_conversions as pc


@dataclasses.dataclass
class Args:
    # Language instruction, e.g. the task prompt used in the training data.
    prompt: str
    # Serial number of the top D435. List serials with `rs-enumerate-devices -s`.
    cam_high_serial: str
    # Serial number of the left wrist D435.
    cam_left_wrist_serial: str
    # Serial number of the right wrist D435.
    cam_right_wrist_serial: str

    # Policy server address.
    host: str = "127.0.0.1"
    # Policy server port.
    port: int = 8000

    # Actions executed from each predicted chunk before querying the policy again.
    # The simulation evaluation in this repository uses 10.
    action_horizon: int = 10
    # Control rate. The RoboSynChallenge datasets are recorded at 25 Hz.
    control_hz: float = 25.0
    # Number of episodes to run.
    num_episodes: int = 1
    # Control steps per episode before the client stops.
    max_episode_steps: int = 1500

    # CAN channel of the left follower arm.
    left_channel: str = "can_left"
    # CAN channel of the right follower arm.
    right_channel: str = "can_right"
    # python-can interface used by pyAgxArm.
    can_interface: str = "socketcan"
    # pyAgxArm PiperFW profile: default (<= S-V1.8-2), v183, v188 or v189 (>= S-V1.8-9).
    firmware: str = "default"
    # Arm speed percentage for position-velocity joint control.
    speed_percent: int = 100

    # Joint mapping real = sign * policy + offset. Keep the defaults when the
    # simulated zero pose matches the real one and single-joint jogs agree.
    left_joint_signs: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # Joint direction signs of the right arm.
    right_joint_signs: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # Joint zero offsets of the left arm, in radians.
    left_joint_offsets: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Joint zero offsets of the right arm, in radians.
    right_joint_offsets: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Opening that the policy's gripper value 1.0 stands for. Use 70 with the
    # pi05_robosynchallenge_sim2real config, and 100 for a policy trained only on
    # unscaled simulated data.
    gripper_full_mm: float = 70.0
    # Largest opening the gripper hardware accepts, in mm.
    gripper_hardware_max_mm: float = 70.0
    # Gripping force in N, range 0-3.
    gripper_force_n: float = 1.0

    # Largest joint change per control step, in radians.
    max_joint_step_rad: float = 0.1

    # Where each episode starts: "zero" moves to the zero pose that the simulated
    # robot starts near, "current" starts wherever the arms are.
    reset_pose: str = "zero"
    # Gripper value at the reset pose. The simulated episodes start closed.
    reset_gripper: float = 0.0
    # Time to move to the reset pose, in seconds.
    reset_duration_s: float = 3.0

    # Read sensors and query the policy, but do not move the arms.
    dry_run: bool = False


def build_robot_config(args: Args) -> _hardware.RobotConfig:
    def arm(channel, signs, offsets):
        return _hardware.ArmConfig(
            channel=channel,
            calibration=pc.ArmCalibration(signs=tuple(signs), offsets_rad=tuple(offsets)),
            firmware=args.firmware,
            interface=args.can_interface,
        )

    return _hardware.RobotConfig(
        left_arm=arm(args.left_channel, args.left_joint_signs, args.left_joint_offsets),
        right_arm=arm(args.right_channel, args.right_joint_signs, args.right_joint_offsets),
        gripper=_hardware.GripperConfig(
            policy_full_mm=args.gripper_full_mm,
            hardware_max_mm=args.gripper_hardware_max_mm,
            force_n=args.gripper_force_n,
        ),
        camera_serials={
            "cam_high": args.cam_high_serial,
            "cam_left_wrist": args.cam_left_wrist_serial,
            "cam_right_wrist": args.cam_right_wrist_serial,
        },
        speed_percent=args.speed_percent,
    )


def reset_state(args: Args) -> np.ndarray | None:
    if args.reset_pose == "current":
        return None
    if args.reset_pose != "zero":
        raise ValueError(f"Unknown reset pose {args.reset_pose!r}; use 'zero' or 'current'.")
    return pc.join_state(np.zeros(6), args.reset_gripper, np.zeros(6), args.reset_gripper)


class _LoggingPolicy:
    """Logs every chunk the server returns, which helps spot unit mismatches early."""

    def __init__(self, policy):
        self._policy = policy

    def infer(self, obs):
        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"])
        logging.info(
            "Chunk %s | state %s | first action %s",
            actions.shape,
            np.round(obs["observation/state"], 3),
            np.round(actions[0], 3),
        )
        return result

    def reset(self):
        self._policy.reset()


def main(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", client.get_server_metadata())

    robot = _hardware.CobotMagicPiper(build_robot_config(args), send_commands=not args.dry_run)
    environment = _env.CobotMagicPiperEnvironment(
        robot,
        args.prompt,
        safety=pc.SafetyFilter(max_joint_step_rad=args.max_joint_step_rad),
        reset_state=reset_state(args),
        reset_duration_s=args.reset_duration_s,
        control_hz=args.control_hz,
    )
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_LoggingPolicy(client),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.control_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    try:
        runtime.run()
    finally:
        robot.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
