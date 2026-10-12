"""Capture real-robot snapshots for checking the sim-to-real alignment.

Each snapshot stores the 14-dim joint state in the policy convention and the
full-resolution RGB images of the three cameras. Feed the files to
``scripts/sim2real/overlay_real_sim.py`` to render the simulator at the same
joint state and overlay both views.

The arms are only read, never commanded. Move them by hand or with the leader
arms between snapshots.

    python -m examples.cobotmagic_piper_real.capture_snapshot \
        --cam-high-serial 123 --cam-left-wrist-serial 456 --cam-right-wrist-serial 789 \
        --out-dir snapshots
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import time

import numpy as np
import tyro

from examples.cobotmagic_piper_real import hardware as _hardware
from examples.cobotmagic_piper_real import main as _main


@dataclasses.dataclass
class Args:
    # Serial number of the top D435.
    cam_high_serial: str
    # Serial number of the left wrist D435.
    cam_left_wrist_serial: str
    # Serial number of the right wrist D435.
    cam_right_wrist_serial: str
    # Directory for the snapshot files.
    out_dir: pathlib.Path = pathlib.Path("snapshots")
    # CAN channel of the left follower arm.
    left_channel: str = "can_left"
    # CAN channel of the right follower arm.
    right_channel: str = "can_right"
    # python-can interface used by pyAgxArm.
    can_interface: str = "socketcan"
    # pyAgxArm PiperFW profile.
    firmware: str = "default"
    # Opening that the gripper value 1.0 stands for, in mm.
    gripper_full_mm: float = 70.0


def main(args: Args) -> None:
    robot_args = _main.Args(
        prompt="",
        cam_high_serial=args.cam_high_serial,
        cam_left_wrist_serial=args.cam_left_wrist_serial,
        cam_right_wrist_serial=args.cam_right_wrist_serial,
        left_channel=args.left_channel,
        right_channel=args.right_channel,
        can_interface=args.can_interface,
        firmware=args.firmware,
        gripper_full_mm=args.gripper_full_mm,
    )
    robot = _hardware.CobotMagicPiper(_main.build_robot_config(robot_args), send_commands=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        index = 0
        while True:
            line = input(f"[{index}] Enter to capture, q to quit: ").strip().lower()
            if line == "q":
                break
            state = robot.read_state()
            images = robot.read_images()
            path = args.out_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}_{index:03d}.npz"
            np.savez_compressed(path, state=state, gripper_full_mm=args.gripper_full_mm, **images)
            logging.info("Saved %s, state %s", path, np.round(state, 3))
            index += 1
    finally:
        robot.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
