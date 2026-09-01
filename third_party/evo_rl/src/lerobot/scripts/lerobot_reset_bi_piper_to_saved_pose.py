#!/usr/bin/env python

"""Slowly reset a bimanual Piper-X follower to a saved joint pose, then disable it."""

import argparse
import json
import time
from pathlib import Path

from lerobot.robots.bi_piper_follower import BiPiperXFollowerConfig
from lerobot.robots.piper_follower import PiperFollowerConfigBase
from lerobot.robots.utils import make_robot_from_config


DEFAULT_POSE_PATH = Path(
    "/home/phl/.cache/huggingface/lerobot/failure_reset_pose/"
    "bi_piperx_follower_phone_handover_bimanual_follower.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slowly move the bimanual Piper-X follower to a saved reset pose, then disable it."
    )
    parser.add_argument("--pose-path", type=Path, default=DEFAULT_POSE_PATH)
    parser.add_argument("--robot-id", default="phone_handover_bimanual_follower")
    parser.add_argument("--left-port", default="can_l_follower")
    parser.add_argument("--right-port", default="can_r_follower")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed-ratio", type=int, default=10)
    parser.add_argument(
        "--keep-enabled-after-reset",
        action="store_true",
        help="Keep arms enabled after reset. Default is to disable arms on disconnect.",
    )
    return parser.parse_args()


def load_target_pose(path: Path) -> dict[str, float]:
    with open(path) as f:
        payload = json.load(f)
    joint_pos = payload["joint_pos"] if isinstance(payload, dict) and "joint_pos" in payload else payload
    return {str(key): float(value) for key, value in joint_pos.items() if str(key).endswith(".pos")}


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0:
        raise ValueError(f"--duration-s must be positive, got {args.duration_s}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    target = load_target_pose(args.pose_path)
    disable_on_disconnect = not args.keep_enabled_after_reset

    cfg = BiPiperXFollowerConfig(
        id=args.robot_id,
        left_arm_config=PiperFollowerConfigBase(
            port=args.left_port,
            require_calibration=False,
            speed_ratio=args.speed_ratio,
            cameras={},
            disable_on_disconnect=disable_on_disconnect,
        ),
        right_arm_config=PiperFollowerConfigBase(
            port=args.right_port,
            require_calibration=False,
            speed_ratio=args.speed_ratio,
            cameras={},
            disable_on_disconnect=disable_on_disconnect,
        ),
    )

    robot = make_robot_from_config(cfg)
    robot.connect(calibrate=False)

    try:
        observation = robot.get_observation()
        joint_keys = [key for key in robot.action_features if key.endswith(".pos") and key in target]
        if not joint_keys:
            raise RuntimeError("No matching joint keys found between robot action features and saved pose.")

        start = {key: float(observation[key]) for key in joint_keys}
        steps = max(int(args.duration_s * args.fps), 1)
        dt_s = 1.0 / args.fps
        print(
            f"Resetting {len(joint_keys)} joints to {args.pose_path} "
            f"over {args.duration_s:.1f}s at {args.fps}Hz."
        )

        for idx in range(1, steps + 1):
            loop_start_t = time.perf_counter()
            alpha = idx / steps
            action = {key: start[key] + (target[key] - start[key]) * alpha for key in joint_keys}
            robot.send_action(action)
            elapsed_s = time.perf_counter() - loop_start_t
            time.sleep(max(dt_s - elapsed_s, 0.0))

        print(
            "Reset pose reached. Disconnecting now; "
            f"disable_on_disconnect={disable_on_disconnect}."
        )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
