"""Render the simulator at real-robot joint states and overlay the camera views.

Capture snapshots on the robot with
``policy/pi05/examples/cobotmagic_piper_real/capture_snapshot.py``. Each
snapshot holds the 14-dim joint state and the RGB images of the three cameras.
This script puts the simulated CobotMagic in the same joint state and writes,
per snapshot and camera, an image with three panels: real, simulated, and a
50/50 blend.

If the arm in the blend panel lines up, the joint zero pose, the joint
directions and the camera calibration agree. A rigid shift of the whole image
points to camera extrinsics. An arm that bends the wrong way points to a joint
sign.

    python scripts/sim2real/overlay_real_sim.py \
        --gym_config configs/click_bell/clear/gym_config.json \
        --action_config configs/click_bell/action_config.json \
        --snapshots snapshots/*.npz --out_dir sim2real_overlay

Use the ``clear`` config so that camera and lighting randomization stay off.
"""

from __future__ import annotations

import argparse
import pathlib

import gymnasium as gym
import numpy as np
from PIL import Image
import torch

import robosynchallenge  # noqa: F401 - registers the RoboSynChallenge tasks.
from robosynchallenge.replay import convert_state_action_trajectory

from embodichain.lab.gym.envs.wrapper import ReplayWrapper
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
import embodichain.lab.gym.utils.gym_utils as gym_utils
from embodichain.utils.logger import log_info, log_warning

gym_utils.DEFAULT_MANAGER_MODULES = gym_utils.DEFAULT_MANAGER_MODULES + [
    "robosynchallenge.managers.actions",
    "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events",
    "robosynchallenge.managers.observations",
]

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
GRIPPER_DIMS = (6, 13)


def real_state_to_sim(
    state: np.ndarray, real_gripper_full_mm: float, sim_gripper_full_mm: float
) -> np.ndarray:
    """Convert a snapshot state to the simulator convention.

    The arm joints are shared. The gripper value is rescaled from the real
    gripper's full opening to the simulated V100 gripper's full opening.
    """
    sim = np.array(state, dtype=np.float32, copy=True).reshape(-1)
    sim[list(GRIPPER_DIMS)] = np.clip(
        sim[list(GRIPPER_DIMS)] * real_gripper_full_mm / sim_gripper_full_mm, 0.0, 1.0
    )
    return sim


def to_uint8_image(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[0]
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * (255.0 if image.max() <= 1.0 else 1.0), 0, 255)
    return image.astype(np.uint8)


def compose(real: np.ndarray, sim: np.ndarray, alpha: float) -> np.ndarray:
    height, width = sim.shape[:2]
    if real.shape[:2] != (height, width):
        real = np.asarray(Image.fromarray(real).resize((width, height), Image.BILINEAR))
    blend = (
        alpha * real.astype(np.float32) + (1.0 - alpha) * sim.astype(np.float32)
    ).astype(np.uint8)
    return np.concatenate([real, sim, blend], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--snapshots",
        nargs="+",
        required=True,
        help="Snapshot .npz files from capture_snapshot.py.",
    )
    parser.add_argument(
        "--out_dir",
        default="sim2real_overlay",
        help="Output directory for the overlay images.",
    )
    parser.add_argument(
        "--real_gripper_full_mm",
        type=float,
        default=None,
        help="Opening of the real gripper value 1.0. Defaults to the value stored in each snapshot, else 70.",
    )
    parser.add_argument(
        "--sim_gripper_full_mm",
        type=float,
        default=100.0,
        help="Opening of the simulated gripper value 1.0.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight of the real image in the blend panel.",
    )
    args = parser.parse_args()
    args.filter_dataset_saving = True
    args.filter_visual_rand = True

    snapshots = [np.load(path) for path in args.snapshots]
    sim_states = []
    for path, snap in zip(args.snapshots, snapshots, strict=True):
        full_mm = args.real_gripper_full_mm
        if full_mm is None:
            full_mm = (
                float(snap["gripper_full_mm"]) if "gripper_full_mm" in snap else 70.0
            )
        sim_states.append(
            real_state_to_sim(snap["state"], full_mm, args.sim_gripper_full_mm)
        )
        log_info(f"{path}: real state {np.round(snap['state'], 3)}")
    states = torch.as_tensor(np.stack(sim_states), dtype=torch.float32)

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        trajectory = convert_state_action_trajectory(
            {"state": states, "action": states}, env.unwrapped, mode="kinematic"
        )
        replay = ReplayWrapper(env.unwrapped, trajectory, mode="kinematic")
        replay.reset()
        for index, (path, snap) in enumerate(
            zip(args.snapshots, snapshots, strict=True)
        ):
            obs = replay.go_to_step(index)
            for camera in CAMERAS:
                if camera not in snap:
                    log_warning(f"{path} has no {camera} image; skipping it.")
                    continue
                sim_image = to_uint8_image(obs["sensor"][camera]["color"])
                panel = compose(np.asarray(snap[camera]), sim_image, args.alpha)
                target = out_dir / f"{pathlib.Path(path).stem}_{camera}.png"
                Image.fromarray(panel).save(target)
                log_info(f"Wrote {target}")
    finally:
        try:
            env.close()
        finally:
            from embodichain.lab.sim.sim_manager import SimulationManager

            SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
