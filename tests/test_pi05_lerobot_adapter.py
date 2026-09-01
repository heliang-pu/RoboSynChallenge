"""Observation/action plumbing of the LeRobot pi0.5 adapter.

Covers the parts that run before the policy is ever loaded, so the tests need
neither a checkpoint nor a LeRobot install.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from policy.pi05_lerobot.deploy_policy import (  # noqa: E402
    encode_action,
    encode_obs,
    resolve_image_keys,
)

ACTION_DIM = 14
GRIPPERS = (6, 13)


class FakeActionSpace:
    shape = (ACTION_DIM,)
    low = np.concatenate([np.full(6, -3.14), [0.0], np.full(6, -3.14), [0.0]]).astype(np.float32)
    high = np.concatenate([np.full(6, 3.14), [0.08], np.full(6, 3.14), [0.08]]).astype(np.float32)


class FakeEnv:
    class unwrapped:  # noqa: N801 - mirrors the gym attribute name
        single_action_space = FakeActionSpace()
        device = "cpu"


def make_obs(height=48, width=64, state_dim=ACTION_DIM):
    sensors = {
        name: {"color": np.full((1, height, width, 3), value, dtype=np.uint8)}
        for value, name in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist"), start=1)
    }
    return {"sensor": sensors, "robot": {"qpos": np.arange(state_dim, dtype=np.float32)[None]}}


# -- camera naming --------------------------------------------------------


def test_resolve_image_keys_robosyn_names():
    keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation.state",
    ]
    assert resolve_image_keys(keys) == {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_left_wrist": "cam_left_wrist",
        "observation.images.cam_right_wrist": "cam_right_wrist",
    }


def test_resolve_image_keys_openpi_names():
    keys = [
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    ]
    assert list(resolve_image_keys(keys).values()) == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]


def test_resolve_image_keys_falls_back_to_declaration_order():
    """camera1/2/3 carry no orientation, so training order decides the mapping."""
    keys = [
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    ]
    assert resolve_image_keys(keys) == {
        "observation.images.camera1": "cam_high",
        "observation.images.camera2": "cam_left_wrist",
        "observation.images.camera3": "cam_right_wrist",
    }


def test_resolve_image_keys_rejects_more_cameras_than_the_env_has():
    keys = [f"observation.images.camera{i}" for i in range(4)]
    with pytest.raises(ValueError, match="only exposes"):
        resolve_image_keys(keys)


def test_resolve_image_keys_requires_a_camera():
    with pytest.raises(ValueError, match="No image features"):
        resolve_image_keys(["observation.state"])


# -- observation encoding -------------------------------------------------


def test_encode_obs_emits_only_policy_inputs_as_chw_uint8():
    image_keys = resolve_image_keys(
        [f"observation.images.{name}" for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
    )
    encoded = encode_obs(make_obs(), image_keys=image_keys)

    assert set(encoded) == {"observation.state", *image_keys}
    for key in image_keys:
        assert encoded[key].shape == (3, 48, 64), key
        assert encoded[key].dtype == np.uint8, key
    # The env hands back a leading env dimension; the policy wants it gone.
    assert encoded["observation.state"].shape == (ACTION_DIM,)


def test_encode_obs_keeps_cameras_distinct():
    image_keys = resolve_image_keys(
        [f"observation.images.{name}" for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
    )
    encoded = encode_obs(make_obs(), image_keys=image_keys)
    first_pixels = {key: int(value[0, 0, 0]) for key, value in encoded.items() if key in image_keys}
    assert len(set(first_pixels.values())) == 3, first_pixels


# -- action encoding ------------------------------------------------------


def test_encode_action_maps_unit_gripper_onto_the_env_range():
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[GRIPPERS[0]] = 1.0  # fully open in 0-1 dataset units
    action[GRIPPERS[1]] = 0.0

    encoded = encode_action(action, FakeEnv(), rescale_gripper=True, gripper_indices=GRIPPERS)

    assert encoded.shape == (1, ACTION_DIM)
    assert encoded[0, GRIPPERS[0]].item() == pytest.approx(FakeActionSpace.high[GRIPPERS[0]])
    assert encoded[0, GRIPPERS[1]].item() == pytest.approx(FakeActionSpace.low[GRIPPERS[1]])


def test_encode_action_leaves_arm_joints_untouched():
    action = np.linspace(-1.0, 1.0, ACTION_DIM).astype(np.float32)
    encoded = encode_action(action, FakeEnv(), rescale_gripper=True, gripper_indices=GRIPPERS)
    arm = [i for i in range(ACTION_DIM) if i not in GRIPPERS]
    assert torch.allclose(encoded[0, arm], torch.from_numpy(action[arm]))


def test_encode_action_without_rescale_is_a_passthrough():
    action = np.linspace(0.0, 1.0, ACTION_DIM).astype(np.float32)
    encoded = encode_action(action, FakeEnv(), rescale_gripper=False, gripper_indices=GRIPPERS)
    assert torch.allclose(encoded[0], torch.from_numpy(action))


def test_encode_action_truncates_a_padded_policy_action():
    """PI0.5 pads actions to max_action_dim; the env only accepts its own width."""
    padded = np.zeros(32, dtype=np.float32)
    encoded = encode_action(padded, FakeEnv(), rescale_gripper=False, gripper_indices=GRIPPERS)
    assert encoded.shape == (1, ACTION_DIM)


def test_encode_action_rejects_a_short_action():
    with pytest.raises(ValueError, match="but env expects"):
        encode_action(np.zeros(7, dtype=np.float32), FakeEnv(), rescale_gripper=False)
