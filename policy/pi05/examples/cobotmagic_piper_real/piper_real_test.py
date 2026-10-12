import numpy as np
import pytest

from examples.cobotmagic_piper_real import env as _env
from examples.cobotmagic_piper_real import piper_conversions as pc


class FakeRobot:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.float32)
        self.commands = []

    def read_state(self):
        return self.state.copy()

    def read_images(self):
        return {
            name: np.full((480, 640, 3), 128, np.uint8) for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        }

    def command(self, target, safety):
        self.commands.append(np.array(target))
        self.state = np.asarray(target, dtype=np.float32)


def test_calibration_round_trip_and_validation():
    cal = pc.ArmCalibration(signs=(1, -1, 1, 1, -1, 1), offsets_rad=(0, 0.1, 0, 0, 0, -0.2))
    q = np.array([0.1, 0.2, -0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(cal.real_to_policy(cal.policy_to_real(q)), q)
    with pytest.raises(ValueError, match="signs"):
        pc.ArmCalibration(signs=(1, 1, 1, 1, 1, 0.5))


def test_gripper_conversion_for_70mm_gripper():
    assert pc.gripper_norm_to_m(0.5, 70.0, 70.0) == pytest.approx(0.035)
    assert pc.gripper_norm_to_m(1.5, 70.0, 70.0) == pytest.approx(0.070)
    # A policy trained on unscaled V100 data asks for up to 100 mm; the hardware caps it.
    assert pc.gripper_norm_to_m(1.0, 100.0, 70.0) == pytest.approx(0.070)
    assert pc.gripper_m_to_norm(0.035, 70.0) == pytest.approx(0.5)
    assert pc.gripper_m_to_norm(0.2, 70.0) == pytest.approx(1.0)


def test_safety_filter_clamps_limits_and_steps():
    safety = pc.SafetyFilter(max_joint_step_rad=0.1, joint_limit_margin_rad=0.0)
    clamped = safety.clamp_to_limits(np.array([5.0, -1.0, 1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(clamped[:3], [pc.PIPER_JOINT_LIMITS_RAD[0, 1], 0.0, 0.0])
    limited, largest = safety.limit_step(np.array([0.5, 0.05]), np.zeros(2))
    np.testing.assert_allclose(limited, [0.1, 0.05])
    assert largest == pytest.approx(0.5)


def test_interpolate_path_ends_at_goal():
    path = pc.interpolate_path(np.zeros(3), np.ones(3), 4)
    assert path.shape == (4, 3)
    np.testing.assert_allclose(path[0], 0.25)
    np.testing.assert_allclose(path[-1], 1.0)


def test_environment_observation_matches_training_keys():
    robot = FakeRobot(np.arange(14) / 100.0)
    env = _env.CobotMagicPiperEnvironment(robot, "Click the bell", safety=pc.SafetyFilter(), reset_state=None)
    env.reset()
    obs = env.get_observation()
    assert set(obs) == {
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
        "observation/state",
        "prompt",
    }
    assert obs["observation/image"].shape == (224, 224, 3)
    assert obs["observation/image"].dtype == np.uint8
    np.testing.assert_allclose(obs["observation/state"], np.arange(14) / 100.0)
    assert obs["prompt"] == "Click the bell"


def test_environment_limits_jumps_and_clips_gripper():
    robot = FakeRobot(np.zeros(14))
    env = _env.CobotMagicPiperEnvironment(robot, "x", safety=pc.SafetyFilter(max_joint_step_rad=0.1), reset_state=None)
    env.reset()
    action = np.zeros(14)
    action[0] = 1.0  # left joint 1 jump
    action[pc.LEFT_GRIPPER] = 1.7
    action[pc.RIGHT_GRIPPER] = -0.2
    env.apply_action({"actions": action})
    sent = robot.commands[-1]
    assert sent[0] == pytest.approx(0.1)
    assert sent[pc.LEFT_GRIPPER] == pytest.approx(1.0)
    assert sent[pc.RIGHT_GRIPPER] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="action dims"):
        env.apply_action({"actions": np.zeros(7)})


def test_environment_reset_moves_smoothly_to_reset_pose():
    start = np.zeros(14)
    start[1] = 0.6
    robot = FakeRobot(start)
    goal = np.zeros(14)
    env = _env.CobotMagicPiperEnvironment(
        robot,
        "x",
        safety=pc.SafetyFilter(max_joint_step_rad=0.1),
        reset_state=goal,
        reset_duration_s=0.1,
        control_hz=100.0,
    )
    env.reset()
    steps = np.array(robot.commands)
    assert len(steps) == 10
    assert np.max(np.abs(np.diff(steps[:, 1]))) <= 0.1 + 1e-9
    np.testing.assert_allclose(steps[-1], goal, atol=1e-9)
