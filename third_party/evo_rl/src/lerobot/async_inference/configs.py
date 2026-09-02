# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


def _validate_xy_string(value: str, *, field_name: str) -> None:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{field_name} must be formatted as 'x,y', got {value!r}")
    float(parts[0])
    float(parts[1])


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    enable_action_safety_limits: bool = field(
        default=True,
        metadata={
            "help": (
                "Clamp Piper-style joint targets before robot.send_action using per-joint absolute "
                "limits and per-cycle speed limits."
            )
        },
    )
    action_safety_log_interval_s: float = field(
        default=1.0,
        metadata={"help": "Minimum seconds between action safety clipping log messages."},
    )
    return_home_on_stop: bool = field(
        default=False,
        metadata={"help": "Move the robot to a saved pose when the async client stops."},
    )
    return_home_pose_path: str = field(
        default="/home/phl/workspace/Evo-RL/configs/poses/phone_slot_ep55_frame30_start_pose.json",
        metadata={"help": "JSON pose file used when return_home_on_stop is enabled."},
    )
    return_home_delay_s: float = field(
        default=10.0,
        metadata={"help": "Seconds to wait after stopping inference before returning to the saved pose."},
    )
    return_home_duration_s: float = field(
        default=8.0,
        metadata={"help": "Seconds used to interpolate from current robot state to the saved pose."},
    )
    stop_on_pose: bool = field(
        default=False,
        metadata={"help": "Stop async inference when robot feedback is close to a saved pose."},
    )
    stop_pose_path: str = field(
        default="/home/phl/workspace/Evo-RL/configs/poses/phone_slot_ep55_final_right_arm_slot_above_pose.json",
        metadata={"help": "JSON pose file used as the stop detector target when stop_on_pose is enabled."},
    )
    stop_pose_tolerance_deg: float = field(
        default=3.0,
        metadata={"help": "Joint tolerance for stop-on-pose detection."},
    )
    stop_pose_gripper_tolerance: float = field(
        default=5.0,
        metadata={"help": "Gripper tolerance for stop-on-pose detection."},
    )
    stop_pose_stable_frames: int = field(
        default=30,
        metadata={"help": "Number of consecutive control frames inside tolerance before stopping."},
    )
    stop_pose_min_runtime_s: float = field(
        default=5.0,
        metadata={"help": "Minimum async runtime before stop-on-pose detection can trigger."},
    )
    stop_pose_dry_run: bool = field(
        default=True,
        metadata={
            "help": (
                "Evaluate and log stop-on-pose status without stopping inference. "
                "Useful for manually ending several runs while collecting tail-state diagnostics."
            )
        },
    )
    stop_pose_tail_frames: int = field(
        default=100,
        metadata={"help": "Number of recent right-arm feedback frames to save on stop."},
    )
    stop_pose_tail_output_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "CSV path for recent stop-pose diagnostics. If omitted, a timestamped file is written "
                "under outputs/async_inference."
            )
        },
    )
    auto_cycle_on_pose: bool = field(
        default=False,
        metadata={
            "help": (
                "When the robot reaches a saved final pose and stays stable, release the gripper, "
                "return to a saved start pose, then resume async inference for repeated trials."
            )
        },
    )
    auto_cycle_pose_path: str = field(
        default="/home/phl/workspace/Evo-RL/configs/poses/phone_slot_dataset_tail_stop_pose.json",
        metadata={
            "help": (
                "JSON target-pose file for auto-cycle final-pose detection. It may contain joint_pos, "
                "tolerances, and stable_delta_tolerances."
            )
        },
    )
    auto_cycle_start_pose_path: str = field(
        default="/home/phl/workspace/Evo-RL/configs/poses/phone_slot_ep55_frame30_start_pose.json",
        metadata={"help": "JSON pose file used as the repeated-trial start pose."},
    )
    auto_cycle_max_cycles: int = field(
        default=10,
        metadata={"help": "Number of autonomous inference cycles to run before stopping."},
    )
    auto_cycle_stable_s: float = field(
        default=2.0,
        metadata={"help": "Seconds the final pose must remain stable before auto-cycling."},
    )
    auto_cycle_min_runtime_s: float = field(
        default=5.0,
        metadata={"help": "Minimum runtime in each cycle before final-pose detection can trigger."},
    )
    auto_cycle_wait_before_release_s: float = field(
        default=5.0,
        metadata={"help": "Seconds to hold the final pose before opening the gripper."},
    )
    auto_cycle_release_gripper_keys: str = field(
        default="right_gripper.pos",
        metadata={"help": "Comma-separated gripper action keys to open during auto-cycle release."},
    )
    auto_cycle_release_gripper_pos: float = field(
        default=90.0,
        metadata={"help": "Open-gripper target used during auto-cycle release."},
    )
    auto_cycle_release_command_s: float = field(
        default=1.0,
        metadata={"help": "Seconds to repeatedly command the gripper-open target."},
    )
    auto_cycle_wait_after_release_s: float = field(
        default=5.0,
        metadata={"help": "Seconds to wait after gripper release before returning home."},
    )
    auto_cycle_return_duration_s: float = field(
        default=3.0,
        metadata={"help": "Seconds used to interpolate back to the auto-cycle start pose."},
    )
    auto_cycle_restart_delay_s: float = field(
        default=2.0,
        metadata={"help": "Seconds to wait at the start pose before resuming inference."},
    )
    handoff_fixed_insert: bool = field(
        default=False,
        metadata={
            "help": (
                "When enabled, detect a loose VLA handoff by right-arm stability and run the "
                "phone-slot fixed insert primitive inside this robot client."
            )
        },
    )
    handoff_fixed_insert_repeat: bool = field(
        default=False,
        metadata={
            "help": (
                "When enabled, return control to VLA after each fixed-insert primitive instead of stopping "
                "after the first handoff."
            )
        },
    )
    handoff_fixed_insert_resume_delay_s: float = field(
        default=0.0,
        metadata={"help": "Optional delay after fixed-insert recovery before resuming VLA observations."},
    )
    handoff_min_runtime_s: float = field(
        default=8.0,
        metadata={"help": "Minimum VLA runtime before fixed-insert handoff detection can trigger."},
    )
    handoff_stable_s: float = field(
        default=1.0,
        metadata={"help": "Seconds of low right-arm joint motion required before handoff."},
    )
    handoff_stable_joint_delta_deg: float = field(
        default=0.8,
        metadata={"help": "Maximum per-frame right-arm joint feedback delta considered stable."},
    )
    handoff_gripper_key: str = field(
        default="right_gripper.pos",
        metadata={"help": "Observation key used to check that the right gripper still holds the phone."},
    )
    handoff_min_gripper_pos: float = field(
        default=6.0,
        metadata={"help": "Minimum gripper position for fixed-insert handoff detection."},
    )
    fixed_insert_pose_path: str = field(
        default="/home/phl/workspace/Evo-RL/artifacts/phone_slot_insert_ready_poses/insert_ready_002.json",
        metadata={"help": "Recorded insert-ready pose used after VLA handoff."},
    )
    fixed_insert_distance_m: float = field(
        default=0.12,
        metadata={"help": "Base/world-Z insertion distance after reaching the insert-ready pose."},
    )
    fixed_insert_pre_comp_lift_m: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional base/world-Z lift after reaching the insert-ready pose and before "
                "visual compensation. The straight insert starts from this lifted nominal height."
            )
        },
    )
    fixed_insert_pre_comp_lift_tol_m: float = field(
        default=0.005,
        metadata={
            "help": (
                "World-Z tolerance for considering the optional pre-compensation lift complete. "
                "This stage is a clearance move, so it only gates on FK Z rather than full pose error."
            )
        },
    )
    fixed_insert_approach_max_joint_step_deg: float = field(
        default=2.0,
        metadata={"help": "Maximum per-cycle joint step while moving to the insert-ready pose."},
    )
    fixed_insert_target_joint_tol_deg: float = field(
        default=1.0,
        metadata={"help": "Joint tolerance for considering the insert-ready pose reached."},
    )
    fixed_insert_approach_timeout_s: float = field(
        default=20.0,
        metadata={"help": "Timeout for moving to the insert-ready pose after VLA handoff."},
    )
    fixed_insert_insert_step_m: float = field(
        default=0.004,
        metadata={"help": "Waypoint spacing for straight fixed insertion."},
    )
    fixed_insert_insert_max_joint_step_deg: float = field(
        default=1.0,
        metadata={"help": "Maximum per-cycle joint step during straight fixed insertion."},
    )
    fixed_insert_insert_control_fps: float = field(
        default=20.0,
        metadata={"help": "Control frequency for straight fixed insertion."},
    )
    fixed_insert_linear_motion_backend: str = field(
        default="dls",
        metadata={
            "help": (
                "Backend for post-approach Cartesian fixed-insert moves. "
                "'dls' uses the local stepwise IK controller; 'pilz' requests PILZ LIN plans "
                "from MoveIt and executes the returned joint trajectory through the current robot client."
            )
        },
    )
    fixed_insert_pilz_planner_script: str = field(
        default="/home/phl/workspace/Evo-RL/scripts/tools/plan_pilz_lin_trajectory.py",
        metadata={"help": "System-Python ROS helper used to request a plan-only PILZ LIN trajectory."},
    )
    fixed_insert_pilz_setup_commands: str = field(
        default="source /opt/ros/jazzy/setup.bash && source /home/phl/workspace/fmc3_robotics_ws/install/setup.bash",
        metadata={"help": "Shell setup commands run before invoking the PILZ planner helper."},
    )
    fixed_insert_pilz_move_group_action: str = field(
        default="/move_action",
        metadata={"help": "MoveIt MoveGroup action name used by the PILZ planner helper."},
    )
    fixed_insert_pilz_group_name: str = field(
        default="arm",
        metadata={"help": "MoveIt planning group used for PILZ fixed-insert LIN motions."},
    )
    fixed_insert_pilz_base_frame: str = field(
        default="base_link",
        metadata={"help": "MoveIt planning frame for PILZ fixed-insert target poses."},
    )
    fixed_insert_pilz_tip_frame: str = field(
        default="link6",
        metadata={"help": "MoveIt tip/link constrained by PILZ fixed-insert target poses."},
    )
    fixed_insert_pilz_allowed_planning_time_s: float = field(
        default=3.0,
        metadata={"help": "MoveIt allowed planning time for each PILZ LIN request."},
    )
    fixed_insert_pilz_timeout_s: float = field(
        default=8.0,
        metadata={"help": "Wall-clock timeout for each PILZ LIN planner helper call."},
    )
    fixed_insert_pilz_num_planning_attempts: int = field(
        default=1,
        metadata={"help": "Number of planning attempts for each PILZ LIN request."},
    )
    fixed_insert_pilz_max_velocity_scaling: float = field(
        default=0.2,
        metadata={"help": "MoveIt velocity scaling factor for PILZ LIN plans."},
    )
    fixed_insert_pilz_max_acceleration_scaling: float = field(
        default=0.2,
        metadata={"help": "MoveIt acceleration scaling factor for PILZ LIN plans."},
    )
    fixed_insert_pilz_execute_time_scale: float = field(
        default=1.0,
        metadata={"help": "Multiplier applied to PILZ trajectory point timing during local execution."},
    )
    fixed_insert_pilz_min_point_dt_s: float = field(
        default=0.01,
        metadata={"help": "Minimum delay between locally executed PILZ trajectory points."},
    )
    fixed_insert_pilz_insert_quintic_interpolation: bool = field(
        default=False,
        metadata={
            "help": (
                "Use quintic joint interpolation when locally executing the final PILZ down-insert trajectory."
            )
        },
    )
    fixed_insert_pilz_insert_settle_timeout_s: float = field(
        default=3.0,
        metadata={
            "help": (
                "Maximum time to wait for non-final PILZ LIN motions such as pre-lift and visual compensation "
                "to reach their FK targets before continuing from the current FK pose."
            )
        },
    )
    fixed_insert_pilz_final_release_settle_timeout_s: float = field(
        default=0.1,
        metadata={
            "help": (
                "Maximum time to wait for the final PILZ down-insert trajectory to settle before releasing the "
                "gripper and returning."
            )
        },
    )
    fixed_insert_release_gripper: bool = field(
        default=True,
        metadata={"help": "Open the right gripper after the fixed insert primitive finishes."},
    )
    fixed_insert_release_gripper_key: str = field(
        default="right_gripper.pos",
        metadata={"help": "Gripper action key to open after fixed insert."},
    )
    fixed_insert_release_gripper_pos: float = field(
        default=90.0,
        metadata={"help": "Open-gripper target sent after fixed insert."},
    )
    fixed_insert_release_command_s: float = field(
        default=1.0,
        metadata={"help": "Seconds to repeatedly command the fixed-insert gripper release."},
    )
    fixed_insert_return_to_start_pose: bool = field(
        default=True,
        metadata={"help": "After fixed insert and gripper release, return both arms to the pose captured at client startup."},
    )
    fixed_insert_return_duration_s: float = field(
        default=8.0,
        metadata={"help": "Seconds used for quintic interpolation back to the fixed-insert startup pose."},
    )
    fixed_insert_head_rgb_compensation: bool = field(
        default=False,
        metadata={"help": "Enable one-shot head-RGB up/down slot-axis compensation before fixed insertion."},
    )
    fixed_insert_head_rgb_image_key: str = field(
        default="right_front",
        metadata={"help": "Raw robot observation image key for the head/front RGB camera."},
    )
    fixed_insert_head_rgb_slot_center_xy: str = field(
        default="",
        metadata={"help": "Fixed head-image slot center pixel as 'x,y'. Required when compensation is enabled."},
    )
    fixed_insert_head_rgb_slot_down_axis_xy: str = field(
        default="0,1",
        metadata={"help": "Fixed head-image slot long-axis down direction as 'x,y'."},
    )
    fixed_insert_head_rgb_base_down_axis_xy: str = field(
        default="0,1",
        metadata={"help": "Base/world XY direction corresponding to slot-axis down compensation as 'x,y'."},
    )
    fixed_insert_head_rgb_deadband_px: float = field(
        default=20.0,
        metadata={"help": "Head-image axial pixel deadband for no fixed compensation."},
    )
    fixed_insert_head_rgb_compensation_m: float = field(
        default=0.02,
        metadata={"help": "Fixed slot-axis compensation distance for up/down decisions."},
    )
    fixed_insert_head_rgb_compensation_sign: float = field(
        default=1.0,
        metadata={"help": "Sign multiplier for the base/world compensation direction."},
    )
    fixed_insert_head_rgb_min_red_area_px: float = field(
        default=1000.0,
        metadata={"help": "Minimum red phone component area in the head RGB image."},
    )
    fixed_insert_head_rgb_timeout_s: float = field(
        default=3.0,
        metadata={"help": "Timeout for executing the one-shot head-RGB compensation move."},
    )
    fixed_insert_head_rgb_max_joint_step_deg: float = field(
        default=0.8,
        metadata={"help": "Maximum per-cycle joint step during head-RGB compensation."},
    )
    fixed_insert_head_rgb_debug_output_dir: str = field(
        default="",
        metadata={
            "help": (
                "Optional directory for saving last.png and last.json diagnostics from the "
                "head-RGB compensation frame."
            )
        },
    )
    fixed_insert_wrist_redline_compensation: bool = field(
        default=False,
        metadata={"help": "Enable one-shot right-wrist red-phone length compensation before fixed insertion."},
    )
    fixed_insert_wrist_redline_image_key: str = field(
        default="right_right_wrist",
        metadata={"help": "Raw robot observation image key for the right wrist RGB camera."},
    )
    fixed_insert_wrist_redline_up_pose_path: str = field(
        default="",
        metadata={"help": "Insert-ready JSON for the upper-grip boundary template."},
    )
    fixed_insert_wrist_redline_center_pose_path: str = field(
        default="",
        metadata={"help": "Insert-ready JSON for the centered template; this still applies center-up compensation."},
    )
    fixed_insert_wrist_redline_down_pose_path: str = field(
        default="",
        metadata={"help": "Insert-ready JSON for the lower-grip boundary template."},
    )
    fixed_insert_wrist_redline_base_down_axis_xy: str = field(
        default="0,1",
        metadata={"help": "Base/world XY direction corresponding to slot-axis down compensation as 'x,y'."},
    )
    fixed_insert_wrist_redline_deadband_px: float = field(
        default=12.0,
        metadata={"help": "Right-wrist redline pixel deadband around the center template."},
    )
    fixed_insert_wrist_redline_max_compensation_m: float = field(
        default=0.04,
        metadata={"help": "Maximum slot-axis compensation from the wrist redline classifier."},
    )
    fixed_insert_wrist_redline_center_up_compensation_m: float = field(
        default=0.02,
        metadata={
            "help": (
                "Slot-axis UP compensation applied at the center wrist redline template. "
                "The wrist redline mapping is linear through up=-max, center=-this value."
            )
        },
    )
    fixed_insert_wrist_redline_up_start_compensation_m: float = field(
        default=-0.02,
        metadata={
            "help": (
                "Signed slot-axis compensation where the wrist redline mapping leaves the center deadband "
                "toward the up template."
            )
        },
    )
    fixed_insert_wrist_redline_up_compensation_m: float = field(
        default=0.04,
        metadata={"help": "Slot-axis UP compensation magnitude applied at the up wrist redline template."},
    )
    fixed_insert_wrist_redline_down_start_compensation_m: float = field(
        default=-0.02,
        metadata={
            "help": (
                "Signed slot-axis compensation where the wrist redline mapping leaves the center deadband "
                "toward the down template."
            )
        },
    )
    fixed_insert_wrist_redline_down_compensation_m: float = field(
        default=0.04,
        metadata={"help": "Slot-axis DOWN compensation applied at the down wrist redline template."},
    )
    fixed_insert_wrist_redline_center_length_override_px: float = field(
        default=0.0,
        metadata={
            "help": (
                "If positive, shift the wrist redline up/center/down template lengths so this pixel length "
                "becomes the zero-compensation center."
            )
        },
    )
    fixed_insert_wrist_redline_compensation_sign: float = field(
        default=1.0,
        metadata={"help": "Sign multiplier for the wrist-redline base/world compensation direction."},
    )
    fixed_insert_wrist_redline_min_red_area_px: float = field(
        default=1000.0,
        metadata={"help": "Minimum red phone component area in the right wrist RGB image."},
    )
    fixed_insert_wrist_redline_timeout_s: float = field(
        default=3.0,
        metadata={"help": "Timeout for executing the one-shot wrist-redline compensation move."},
    )
    fixed_insert_wrist_redline_max_joint_step_deg: float = field(
        default=0.8,
        metadata={"help": "Maximum per-cycle joint step during wrist-redline compensation."},
    )
    fixed_insert_wrist_redline_debug_output_dir: str = field(
        default="",
        metadata={
            "help": (
                "Optional directory for saving last.png and last.json diagnostics from the "
                "wrist-redline compensation frame."
            )
        },
    )
    display_camera_views: bool = field(
        default=False,
        metadata={"help": "Show all camera images from the robot client in a local OpenCV window."},
    )
    display_camera_scale: float = field(
        default=0.4,
        metadata={"help": "Scale factor used when previewing camera images in the robot client."},
    )
    display_camera_window_name: str = field(
        default="Evo-RL async camera views",
        metadata={"help": "OpenCV window name used for async camera preview."},
    )

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )
    debug_pipeline_trace: bool = field(
        default=False,
        metadata={"help": "Print low-rate async inference pipeline trace logs for hardware debugging."},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.action_safety_log_interval_s < 0:
            raise ValueError(
                "action_safety_log_interval_s must be non-negative, "
                f"got {self.action_safety_log_interval_s}"
            )
        if self.return_home_delay_s < 0:
            raise ValueError(f"return_home_delay_s must be non-negative, got {self.return_home_delay_s}")
        if self.return_home_duration_s <= 0:
            raise ValueError(
                f"return_home_duration_s must be positive, got {self.return_home_duration_s}"
            )
        if self.stop_pose_tolerance_deg < 0:
            raise ValueError(
                f"stop_pose_tolerance_deg must be non-negative, got {self.stop_pose_tolerance_deg}"
            )
        if self.stop_pose_gripper_tolerance < 0:
            raise ValueError(
                f"stop_pose_gripper_tolerance must be non-negative, got {self.stop_pose_gripper_tolerance}"
            )
        if self.stop_pose_stable_frames <= 0:
            raise ValueError(
                f"stop_pose_stable_frames must be positive, got {self.stop_pose_stable_frames}"
            )
        if self.stop_pose_min_runtime_s < 0:
            raise ValueError(
                f"stop_pose_min_runtime_s must be non-negative, got {self.stop_pose_min_runtime_s}"
            )
        if self.stop_pose_tail_frames <= 0:
            raise ValueError(f"stop_pose_tail_frames must be positive, got {self.stop_pose_tail_frames}")
        if self.auto_cycle_max_cycles <= 0:
            raise ValueError(f"auto_cycle_max_cycles must be positive, got {self.auto_cycle_max_cycles}")
        if self.auto_cycle_stable_s <= 0:
            raise ValueError(f"auto_cycle_stable_s must be positive, got {self.auto_cycle_stable_s}")
        if self.auto_cycle_min_runtime_s < 0:
            raise ValueError(
                f"auto_cycle_min_runtime_s must be non-negative, got {self.auto_cycle_min_runtime_s}"
            )
        if self.auto_cycle_wait_before_release_s < 0:
            raise ValueError(
                "auto_cycle_wait_before_release_s must be non-negative, "
                f"got {self.auto_cycle_wait_before_release_s}"
            )
        if self.auto_cycle_release_command_s <= 0:
            raise ValueError(
                f"auto_cycle_release_command_s must be positive, got {self.auto_cycle_release_command_s}"
            )
        if self.auto_cycle_wait_after_release_s < 0:
            raise ValueError(
                "auto_cycle_wait_after_release_s must be non-negative, "
                f"got {self.auto_cycle_wait_after_release_s}"
            )
        if self.auto_cycle_return_duration_s <= 0:
            raise ValueError(
                f"auto_cycle_return_duration_s must be positive, got {self.auto_cycle_return_duration_s}"
            )
        if self.auto_cycle_restart_delay_s < 0:
            raise ValueError(
                f"auto_cycle_restart_delay_s must be non-negative, got {self.auto_cycle_restart_delay_s}"
            )
        if self.handoff_min_runtime_s < 0:
            raise ValueError(f"handoff_min_runtime_s must be non-negative, got {self.handoff_min_runtime_s}")
        if self.handoff_fixed_insert_resume_delay_s < 0:
            raise ValueError(
                "handoff_fixed_insert_resume_delay_s must be non-negative, "
                f"got {self.handoff_fixed_insert_resume_delay_s}"
            )
        if self.handoff_stable_s <= 0:
            raise ValueError(f"handoff_stable_s must be positive, got {self.handoff_stable_s}")
        if self.handoff_stable_joint_delta_deg < 0:
            raise ValueError(
                "handoff_stable_joint_delta_deg must be non-negative, "
                f"got {self.handoff_stable_joint_delta_deg}"
            )
        if not self.handoff_gripper_key:
            raise ValueError("handoff_gripper_key cannot be empty")
        if self.handoff_min_gripper_pos < 0:
            raise ValueError(f"handoff_min_gripper_pos must be non-negative, got {self.handoff_min_gripper_pos}")
        if self.fixed_insert_distance_m <= 0:
            raise ValueError(f"fixed_insert_distance_m must be positive, got {self.fixed_insert_distance_m}")
        if self.fixed_insert_pre_comp_lift_m < 0:
            raise ValueError(
                f"fixed_insert_pre_comp_lift_m must be non-negative, got {self.fixed_insert_pre_comp_lift_m}"
            )
        if self.fixed_insert_pre_comp_lift_tol_m < 0:
            raise ValueError(
                "fixed_insert_pre_comp_lift_tol_m must be non-negative, "
                f"got {self.fixed_insert_pre_comp_lift_tol_m}"
            )
        if self.fixed_insert_approach_max_joint_step_deg <= 0:
            raise ValueError(
                "fixed_insert_approach_max_joint_step_deg must be positive, "
                f"got {self.fixed_insert_approach_max_joint_step_deg}"
            )
        if self.fixed_insert_target_joint_tol_deg <= 0:
            raise ValueError(
                f"fixed_insert_target_joint_tol_deg must be positive, got {self.fixed_insert_target_joint_tol_deg}"
            )
        if self.fixed_insert_approach_timeout_s <= 0:
            raise ValueError(
                f"fixed_insert_approach_timeout_s must be positive, got {self.fixed_insert_approach_timeout_s}"
            )
        if self.fixed_insert_insert_step_m <= 0:
            raise ValueError(f"fixed_insert_insert_step_m must be positive, got {self.fixed_insert_insert_step_m}")
        if self.fixed_insert_insert_max_joint_step_deg <= 0:
            raise ValueError(
                "fixed_insert_insert_max_joint_step_deg must be positive, "
                f"got {self.fixed_insert_insert_max_joint_step_deg}"
            )
        if self.fixed_insert_insert_control_fps <= 0:
            raise ValueError(
                f"fixed_insert_insert_control_fps must be positive, got {self.fixed_insert_insert_control_fps}"
            )
        if self.fixed_insert_linear_motion_backend not in ("dls", "pilz"):
            raise ValueError(
                "fixed_insert_linear_motion_backend must be 'dls' or 'pilz', "
                f"got {self.fixed_insert_linear_motion_backend!r}"
            )
        if self.fixed_insert_linear_motion_backend == "pilz":
            if not self.fixed_insert_pilz_planner_script.strip():
                raise ValueError("fixed_insert_pilz_planner_script cannot be empty when PILZ backend is enabled")
            if not self.fixed_insert_pilz_group_name.strip():
                raise ValueError("fixed_insert_pilz_group_name cannot be empty when PILZ backend is enabled")
            if not self.fixed_insert_pilz_base_frame.strip():
                raise ValueError("fixed_insert_pilz_base_frame cannot be empty when PILZ backend is enabled")
            if not self.fixed_insert_pilz_tip_frame.strip():
                raise ValueError("fixed_insert_pilz_tip_frame cannot be empty when PILZ backend is enabled")
            if self.fixed_insert_pilz_allowed_planning_time_s <= 0:
                raise ValueError(
                    "fixed_insert_pilz_allowed_planning_time_s must be positive, "
                    f"got {self.fixed_insert_pilz_allowed_planning_time_s}"
                )
            if self.fixed_insert_pilz_timeout_s <= 0:
                raise ValueError(f"fixed_insert_pilz_timeout_s must be positive, got {self.fixed_insert_pilz_timeout_s}")
            if self.fixed_insert_pilz_num_planning_attempts <= 0:
                raise ValueError(
                    "fixed_insert_pilz_num_planning_attempts must be positive, "
                    f"got {self.fixed_insert_pilz_num_planning_attempts}"
                )
            if not 0.0 < self.fixed_insert_pilz_max_velocity_scaling <= 1.0:
                raise ValueError(
                    "fixed_insert_pilz_max_velocity_scaling must be in (0, 1], "
                    f"got {self.fixed_insert_pilz_max_velocity_scaling}"
                )
            if not 0.0 < self.fixed_insert_pilz_max_acceleration_scaling <= 1.0:
                raise ValueError(
                    "fixed_insert_pilz_max_acceleration_scaling must be in (0, 1], "
                    f"got {self.fixed_insert_pilz_max_acceleration_scaling}"
                )
            if self.fixed_insert_pilz_execute_time_scale <= 0:
                raise ValueError(
                    "fixed_insert_pilz_execute_time_scale must be positive, "
                    f"got {self.fixed_insert_pilz_execute_time_scale}"
                )
            if self.fixed_insert_pilz_min_point_dt_s < 0:
                raise ValueError(
                    "fixed_insert_pilz_min_point_dt_s must be non-negative, "
                    f"got {self.fixed_insert_pilz_min_point_dt_s}"
                )
            if self.fixed_insert_pilz_insert_settle_timeout_s <= 0:
                raise ValueError(
                    "fixed_insert_pilz_insert_settle_timeout_s must be positive, "
                    f"got {self.fixed_insert_pilz_insert_settle_timeout_s}"
                )
            if self.fixed_insert_pilz_final_release_settle_timeout_s <= 0:
                raise ValueError(
                    "fixed_insert_pilz_final_release_settle_timeout_s must be positive, "
                    f"got {self.fixed_insert_pilz_final_release_settle_timeout_s}"
                )
        if self.fixed_insert_release_gripper and not self.fixed_insert_release_gripper_key.strip():
            raise ValueError("fixed_insert_release_gripper_key cannot be empty when release is enabled")
        if self.fixed_insert_release_gripper_pos < 0:
            raise ValueError(
                f"fixed_insert_release_gripper_pos must be non-negative, got {self.fixed_insert_release_gripper_pos}"
            )
        if self.fixed_insert_release_command_s <= 0:
            raise ValueError(
                f"fixed_insert_release_command_s must be positive, got {self.fixed_insert_release_command_s}"
            )
        if self.fixed_insert_return_duration_s <= 0:
            raise ValueError(
                f"fixed_insert_return_duration_s must be positive, got {self.fixed_insert_return_duration_s}"
            )
        if self.fixed_insert_head_rgb_compensation:
            if not self.fixed_insert_head_rgb_image_key.strip():
                raise ValueError("fixed_insert_head_rgb_image_key cannot be empty when compensation is enabled")
            if not self.fixed_insert_head_rgb_slot_center_xy.strip():
                raise ValueError("fixed_insert_head_rgb_slot_center_xy is required when compensation is enabled")
            _validate_xy_string(
                self.fixed_insert_head_rgb_slot_center_xy,
                field_name="fixed_insert_head_rgb_slot_center_xy",
            )
        _validate_xy_string(
            self.fixed_insert_head_rgb_slot_down_axis_xy,
            field_name="fixed_insert_head_rgb_slot_down_axis_xy",
        )
        _validate_xy_string(
            self.fixed_insert_head_rgb_base_down_axis_xy,
            field_name="fixed_insert_head_rgb_base_down_axis_xy",
        )
        if self.fixed_insert_head_rgb_deadband_px < 0:
            raise ValueError(
                f"fixed_insert_head_rgb_deadband_px must be non-negative, got {self.fixed_insert_head_rgb_deadband_px}"
            )
        if self.fixed_insert_head_rgb_compensation_m <= 0:
            raise ValueError(
                "fixed_insert_head_rgb_compensation_m must be positive, "
                f"got {self.fixed_insert_head_rgb_compensation_m}"
            )
        if self.fixed_insert_head_rgb_compensation_sign not in (-1.0, 1.0):
            raise ValueError(
                "fixed_insert_head_rgb_compensation_sign must be -1.0 or 1.0, "
                f"got {self.fixed_insert_head_rgb_compensation_sign}"
            )
        if self.fixed_insert_head_rgb_min_red_area_px <= 0:
            raise ValueError(
                "fixed_insert_head_rgb_min_red_area_px must be positive, "
                f"got {self.fixed_insert_head_rgb_min_red_area_px}"
            )
        if self.fixed_insert_head_rgb_timeout_s <= 0:
            raise ValueError(
                f"fixed_insert_head_rgb_timeout_s must be positive, got {self.fixed_insert_head_rgb_timeout_s}"
            )
        if self.fixed_insert_head_rgb_max_joint_step_deg <= 0:
            raise ValueError(
                "fixed_insert_head_rgb_max_joint_step_deg must be positive, "
                f"got {self.fixed_insert_head_rgb_max_joint_step_deg}"
            )
        if self.fixed_insert_wrist_redline_compensation:
            if not self.fixed_insert_wrist_redline_image_key.strip():
                raise ValueError(
                    "fixed_insert_wrist_redline_image_key cannot be empty when compensation is enabled"
                )
            if not self.fixed_insert_wrist_redline_up_pose_path.strip():
                raise ValueError(
                    "fixed_insert_wrist_redline_up_pose_path is required when compensation is enabled"
                )
            if not self.fixed_insert_wrist_redline_center_pose_path.strip():
                raise ValueError(
                    "fixed_insert_wrist_redline_center_pose_path is required when compensation is enabled"
                )
            if not self.fixed_insert_wrist_redline_down_pose_path.strip():
                raise ValueError(
                    "fixed_insert_wrist_redline_down_pose_path is required when compensation is enabled"
                )
        _validate_xy_string(
            self.fixed_insert_wrist_redline_base_down_axis_xy,
            field_name="fixed_insert_wrist_redline_base_down_axis_xy",
        )
        if self.fixed_insert_wrist_redline_deadband_px < 0:
            raise ValueError(
                "fixed_insert_wrist_redline_deadband_px must be non-negative, "
                f"got {self.fixed_insert_wrist_redline_deadband_px}"
            )
        if self.fixed_insert_wrist_redline_max_compensation_m <= 0:
            raise ValueError(
                "fixed_insert_wrist_redline_max_compensation_m must be positive, "
                f"got {self.fixed_insert_wrist_redline_max_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_center_length_override_px < 0:
            raise ValueError(
                "fixed_insert_wrist_redline_center_length_override_px must be non-negative, "
                f"got {self.fixed_insert_wrist_redline_center_length_override_px}"
            )
        if self.fixed_insert_wrist_redline_center_up_compensation_m < 0:
            raise ValueError(
                "fixed_insert_wrist_redline_center_up_compensation_m must be non-negative, "
                f"got {self.fixed_insert_wrist_redline_center_up_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_center_up_compensation_m > self.fixed_insert_wrist_redline_max_compensation_m:
            raise ValueError(
                "fixed_insert_wrist_redline_center_up_compensation_m must be less than or equal to "
                "fixed_insert_wrist_redline_max_compensation_m, "
                f"got {self.fixed_insert_wrist_redline_center_up_compensation_m} > "
                f"{self.fixed_insert_wrist_redline_max_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_up_compensation_m <= 0:
            raise ValueError(
                "fixed_insert_wrist_redline_up_compensation_m must be positive, "
                f"got {self.fixed_insert_wrist_redline_up_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_up_compensation_m > self.fixed_insert_wrist_redline_max_compensation_m:
            raise ValueError(
                "fixed_insert_wrist_redline_up_compensation_m must be less than or equal to "
                "fixed_insert_wrist_redline_max_compensation_m, "
                f"got {self.fixed_insert_wrist_redline_up_compensation_m} > "
                f"{self.fixed_insert_wrist_redline_max_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_up_start_compensation_m < -self.fixed_insert_wrist_redline_up_compensation_m:
            raise ValueError(
                "fixed_insert_wrist_redline_up_start_compensation_m must be greater than or equal to "
                "-fixed_insert_wrist_redline_up_compensation_m, "
                f"got {self.fixed_insert_wrist_redline_up_start_compensation_m} < "
                f"{-self.fixed_insert_wrist_redline_up_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_up_start_compensation_m > 0:
            raise ValueError(
                "fixed_insert_wrist_redline_up_start_compensation_m must be less than or equal to 0, "
                f"got {self.fixed_insert_wrist_redline_up_start_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_down_start_compensation_m < -self.fixed_insert_wrist_redline_max_compensation_m:
            raise ValueError(
                "fixed_insert_wrist_redline_down_start_compensation_m must be greater than or equal to "
                "-fixed_insert_wrist_redline_max_compensation_m, "
                f"got {self.fixed_insert_wrist_redline_down_start_compensation_m} < "
                f"{-self.fixed_insert_wrist_redline_max_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_down_compensation_m <= 0:
            raise ValueError(
                "fixed_insert_wrist_redline_down_compensation_m must be positive, "
                f"got {self.fixed_insert_wrist_redline_down_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_down_start_compensation_m > self.fixed_insert_wrist_redline_down_compensation_m:
            raise ValueError(
                "fixed_insert_wrist_redline_down_start_compensation_m must be less than or equal to "
                "fixed_insert_wrist_redline_down_compensation_m, "
                f"got {self.fixed_insert_wrist_redline_down_start_compensation_m} > "
                f"{self.fixed_insert_wrist_redline_down_compensation_m}"
            )
        if self.fixed_insert_wrist_redline_compensation_sign not in (-1.0, 1.0):
            raise ValueError(
                "fixed_insert_wrist_redline_compensation_sign must be -1.0 or 1.0, "
                f"got {self.fixed_insert_wrist_redline_compensation_sign}"
            )
        if self.fixed_insert_wrist_redline_min_red_area_px <= 0:
            raise ValueError(
                "fixed_insert_wrist_redline_min_red_area_px must be positive, "
                f"got {self.fixed_insert_wrist_redline_min_red_area_px}"
            )
        if self.fixed_insert_wrist_redline_timeout_s <= 0:
            raise ValueError(
                "fixed_insert_wrist_redline_timeout_s must be positive, "
                f"got {self.fixed_insert_wrist_redline_timeout_s}"
            )
        if self.fixed_insert_wrist_redline_max_joint_step_deg <= 0:
            raise ValueError(
                "fixed_insert_wrist_redline_max_joint_step_deg must be positive, "
                f"got {self.fixed_insert_wrist_redline_max_joint_step_deg}"
            )
        if self.display_camera_scale <= 0:
            raise ValueError(f"display_camera_scale must be positive, got {self.display_camera_scale}")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "debug_pipeline_trace": self.debug_pipeline_trace,
            "aggregate_fn_name": self.aggregate_fn_name,
            "enable_action_safety_limits": self.enable_action_safety_limits,
            "action_safety_log_interval_s": self.action_safety_log_interval_s,
            "return_home_on_stop": self.return_home_on_stop,
            "return_home_pose_path": self.return_home_pose_path,
            "return_home_delay_s": self.return_home_delay_s,
            "return_home_duration_s": self.return_home_duration_s,
            "stop_on_pose": self.stop_on_pose,
            "stop_pose_path": self.stop_pose_path,
            "stop_pose_tolerance_deg": self.stop_pose_tolerance_deg,
            "stop_pose_gripper_tolerance": self.stop_pose_gripper_tolerance,
            "stop_pose_stable_frames": self.stop_pose_stable_frames,
            "stop_pose_min_runtime_s": self.stop_pose_min_runtime_s,
            "stop_pose_dry_run": self.stop_pose_dry_run,
            "stop_pose_tail_frames": self.stop_pose_tail_frames,
            "stop_pose_tail_output_path": self.stop_pose_tail_output_path,
            "auto_cycle_on_pose": self.auto_cycle_on_pose,
            "auto_cycle_pose_path": self.auto_cycle_pose_path,
            "auto_cycle_start_pose_path": self.auto_cycle_start_pose_path,
            "auto_cycle_max_cycles": self.auto_cycle_max_cycles,
            "auto_cycle_stable_s": self.auto_cycle_stable_s,
            "auto_cycle_min_runtime_s": self.auto_cycle_min_runtime_s,
            "auto_cycle_wait_before_release_s": self.auto_cycle_wait_before_release_s,
            "auto_cycle_release_gripper_keys": self.auto_cycle_release_gripper_keys,
            "auto_cycle_release_gripper_pos": self.auto_cycle_release_gripper_pos,
            "auto_cycle_release_command_s": self.auto_cycle_release_command_s,
            "auto_cycle_wait_after_release_s": self.auto_cycle_wait_after_release_s,
            "auto_cycle_return_duration_s": self.auto_cycle_return_duration_s,
            "auto_cycle_restart_delay_s": self.auto_cycle_restart_delay_s,
            "handoff_fixed_insert": self.handoff_fixed_insert,
            "handoff_fixed_insert_repeat": self.handoff_fixed_insert_repeat,
            "handoff_fixed_insert_resume_delay_s": self.handoff_fixed_insert_resume_delay_s,
            "handoff_min_runtime_s": self.handoff_min_runtime_s,
            "handoff_stable_s": self.handoff_stable_s,
            "handoff_stable_joint_delta_deg": self.handoff_stable_joint_delta_deg,
            "handoff_gripper_key": self.handoff_gripper_key,
            "handoff_min_gripper_pos": self.handoff_min_gripper_pos,
            "fixed_insert_pose_path": self.fixed_insert_pose_path,
            "fixed_insert_distance_m": self.fixed_insert_distance_m,
            "fixed_insert_pre_comp_lift_m": self.fixed_insert_pre_comp_lift_m,
            "fixed_insert_pre_comp_lift_tol_m": self.fixed_insert_pre_comp_lift_tol_m,
            "fixed_insert_approach_max_joint_step_deg": self.fixed_insert_approach_max_joint_step_deg,
            "fixed_insert_target_joint_tol_deg": self.fixed_insert_target_joint_tol_deg,
            "fixed_insert_approach_timeout_s": self.fixed_insert_approach_timeout_s,
            "fixed_insert_insert_step_m": self.fixed_insert_insert_step_m,
            "fixed_insert_insert_max_joint_step_deg": self.fixed_insert_insert_max_joint_step_deg,
            "fixed_insert_insert_control_fps": self.fixed_insert_insert_control_fps,
            "fixed_insert_linear_motion_backend": self.fixed_insert_linear_motion_backend,
            "fixed_insert_pilz_planner_script": self.fixed_insert_pilz_planner_script,
            "fixed_insert_pilz_setup_commands": self.fixed_insert_pilz_setup_commands,
            "fixed_insert_pilz_move_group_action": self.fixed_insert_pilz_move_group_action,
            "fixed_insert_pilz_group_name": self.fixed_insert_pilz_group_name,
            "fixed_insert_pilz_base_frame": self.fixed_insert_pilz_base_frame,
            "fixed_insert_pilz_tip_frame": self.fixed_insert_pilz_tip_frame,
            "fixed_insert_pilz_allowed_planning_time_s": self.fixed_insert_pilz_allowed_planning_time_s,
            "fixed_insert_pilz_timeout_s": self.fixed_insert_pilz_timeout_s,
            "fixed_insert_pilz_num_planning_attempts": self.fixed_insert_pilz_num_planning_attempts,
            "fixed_insert_pilz_max_velocity_scaling": self.fixed_insert_pilz_max_velocity_scaling,
            "fixed_insert_pilz_max_acceleration_scaling": self.fixed_insert_pilz_max_acceleration_scaling,
            "fixed_insert_pilz_execute_time_scale": self.fixed_insert_pilz_execute_time_scale,
            "fixed_insert_pilz_min_point_dt_s": self.fixed_insert_pilz_min_point_dt_s,
            "fixed_insert_pilz_insert_quintic_interpolation": (
                self.fixed_insert_pilz_insert_quintic_interpolation
            ),
            "fixed_insert_pilz_insert_settle_timeout_s": self.fixed_insert_pilz_insert_settle_timeout_s,
            "fixed_insert_pilz_final_release_settle_timeout_s": (
                self.fixed_insert_pilz_final_release_settle_timeout_s
            ),
            "fixed_insert_release_gripper": self.fixed_insert_release_gripper,
            "fixed_insert_release_gripper_key": self.fixed_insert_release_gripper_key,
            "fixed_insert_release_gripper_pos": self.fixed_insert_release_gripper_pos,
            "fixed_insert_release_command_s": self.fixed_insert_release_command_s,
            "fixed_insert_return_to_start_pose": self.fixed_insert_return_to_start_pose,
            "fixed_insert_return_duration_s": self.fixed_insert_return_duration_s,
            "fixed_insert_head_rgb_compensation": self.fixed_insert_head_rgb_compensation,
            "fixed_insert_head_rgb_image_key": self.fixed_insert_head_rgb_image_key,
            "fixed_insert_head_rgb_slot_center_xy": self.fixed_insert_head_rgb_slot_center_xy,
            "fixed_insert_head_rgb_slot_down_axis_xy": self.fixed_insert_head_rgb_slot_down_axis_xy,
            "fixed_insert_head_rgb_base_down_axis_xy": self.fixed_insert_head_rgb_base_down_axis_xy,
            "fixed_insert_head_rgb_deadband_px": self.fixed_insert_head_rgb_deadband_px,
            "fixed_insert_head_rgb_compensation_m": self.fixed_insert_head_rgb_compensation_m,
            "fixed_insert_head_rgb_compensation_sign": self.fixed_insert_head_rgb_compensation_sign,
            "fixed_insert_head_rgb_min_red_area_px": self.fixed_insert_head_rgb_min_red_area_px,
            "fixed_insert_head_rgb_timeout_s": self.fixed_insert_head_rgb_timeout_s,
            "fixed_insert_head_rgb_max_joint_step_deg": self.fixed_insert_head_rgb_max_joint_step_deg,
            "fixed_insert_head_rgb_debug_output_dir": self.fixed_insert_head_rgb_debug_output_dir,
            "fixed_insert_wrist_redline_compensation": self.fixed_insert_wrist_redline_compensation,
            "fixed_insert_wrist_redline_image_key": self.fixed_insert_wrist_redline_image_key,
            "fixed_insert_wrist_redline_up_pose_path": self.fixed_insert_wrist_redline_up_pose_path,
            "fixed_insert_wrist_redline_center_pose_path": self.fixed_insert_wrist_redline_center_pose_path,
            "fixed_insert_wrist_redline_down_pose_path": self.fixed_insert_wrist_redline_down_pose_path,
            "fixed_insert_wrist_redline_base_down_axis_xy": self.fixed_insert_wrist_redline_base_down_axis_xy,
            "fixed_insert_wrist_redline_deadband_px": self.fixed_insert_wrist_redline_deadband_px,
            "fixed_insert_wrist_redline_max_compensation_m": self.fixed_insert_wrist_redline_max_compensation_m,
            "fixed_insert_wrist_redline_center_up_compensation_m": self.fixed_insert_wrist_redline_center_up_compensation_m,
            "fixed_insert_wrist_redline_up_start_compensation_m": self.fixed_insert_wrist_redline_up_start_compensation_m,
            "fixed_insert_wrist_redline_up_compensation_m": self.fixed_insert_wrist_redline_up_compensation_m,
            "fixed_insert_wrist_redline_down_start_compensation_m": self.fixed_insert_wrist_redline_down_start_compensation_m,
            "fixed_insert_wrist_redline_down_compensation_m": self.fixed_insert_wrist_redline_down_compensation_m,
            "fixed_insert_wrist_redline_center_length_override_px": self.fixed_insert_wrist_redline_center_length_override_px,
            "fixed_insert_wrist_redline_compensation_sign": self.fixed_insert_wrist_redline_compensation_sign,
            "fixed_insert_wrist_redline_min_red_area_px": self.fixed_insert_wrist_redline_min_red_area_px,
            "fixed_insert_wrist_redline_timeout_s": self.fixed_insert_wrist_redline_timeout_s,
            "fixed_insert_wrist_redline_max_joint_step_deg": self.fixed_insert_wrist_redline_max_joint_step_deg,
            "fixed_insert_wrist_redline_debug_output_dir": self.fixed_insert_wrist_redline_debug_output_dir,
            "display_camera_views": self.display_camera_views,
            "display_camera_scale": self.display_camera_scale,
            "display_camera_window_name": self.display_camera_window_name,
        }
