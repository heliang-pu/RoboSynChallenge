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

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import argparse
import json
import math
import pickle  # nosec
import csv
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Queue
from typing import Any, TypeAlias

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_piper_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    piper_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)
from .handoff import HandoffStabilityConfig, HandoffStabilityDetector
from lerobot.utils.piper_sdk import unit_to_milli

PIPER_ASYNC_ABSOLUTE_ACTION_LIMITS: dict[str, tuple[float, float]] = {
    # Same limits as the validated safe replay script.
    "left_joint_1.pos": (-145.0, 145.0),
    "left_joint_3.pos": (-170.0, 5.0),
    "left_joint_4.pos": (-85.0, 85.0),
    "left_joint_5.pos": (-85.0, 85.0),
    "left_joint_6.pos": (-165.0, 165.0),
    "left_gripper.pos": (0.0, 90.0),
    "right_joint_1.pos": (-145.0, 145.0),
    "right_joint_3.pos": (-170.0, 5.0),
    "right_joint_4.pos": (-85.0, 85.0),
    "right_joint_5.pos": (-85.0, 85.0),
    "right_joint_6.pos": (-165.0, 165.0),
    "right_gripper.pos": (0.0, 90.0),
}

SpeedLimitDegPerS: TypeAlias = float | tuple[float, float]

PIPER_ASYNC_SPEED_LIMITS_DEG_PER_S: dict[str, SpeedLimitDegPerS] = {
    # Tuple values are (negative speed, positive speed) in deg/s.
    # Per-step delta bounds are these speeds divided by the async client fps.
    "left_joint_1.pos": (-60.0, 90.0),
    "left_joint_2.pos": (-90.0, 150.0),
    "left_joint_3.pos": (-150.0, 120.0),
    "left_joint_4.pos": (-180.0, 150.0),
    "left_joint_5.pos": (-60.0, 90.0),
    "left_joint_6.pos": (-120.0, 120.0),
    "left_gripper.pos": (-300.0, 300.0),
    "right_joint_1.pos": (-60.0, 90.0),
    "right_joint_2.pos": (-90.0, 180.0),
    "right_joint_3.pos": (-90.0, 90.0),
    "right_joint_4.pos": (-240.0, 180.0),
    "right_joint_5.pos": (-120.0, 90.0),
    "right_joint_6.pos": (-120.0, 180.0),
    "right_gripper.pos": (-300.0, 300.0),
}


def clip_piper_async_action(
    action: dict[str, float],
    observation: dict[str, Any],
    fps: int,
    absolute_limits: dict[str, tuple[float, float]] = PIPER_ASYNC_ABSOLUTE_ACTION_LIMITS,
    speed_limits_deg_per_s: dict[str, SpeedLimitDegPerS] = PIPER_ASYNC_SPEED_LIMITS_DEG_PER_S,
) -> tuple[dict[str, float], dict[str, float | int]]:
    """Clamp async policy targets by absolute range and current-state speed limits."""

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    clipped_action: dict[str, float] = {}
    num_abs_clipped = 0
    num_speed_clipped = 0
    max_abs_delta_before_clip = 0.0
    max_abs_delta_after_clip = 0.0

    for key, value in action.items():
        target = float(value)
        if not math.isfinite(target):
            raise RuntimeError(f"Unsafe async policy action for {key}: action={target}")

        if key in absolute_limits:
            low, high = absolute_limits[key]
            bounded_target = min(high, max(low, target))
            if bounded_target != target:
                num_abs_clipped += 1
            target = bounded_target

        if key in speed_limits_deg_per_s and key in observation:
            current = float(observation[key])
            if not math.isfinite(current):
                raise RuntimeError(f"Unsafe async robot observation for {key}: observation={current}")

            speed_limit = speed_limits_deg_per_s[key]
            if isinstance(speed_limit, tuple):
                min_speed, max_speed = speed_limit
                if min_speed > 0 or max_speed < 0 or min_speed > max_speed:
                    raise ValueError(
                        f"Invalid directional speed limit for {key}: expected "
                        f"(negative_speed, positive_speed), got {speed_limit}"
                    )
                min_delta = min_speed / float(fps)
                max_delta = max_speed / float(fps)
            else:
                max_delta = abs(speed_limit) / float(fps)
                min_delta = -max_delta

            delta = target - current
            clipped_delta = min(max_delta, max(min_delta, delta))
            max_abs_delta_before_clip = max(max_abs_delta_before_clip, abs(delta))
            max_abs_delta_after_clip = max(max_abs_delta_after_clip, abs(clipped_delta))
            if clipped_delta != delta:
                num_speed_clipped += 1
            target = current + clipped_delta

        clipped_action[key] = target

    return clipped_action, {
        "num_abs_clipped": num_abs_clipped,
        "num_speed_clipped": num_speed_clipped,
        "max_abs_delta_before_clip": max_abs_delta_before_clip,
        "max_abs_delta_after_clip": max_abs_delta_after_clip,
    }


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing
        self._last_action_safety_log_t = 0.0
        self._start_time_s = time.perf_counter()
        self._stop_pose_stable_count = 0
        self._stop_pose_target = self._load_pose_file(config.stop_pose_path) if config.stop_on_pose else {}
        self._stop_pose_tail_buffer = deque(maxlen=config.stop_pose_tail_frames)
        (
            self._auto_cycle_target,
            self._auto_cycle_tolerances,
            self._auto_cycle_stable_delta_tolerances,
        ) = (
            self._load_pose_target_spec(config.auto_cycle_pose_path)
            if config.auto_cycle_on_pose
            else ({}, {}, {})
        )
        self._auto_cycle_stable_count = 0
        self._auto_cycle_count = 0
        self._auto_cycle_cycle_start_t = self._start_time_s
        self._auto_cycle_previous_pose: dict[str, float] | None = None
        self._auto_cycle_recovery_active = False
        self._handoff_fixed_insert_active = False
        self._handoff_fixed_insert_done = False
        self._handoff_vla_started_s: float | None = None
        self._handoff_fixed_insert_detector = HandoffStabilityDetector(
            HandoffStabilityConfig(
                enabled=config.handoff_fixed_insert,
                fps=config.fps,
                min_runtime_s=config.handoff_min_runtime_s,
                stable_s=config.handoff_stable_s,
                stable_joint_delta_deg=config.handoff_stable_joint_delta_deg,
                gripper_key=config.handoff_gripper_key,
                min_gripper_pos=config.handoff_min_gripper_pos,
            ),
            start_time_s=self._start_time_s,
        )
        self._skip_return_home_on_stop_once = False
        self._camera_display_warning_logged = False
        self._pipeline_trace_last_t: dict[str, float] = {}
        self._fixed_insert_start_pose = (
            self._capture_current_joint_pose() if config.fixed_insert_return_to_start_pose else {}
        )

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def _trace_pipeline(self, key: str, message: str, *args, min_interval_s: float = 1.0) -> None:
        if not self.config.debug_pipeline_trace:
            return

        now_s = time.perf_counter()
        if now_s - self._pipeline_trace_last_t.get(key, 0.0) < min_interval_s:
            return
        self._pipeline_trace_last_t[key] = now_s
        self.logger.info("[pipeline] " + message, *args)

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self._write_stop_pose_tail_if_needed()

        if self.config.return_home_on_stop and not self._skip_return_home_on_stop_once:
            self._return_robot_to_home_pose()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def _load_pose_file(self, path: str) -> dict[str, float]:
        pose_path = Path(path).expanduser()
        with open(pose_path) as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "target_pose" in payload:
            joint_pos = payload["target_pose"]
        elif isinstance(payload, dict) and "joint_pos" in payload:
            joint_pos = payload["joint_pos"]
        else:
            joint_pos = payload
        return {str(key): float(value) for key, value in joint_pos.items() if str(key).endswith(".pos")}

    def _load_pose_target_spec(
        self,
        path: str,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        pose_path = Path(path).expanduser()
        with open(pose_path) as f:
            payload = json.load(f)

        if "target_pose" in payload:
            joint_pos = payload["target_pose"]
        elif "joint_pos" in payload:
            joint_pos = payload["joint_pos"]
        else:
            joint_pos = payload

        target = {str(key): float(value) for key, value in joint_pos.items() if str(key).endswith(".pos")}
        raw_tolerances = payload.get("tolerances", {}) if isinstance(payload, dict) else {}
        raw_stable_delta_tolerances = (
            payload.get("stable_delta_tolerances", {}) if isinstance(payload, dict) else {}
        )

        tolerances = {
            key: float(raw_tolerances.get(key, 5.0 if "gripper" in key else 3.0)) for key in target
        }
        stable_delta_tolerances = {
            key: float(raw_stable_delta_tolerances.get(key, 1.0 if "gripper" in key else 0.5))
            for key in target
        }
        return target, tolerances, stable_delta_tolerances

    def _load_return_home_pose(self) -> dict[str, float]:
        return self._load_pose_file(self.config.return_home_pose_path)

    def _flush_action_queue(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()
        self.must_go.set()

    def _resolve_stop_pose_tail_output_path(self) -> Path:
        if self.config.stop_pose_tail_output_path:
            path = Path(self.config.stop_pose_tail_output_path).expanduser()
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = Path("outputs/async_inference") / f"stop_pose_tail_{timestamp}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_stop_pose_tail_if_needed(self) -> None:
        if not self.config.stop_on_pose or not self._stop_pose_tail_buffer:
            return

        path = self._resolve_stop_pose_tail_output_path()
        rows = list(self._stop_pose_tail_buffer)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.logger.info("Saved stop-pose tail diagnostics to %s", path)

    def _sleep_with_shutdown(self, duration_s: float) -> None:
        end_t = time.perf_counter() + max(duration_s, 0.0)
        while time.perf_counter() < end_t and self.running:
            time.sleep(min(0.1, end_t - time.perf_counter()))

    @staticmethod
    def _interpolation_alpha(alpha: float, interpolation: str) -> float:
        alpha = min(1.0, max(0.0, float(alpha)))
        if interpolation == "linear":
            return alpha
        if interpolation == "quintic":
            return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    def _capture_current_joint_pose(self) -> dict[str, float]:
        observation = self._get_action_safety_observation()
        pose: dict[str, float] = {}
        for key in self.robot.action_features:
            if key not in observation or "gripper" in key:
                continue
            pose[key] = float(observation[key])
        return pose

    def _move_robot_to_pose(
        self,
        target: dict[str, float],
        *,
        pose_label: str,
        duration_s: float,
        interpolation: str = "linear",
    ) -> bool:
        observation = self.robot.get_observation()
        joint_keys = [key for key in self.robot.action_features if key in target and key in observation]
        if not joint_keys:
            self.logger.warning("%s skipped: no matching pose keys found.", pose_label)
            return False

        self.logger.info(
            "%s: moving %d joints over %.2fs.",
            pose_label,
            len(joint_keys),
            duration_s,
        )

        start = {key: float(observation[key]) for key in joint_keys}
        steps = max(int(duration_s * self.config.fps), 1)
        dt_s = 1.0 / self.config.fps
        for idx in range(1, steps + 1):
            loop_start_t = time.perf_counter()
            alpha = self._interpolation_alpha(idx / steps, interpolation)
            action = {key: start[key] + (target[key] - start[key]) * alpha for key in joint_keys}
            action, safety_info = clip_piper_async_action(
                action,
                self._get_action_safety_observation(),
                self.config.fps,
            )
            if safety_info["num_abs_clipped"] or safety_info["num_speed_clipped"]:
                now = time.perf_counter()
                if now - self._last_action_safety_log_t >= self.config.action_safety_log_interval_s:
                    self.logger.warning(
                        "%s safety clipped targets: absolute=%s speed=%s, max delta %.2f -> %.2f.",
                        pose_label,
                        safety_info["num_abs_clipped"],
                        safety_info["num_speed_clipped"],
                        safety_info["max_abs_delta_before_clip"],
                        safety_info["max_abs_delta_after_clip"],
                    )
                    self._last_action_safety_log_t = now
            self.robot.send_action(action)
            elapsed_s = time.perf_counter() - loop_start_t
            time.sleep(max(dt_s - elapsed_s, 0.0))

        self.logger.info("%s reached.", pose_label)
        return True

    def _return_fixed_insert_to_start_pose(self) -> None:
        if not self.config.fixed_insert_return_to_start_pose:
            return
        if not self._fixed_insert_start_pose:
            self.logger.warning("Fixed insert return-start skipped: startup pose was not captured.")
            return
        self._move_robot_to_pose(
            self._fixed_insert_start_pose,
            pose_label="Fixed insert return-start",
            duration_s=self.config.fixed_insert_return_duration_s,
            interpolation="quintic",
        )

    def _return_robot_to_home_pose(self) -> None:
        try:
            target = self._load_return_home_pose()

            self.logger.info(
                "Return-home enabled: waiting %.2fs, then moving to %s over %.2fs.",
                self.config.return_home_delay_s,
                self.config.return_home_pose_path,
                self.config.return_home_duration_s,
            )
            time.sleep(self.config.return_home_delay_s)
            self._move_robot_to_pose(
                target,
                pose_label="Return-home",
                duration_s=self.config.return_home_duration_s,
            )
        except Exception as exc:
            self.logger.exception("Return-home failed; disconnecting robot without returning home: %s", exc)

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")
            self._trace_pipeline("send_observation_done", "sent observation timestep #%s", obs_timestep)

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                if self._auto_cycle_recovery_active or self._handoff_fixed_insert_active:
                    continue

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start
                self._trace_pipeline(
                    "received_actions",
                    "received %d actions from bridge in %.1fms",
                    len(timed_actions),
                    deserialize_time * 1000,
                )

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        if self._handoff_fixed_insert_active:
            return False
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def _get_action_safety_observation(self) -> dict[str, Any]:
        """Read joint-only observations for action safety without grabbing camera frames every step."""

        if hasattr(self.robot, "left_arm") and hasattr(self.robot, "right_arm"):
            left_arm = getattr(self.robot, "left_arm")
            right_arm = getattr(self.robot, "right_arm")
            if hasattr(left_arm, "_read_raw_observation") and hasattr(right_arm, "_read_raw_observation"):
                left_obs = left_arm._read_raw_observation()
                right_obs = right_arm._read_raw_observation()
                return {
                    **{f"left_{key}": value for key, value in left_obs.items()},
                    **{f"right_{key}": value for key, value in right_obs.items()},
                }

        if hasattr(self.robot, "_read_raw_observation"):
            return self.robot._read_raw_observation()

        return self.robot.get_observation()

    def _maybe_display_camera_views(self, raw_observation: RawObservation) -> None:
        if not self.config.display_camera_views:
            return

        image_items = [
            (key, value)
            for key, value in raw_observation.items()
            if key.startswith("observation.images.")
        ]
        if not image_items:
            return

        try:
            import cv2
            import numpy as np

            panels = []
            for key, image in sorted(image_items):
                image_array = np.asarray(image)
                if image_array.ndim != 3:
                    continue
                if image_array.shape[0] in (1, 3) and image_array.shape[-1] not in (1, 3):
                    image_array = np.moveaxis(image_array, 0, -1)
                if image_array.dtype != np.uint8:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                if image_array.shape[-1] == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                scale = self.config.display_camera_scale
                if scale != 1.0:
                    image_array = cv2.resize(
                        image_array,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                label = key.removeprefix("observation.images.")
                cv2.putText(
                    image_array,
                    label,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(image_array)

            if not panels:
                return

            min_height = min(panel.shape[0] for panel in panels)
            resized_panels = [
                cv2.resize(
                    panel,
                    (int(panel.shape[1] * min_height / panel.shape[0]), min_height),
                    interpolation=cv2.INTER_AREA,
                )
                for panel in panels
            ]
            preview = np.concatenate(resized_panels, axis=1)
            cv2.imshow(self.config.display_camera_window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.logger.info("Camera preview received 'q'; stopping client.")
                self.shutdown_event.set()
        except Exception as exc:
            if not self._camera_display_warning_logged:
                self.logger.warning("Camera preview disabled after display error: %s", exc)
                self._camera_display_warning_logged = True

    def _pose_target_status(
        self,
        raw_observation: RawObservation,
        target: dict[str, float],
        tolerances: dict[str, float],
    ) -> tuple[bool, float, list[str], dict[str, float]]:
        max_error = 0.0
        all_within_tolerance = True
        missing_keys: list[str] = []
        target_errors: dict[str, float] = {}
        for key, target_value in target.items():
            if key not in raw_observation:
                missing_keys.append(key)
                all_within_tolerance = False
                continue
            current = float(raw_observation[key])
            error = target_value - current
            target_errors[key] = error
            max_error = max(max_error, abs(error))
            if abs(error) > tolerances[key]:
                all_within_tolerance = False
        return all_within_tolerance, max_error, missing_keys, target_errors

    def _pose_delta_is_stable(
        self,
        raw_observation: RawObservation,
        previous_pose: dict[str, float] | None,
        target: dict[str, float],
        stable_delta_tolerances: dict[str, float],
    ) -> tuple[bool, float]:
        if previous_pose is None:
            return False, math.inf

        max_delta = 0.0
        all_stable = True
        for key in target:
            if key not in raw_observation or key not in previous_pose:
                return False, math.inf
            delta = abs(float(raw_observation[key]) - previous_pose[key])
            max_delta = max(max_delta, delta)
            if delta > stable_delta_tolerances[key]:
                all_stable = False
        return all_stable, max_delta

    def _send_auto_cycle_gripper_release(self) -> None:
        gripper_keys = [
            key.strip() for key in self.config.auto_cycle_release_gripper_keys.split(",") if key.strip()
        ]
        if not gripper_keys:
            self.logger.warning("Auto-cycle release skipped: no gripper keys configured.")
            return

        steps = max(int(self.config.auto_cycle_release_command_s * self.config.fps), 1)
        dt_s = 1.0 / self.config.fps
        self.logger.info(
            "Auto-cycle: opening gripper keys %s to %.2f for %.2fs.",
            gripper_keys,
            self.config.auto_cycle_release_gripper_pos,
            self.config.auto_cycle_release_command_s,
        )
        for _ in range(steps):
            loop_start_t = time.perf_counter()
            action = {key: self.config.auto_cycle_release_gripper_pos for key in gripper_keys}
            if self.config.enable_action_safety_limits:
                action, _ = clip_piper_async_action(
                    action,
                    self._get_action_safety_observation(),
                    self.config.fps,
                )
            self.robot.send_action(action)
            elapsed_s = time.perf_counter() - loop_start_t
            time.sleep(max(dt_s - elapsed_s, 0.0))

    def _run_auto_cycle_sequence(self) -> None:
        self._auto_cycle_recovery_active = True
        self._flush_action_queue()
        self._auto_cycle_count += 1
        cycle_idx = self._auto_cycle_count
        try:
            self.logger.info(
                "Auto-cycle %d/%d reached final pose. Holding %.2fs before release.",
                cycle_idx,
                self.config.auto_cycle_max_cycles,
                self.config.auto_cycle_wait_before_release_s,
            )
            self._sleep_with_shutdown(self.config.auto_cycle_wait_before_release_s)
            if not self.running:
                return

            self._send_auto_cycle_gripper_release()
            self.logger.info(
                "Auto-cycle %d/%d: released gripper, waiting %.2fs.",
                cycle_idx,
                self.config.auto_cycle_max_cycles,
                self.config.auto_cycle_wait_after_release_s,
            )
            self._sleep_with_shutdown(self.config.auto_cycle_wait_after_release_s)
            if not self.running:
                return

            start_pose = self._load_pose_file(self.config.auto_cycle_start_pose_path)
            self._move_robot_to_pose(
                start_pose,
                pose_label=f"Auto-cycle {cycle_idx}/{self.config.auto_cycle_max_cycles} return-start",
                duration_s=self.config.auto_cycle_return_duration_s,
            )

            if cycle_idx >= self.config.auto_cycle_max_cycles:
                self.logger.info(
                    "Auto-cycle max cycles reached (%d). Stopping client at start pose.",
                    self.config.auto_cycle_max_cycles,
                )
                self._skip_return_home_on_stop_once = True
                self.shutdown_event.set()
                return

            self.logger.info(
                "Auto-cycle %d/%d: waiting %.2fs before resuming inference.",
                cycle_idx,
                self.config.auto_cycle_max_cycles,
                self.config.auto_cycle_restart_delay_s,
            )
            self._sleep_with_shutdown(self.config.auto_cycle_restart_delay_s)
        finally:
            self._flush_action_queue()
            self._auto_cycle_stable_count = 0
            self._auto_cycle_previous_pose = None
            self._auto_cycle_cycle_start_t = time.perf_counter()
            self._auto_cycle_recovery_active = False

    def _maybe_auto_cycle_on_pose(self, raw_observation: RawObservation) -> bool:
        if (
            not self.config.auto_cycle_on_pose
            or not self._auto_cycle_target
            or self._auto_cycle_recovery_active
        ):
            return False

        current_pose = {
            key: float(raw_observation[key])
            for key in self._auto_cycle_target
            if key in raw_observation
        }
        runtime_s = time.perf_counter() - self._auto_cycle_cycle_start_t
        min_runtime_reached = runtime_s >= self.config.auto_cycle_min_runtime_s
        within_tolerance, max_error, missing_keys, _target_errors = self._pose_target_status(
            raw_observation,
            self._auto_cycle_target,
            self._auto_cycle_tolerances,
        )
        stable_delta, max_delta = self._pose_delta_is_stable(
            raw_observation,
            self._auto_cycle_previous_pose,
            self._auto_cycle_target,
            self._auto_cycle_stable_delta_tolerances,
        )
        self._auto_cycle_previous_pose = current_pose

        if missing_keys:
            self.logger.warning(
                "Auto-cycle target has keys missing from robot observation: %s", missing_keys
            )
            self._auto_cycle_stable_count = 0
            return False

        should_count = within_tolerance and stable_delta and min_runtime_reached
        if should_count:
            self._auto_cycle_stable_count += 1
        else:
            self._auto_cycle_stable_count = 0

        stable_required = max(int(self.config.auto_cycle_stable_s * self.config.fps), 1)
        if self._auto_cycle_stable_count == 1 or self._auto_cycle_stable_count % self.config.fps == 0:
            self.logger.info(
                "Auto-cycle final-pose check: stable=%d/%d max_error=%.2f max_delta=%.2f "
                "within_pose=%s stable_delta=%s min_runtime=%s cycle=%d/%d.",
                self._auto_cycle_stable_count,
                stable_required,
                max_error,
                max_delta,
                within_tolerance,
                stable_delta,
                min_runtime_reached,
                self._auto_cycle_count + 1,
                self.config.auto_cycle_max_cycles,
            )

        if self._auto_cycle_stable_count >= stable_required:
            self._run_auto_cycle_sequence()
            return True

        return False

    def _make_fixed_insert_args(self):
        return argparse.Namespace(
            side="right",
            pose=Path(self.config.fixed_insert_pose_path).expanduser(),
            robot_type=self.config.robot.type,
            target_frame="link6",
            urdf_path=None,
            joint_name=None,
            approach_control_fps=float(self.config.fps),
            approach_max_joint_step_deg=float(self.config.fixed_insert_approach_max_joint_step_deg),
            approach_timeout_s=float(self.config.fixed_insert_approach_timeout_s),
            target_joint_tol_deg=float(self.config.fixed_insert_target_joint_tol_deg),
            insert_distance_m=float(self.config.fixed_insert_distance_m),
            insert_step_m=float(self.config.fixed_insert_insert_step_m),
            insert_control_fps=float(self.config.fixed_insert_insert_control_fps),
            insert_max_joint_step_deg=float(self.config.fixed_insert_insert_max_joint_step_deg),
            insert_final_settle_s=0.0,
            pilz_planner_script=Path(self.config.fixed_insert_pilz_planner_script).expanduser(),
            pilz_setup_commands=self.config.fixed_insert_pilz_setup_commands,
            pilz_move_group_action=self.config.fixed_insert_pilz_move_group_action,
            pilz_group_name=self.config.fixed_insert_pilz_group_name,
            pilz_base_frame=self.config.fixed_insert_pilz_base_frame,
            pilz_tip_frame=self.config.fixed_insert_pilz_tip_frame,
            pilz_allowed_planning_time_s=float(self.config.fixed_insert_pilz_allowed_planning_time_s),
            pilz_timeout_s=float(self.config.fixed_insert_pilz_timeout_s),
            pilz_num_planning_attempts=int(self.config.fixed_insert_pilz_num_planning_attempts),
            pilz_max_velocity_scaling=float(self.config.fixed_insert_pilz_max_velocity_scaling),
            pilz_max_acceleration_scaling=float(self.config.fixed_insert_pilz_max_acceleration_scaling),
            pilz_execute_time_scale=float(self.config.fixed_insert_pilz_execute_time_scale),
            pilz_min_point_dt_s=float(self.config.fixed_insert_pilz_min_point_dt_s),
            pilz_insert_quintic_interpolation=bool(
                self.config.fixed_insert_pilz_insert_quintic_interpolation
            ),
            pilz_insert_settle_timeout_s=float(self.config.fixed_insert_pilz_insert_settle_timeout_s),
            pilz_final_release_settle_timeout_s=float(
                self.config.fixed_insert_pilz_final_release_settle_timeout_s
            ),
            target_pos_tol_m=0.005,
            target_rot_tol_deg=3.0,
            dls_damping=0.08,
            jacobian_step_deg=0.25,
            position_weight=20.0,
            rotation_weight=1.0,
            status_period_s=0.25,
            hold_recorded_gripper=True,
        )

    @staticmethod
    def _is_fixed_insert_pilz_settle_timeout(exc: BaseException) -> bool:
        return "timed out waiting for executed trajectory to settle" in str(exc)

    @staticmethod
    def _fixed_insert_pilz_settle_args(args: argparse.Namespace) -> argparse.Namespace:
        settle_args = argparse.Namespace(**vars(args))
        settle_args.approach_timeout_s = float(getattr(args, "pilz_insert_settle_timeout_s", 3.0))
        return settle_args

    @staticmethod
    def _fixed_insert_pilz_final_release_settle_args(args: argparse.Namespace) -> argparse.Namespace:
        settle_args = argparse.Namespace(**vars(args))
        settle_args.approach_timeout_s = float(
            getattr(args, "pilz_final_release_settle_timeout_s", 0.1)
        )
        return settle_args

    def _fixed_insert_current_fk_pose_or(
        self,
        *,
        fixed_insert: object,
        robot: object,
        kinematics: object,
        args: argparse.Namespace,
        fallback_pose: object,
        stage: str,
    ):
        import numpy as np

        try:
            _observation, _joints, current_pose = fixed_insert.current_joints_pose_observation(
                robot=robot,
                kinematics=kinematics,
                side=args.side,
            )
            pose = np.asarray(current_pose, dtype=float).reshape(4, 4).copy()
            self.logger.info(
                "Fixed insert %s: continuing from current FK link6_xyz=[%+.4f,%+.4f,%+.4f].",
                stage,
                float(pose[0, 3]),
                float(pose[1, 3]),
                float(pose[2, 3]),
            )
            return pose
        except Exception as pose_exc:
            fallback = np.asarray(fallback_pose, dtype=float).reshape(4, 4).copy()
            self.logger.warning(
                "Fixed insert %s: failed to read current FK after non-critical settle timeout; "
                "continuing from fallback xyz=[%+.4f,%+.4f,%+.4f]: %s",
                stage,
                float(fallback[0, 3]),
                float(fallback[1, 3]),
                float(fallback[2, 3]),
                pose_exc,
            )
            return fallback

    def _send_fixed_insert_gripper_release(self) -> None:
        if not self.config.fixed_insert_release_gripper:
            return

        gripper_key = self.config.fixed_insert_release_gripper_key.strip()
        if not gripper_key:
            self.logger.warning("Fixed insert gripper release skipped: no gripper key configured.")
            return

        release_pos = float(self.config.fixed_insert_release_gripper_pos)
        if gripper_key in PIPER_ASYNC_ABSOLUTE_ACTION_LIMITS:
            low, high = PIPER_ASYNC_ABSOLUTE_ACTION_LIMITS[gripper_key]
            clamped_release_pos = min(high, max(low, release_pos))
            if clamped_release_pos != release_pos:
                self.logger.warning(
                    "Fixed insert gripper release target %.2f clipped to %.2f for %s.",
                    release_pos,
                    clamped_release_pos,
                    gripper_key,
                )
            release_pos = clamped_release_pos

        piper_arm = None
        direct_piper_arm = None
        direct_piper_config = None
        if gripper_key == "right_gripper.pos" and hasattr(self.robot, "right_arm"):
            piper_arm = getattr(self.robot, "right_arm")
            direct_piper_arm = getattr(piper_arm, "arm", None)
            direct_piper_config = getattr(piper_arm, "config", None)
        elif gripper_key == "left_gripper.pos" and hasattr(self.robot, "left_arm"):
            piper_arm = getattr(self.robot, "left_arm")
            direct_piper_arm = getattr(piper_arm, "arm", None)
            direct_piper_config = getattr(piper_arm, "config", None)
        use_piper_arm_send_action = piper_arm is not None and hasattr(piper_arm, "send_action")
        use_direct_piper_gripper = direct_piper_arm is not None and hasattr(direct_piper_arm, "GripperCtrl")

        steps = max(int(self.config.fixed_insert_release_command_s * self.config.fps), 1)
        dt_s = 1.0 / self.config.fps
        self.logger.info(
            "Fixed insert: opening %s to %.2f for %.2fs.",
            gripper_key,
            release_pos,
            self.config.fixed_insert_release_command_s,
        )
        for _ in range(steps):
            loop_start_t = time.perf_counter()
            if use_piper_arm_send_action:
                piper_arm.send_action({"gripper.pos": release_pos})
            elif use_direct_piper_gripper:
                direct_piper_arm.GripperCtrl(
                    unit_to_milli(release_pos),
                    getattr(direct_piper_config, "gripper_effort_default", 1000),
                    getattr(direct_piper_config, "gripper_status_code", 0x01),
                    0x00,
                )
            else:
                action = {gripper_key: release_pos}
                self.robot.send_action(action)
            elapsed_s = time.perf_counter() - loop_start_t
            time.sleep(max(dt_s - elapsed_s, 0.0))

    def _fixed_insert_head_rgb_image_from_observation(self, observation: dict[str, object]) -> object | None:
        return self._fixed_insert_image_from_observation(
            observation,
            configured_key=self.config.fixed_insert_head_rgb_image_key,
        )

    def _fixed_insert_wrist_redline_image_from_observation(self, observation: dict[str, object]) -> object | None:
        return self._fixed_insert_image_from_observation(
            observation,
            configured_key=self.config.fixed_insert_wrist_redline_image_key,
        )

    @staticmethod
    def _fixed_insert_image_from_observation(
        observation: dict[str, object],
        *,
        configured_key: str,
    ) -> object | None:
        configured_key = configured_key.strip()
        candidate_keys = [
            configured_key,
            f"observation.images.{configured_key}",
            configured_key.removeprefix("observation.images."),
        ]
        for key in candidate_keys:
            if key in observation:
                return observation[key]
        return None

    @staticmethod
    def _as_rgb_hwc_uint8(image: object):
        import numpy as np

        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"expected HxWx3 or 3xHxW image, got shape {array.shape}")
        if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != 3:
            raise ValueError(f"expected 3-channel image, got shape {array.shape}")
        if array.dtype != np.uint8:
            if float(np.nanmax(array)) <= 1.5:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _write_fixed_insert_head_rgb_debug_artifacts(
        self,
        *,
        image_rgb: object,
        result: object,
        head_rgb: object,
    ) -> None:
        output_text = self.config.fixed_insert_head_rgb_debug_output_dir.strip()
        if not output_text:
            return
        try:
            output_dir = Path(output_text).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            image_array = self._as_rgb_hwc_uint8(image_rgb)
            overlay_bgr = head_rgb.render_head_rgb_compensation_overlay_bgr(
                image_array,
                result=result,
                slot_center_xy=self.config.fixed_insert_head_rgb_slot_center_xy,
                slot_down_axis_xy=self.config.fixed_insert_head_rgb_slot_down_axis_xy,
            )
            import cv2

            if not cv2.imwrite(str(output_dir / "last.png"), overlay_bgr):
                raise RuntimeError(f"failed to write {output_dir / 'last.png'}")
            red_center = result.red_center_xy
            payload = {
                "image_key": self.config.fixed_insert_head_rgb_image_key,
                "decision": result.decision.value,
                "axial_error_px": float(result.axial_error_px),
                "compensation_m": float(result.compensation_m),
                "red_center_xy": None if red_center is None else [float(red_center[0]), float(red_center[1])],
                "reject_reason": result.reject_reason,
                "slot_center_xy": list(
                    head_rgb.parse_xy(
                        self.config.fixed_insert_head_rgb_slot_center_xy,
                        label="fixed_insert_head_rgb_slot_center_xy",
                    )
                ),
                "slot_down_axis_xy": list(
                    head_rgb.parse_xy(
                        self.config.fixed_insert_head_rgb_slot_down_axis_xy,
                        label="fixed_insert_head_rgb_slot_down_axis_xy",
                    )
                ),
                "base_down_axis_xy": list(
                    head_rgb.parse_xy(
                        self.config.fixed_insert_head_rgb_base_down_axis_xy,
                        label="fixed_insert_head_rgb_base_down_axis_xy",
                    )
                ),
            }
            (output_dir / "last.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except Exception as exc:
            self.logger.warning("Fixed insert head-RGB debug artifact write failed: %s", exc)

    def _write_fixed_insert_wrist_redline_debug_artifacts(
        self,
        *,
        image_rgb: object,
        result: object,
        reference: object,
        wrist_redline: object,
    ) -> None:
        output_text = self.config.fixed_insert_wrist_redline_debug_output_dir.strip()
        if not output_text:
            return
        try:
            output_dir = Path(output_text).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            image_array = self._as_rgb_hwc_uint8(image_rgb)
            overlay_bgr = wrist_redline.render_wrist_redline_overlay_bgr(image_array, result=result)
            import cv2

            if not cv2.imwrite(str(output_dir / "last.png"), overlay_bgr):
                raise RuntimeError(f"failed to write {output_dir / 'last.png'}")
            measurement = result.measurement
            payload = {
                "image_key": self.config.fixed_insert_wrist_redline_image_key,
                "decision": result.decision.value,
                "length_px": float(result.length_px),
                "length_error_px": float(result.length_error_px),
                "normalized_error": float(result.normalized_error),
                "compensation_m": float(result.compensation_m),
                "reject_reason": result.reject_reason,
                "bbox_xywh": None if measurement is None else list(measurement.bbox_xywh),
                "line_xyxy": None if measurement is None else list(measurement.line_xyxy),
                "reference": {
                    "up_length_px": float(reference.up_length_px),
                    "center_length_px": float(reference.center_length_px),
                    "down_length_px": float(reference.down_length_px),
                    "deadband_px": float(reference.deadband_px),
                    "max_compensation_m": float(reference.max_compensation_m),
                    "center_up_compensation_m": float(reference.center_up_compensation_m),
                    "up_start_compensation_m": (
                        None
                        if reference.up_start_compensation_m is None
                        else float(reference.up_start_compensation_m)
                    ),
                    "up_compensation_m": (
                        float(reference.max_compensation_m)
                        if reference.up_compensation_m is None
                        else float(reference.up_compensation_m)
                    ),
                    "down_start_compensation_m": (
                        None
                        if reference.down_start_compensation_m is None
                        else float(reference.down_start_compensation_m)
                    ),
                    "down_compensation_m": (
                        float(reference.max_compensation_m)
                        if reference.down_compensation_m is None
                        else float(reference.down_compensation_m)
                    ),
                },
                "base_down_axis_xy": list(
                    wrist_redline.parse_xy(
                        self.config.fixed_insert_wrist_redline_base_down_axis_xy,
                        label="fixed_insert_wrist_redline_base_down_axis_xy",
                    )
                ),
            }
            (output_dir / "last.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except Exception as exc:
            self.logger.warning("Fixed insert wrist-redline debug artifact write failed: %s", exc)

    def _run_fixed_insert_head_rgb_compensation(
        self,
        *,
        fixed_insert: object,
        robot: object,
        kinematics: object,
        args: argparse.Namespace,
        target_gripper_pos: float | None,
        base_pose: object | None = None,
        min_z_m: float | None = None,
    ):
        import numpy as np
        from scripts.tools import head_rgb_slot_compensation as head_rgb

        observation, _joints, current_pose = fixed_insert.current_joints_pose_observation(
            robot=robot,
            kinematics=kinematics,
            side=args.side,
            include_cameras=True,
        )
        image = self._fixed_insert_head_rgb_image_from_observation(dict(observation))
        if image is None:
            self.logger.warning(
                "Fixed insert head-RGB compensation skipped: image key %s not found.",
                self.config.fixed_insert_head_rgb_image_key,
            )
            return np.asarray(base_pose if base_pose is not None else current_pose, dtype=float).reshape(4, 4)

        try:
            image_rgb = self._as_rgb_hwc_uint8(image)
            result = head_rgb.classify_head_rgb_slot_compensation(
                image_rgb,
                slot_center_xy=self.config.fixed_insert_head_rgb_slot_center_xy,
                slot_down_axis_xy=self.config.fixed_insert_head_rgb_slot_down_axis_xy,
                deadband_px=float(self.config.fixed_insert_head_rgb_deadband_px),
                compensation_m=float(self.config.fixed_insert_head_rgb_compensation_m),
                min_red_area_px=float(self.config.fixed_insert_head_rgb_min_red_area_px),
            )
        except Exception as exc:
            self.logger.warning("Fixed insert head-RGB compensation skipped: classification failed: %s", exc)
            return np.asarray(base_pose if base_pose is not None else current_pose, dtype=float).reshape(4, 4)

        self.logger.info(
            "Fixed insert head-RGB compensation: decision=%s axial=%+.1fpx comp=%+.3fm red_center=%s reason=%s",
            result.decision.value,
            float(result.axial_error_px),
            float(result.compensation_m),
            result.red_center_xy,
            result.reject_reason,
        )
        print(
            "head-rgb-compensation "
            f"decision={result.decision.value} axial={float(result.axial_error_px):+.1f}px "
            f"comp={float(result.compensation_m):+.3f}m red_center={result.red_center_xy} "
            f"reason={result.reject_reason}",
            flush=True,
        )
        self._write_fixed_insert_head_rgb_debug_artifacts(
            image_rgb=image_rgb,
            result=result,
            head_rgb=head_rgb,
        )
        if float(result.compensation_m) == 0.0:
            return np.asarray(base_pose if base_pose is not None else current_pose, dtype=float).reshape(4, 4)

        desired_pose = head_rgb.compensated_pose_from_base_axis(
            np.asarray(base_pose if base_pose is not None else current_pose, dtype=float).reshape(4, 4),
            compensation_m=float(result.compensation_m),
            base_down_axis_xy=self.config.fixed_insert_head_rgb_base_down_axis_xy,
            compensation_sign=float(self.config.fixed_insert_head_rgb_compensation_sign),
        )
        stage = f"head-rgb-comp-{result.decision.value}"
        if self.config.fixed_insert_linear_motion_backend == "pilz":
            fixed_insert.execute_pilz_linear_pose(
                args=self._fixed_insert_pilz_settle_args(args),
                robot=robot,
                kinematics=kinematics,
                desired_pose=desired_pose,
                stage=f"{stage}-pilz",
                target_gripper_pos=target_gripper_pos,
            )
        else:
            fixed_insert.execute_until_locked_xy_pose(
                args=args,
                robot=robot,
                kinematics=kinematics,
                desired_pose=desired_pose,
                stage=stage,
                control_fps=float(getattr(args, "approach_control_fps", self.config.fps)),
                max_joint_step_deg=float(self.config.fixed_insert_head_rgb_max_joint_step_deg),
                timeout_s=float(self.config.fixed_insert_head_rgb_timeout_s),
                target_gripper_pos=target_gripper_pos,
                min_z_m=min_z_m,
            )
        return np.asarray(desired_pose, dtype=float).reshape(4, 4)

    def _run_fixed_insert_wrist_redline_compensation(
        self,
        *,
        fixed_insert: object,
        robot: object,
        kinematics: object,
        args: argparse.Namespace,
        target_gripper_pos: float | None,
        base_pose: object | None = None,
        min_z_m: float | None = None,
    ):
        import numpy as np
        from scripts.tools import wrist_redline_grip_compensation as wrist_redline

        observation, _joints, current_pose = fixed_insert.current_joints_pose_observation(
            robot=robot,
            kinematics=kinematics,
            side=args.side,
            include_cameras=True,
        )
        image = self._fixed_insert_wrist_redline_image_from_observation(dict(observation))
        if image is None:
            self.logger.warning(
                "Fixed insert wrist-redline compensation skipped: image key %s not found.",
                self.config.fixed_insert_wrist_redline_image_key,
            )
            return np.asarray(base_pose if base_pose is not None else current_pose, dtype=float).reshape(4, 4)

        try:
            reference = wrist_redline.load_reference_from_pose_images(
                up_pose_path=Path(self.config.fixed_insert_wrist_redline_up_pose_path).expanduser(),
                center_pose_path=Path(self.config.fixed_insert_wrist_redline_center_pose_path).expanduser(),
                down_pose_path=Path(self.config.fixed_insert_wrist_redline_down_pose_path).expanduser(),
                camera_name="right_wrist",
                min_red_area_px=float(self.config.fixed_insert_wrist_redline_min_red_area_px),
                deadband_px=float(self.config.fixed_insert_wrist_redline_deadband_px),
                max_compensation_m=float(self.config.fixed_insert_wrist_redline_max_compensation_m),
                center_up_compensation_m=float(self.config.fixed_insert_wrist_redline_center_up_compensation_m),
                up_start_compensation_m=float(
                    self.config.fixed_insert_wrist_redline_up_start_compensation_m
                ),
                up_compensation_m=float(self.config.fixed_insert_wrist_redline_up_compensation_m),
                down_start_compensation_m=float(
                    self.config.fixed_insert_wrist_redline_down_start_compensation_m
                ),
                down_compensation_m=float(self.config.fixed_insert_wrist_redline_down_compensation_m),
            )
            center_length_override_px = float(
                self.config.fixed_insert_wrist_redline_center_length_override_px
            )
            if center_length_override_px > 0.0:
                original_center_length_px = float(reference.center_length_px)
                reference = wrist_redline.shift_reference_center_length(
                    reference,
                    center_length_px=center_length_override_px,
                )
                self.logger.info(
                    "Fixed insert wrist-redline reference shifted: center %.1fpx -> %.1fpx "
                    "(up=%.1fpx center=%.1fpx down=%.1fpx).",
                    original_center_length_px,
                    center_length_override_px,
                    float(reference.up_length_px),
                    float(reference.center_length_px),
                    float(reference.down_length_px),
                )
            image_rgb = self._as_rgb_hwc_uint8(image)
            result = wrist_redline.classify_wrist_redline_grip(
                image_rgb,
                reference=reference,
                min_red_area_px=float(self.config.fixed_insert_wrist_redline_min_red_area_px),
            )
        except Exception as exc:
            self.logger.warning("Fixed insert wrist-redline compensation skipped: classification failed: %s", exc)
            return np.asarray(base_pose if base_pose is not None else current_pose, dtype=float).reshape(4, 4)

        self.logger.info(
            "Fixed insert wrist-redline compensation: decision=%s length=%+.1fpx error=%+.1fpx comp=%+.3fm reason=%s",
            result.decision.value,
            float(result.length_px),
            float(result.length_error_px),
            float(result.compensation_m),
            result.reject_reason,
        )
        print(
            "wrist-redline-compensation "
            f"decision={result.decision.value} length={float(result.length_px):.1f}px "
            f"error={float(result.length_error_px):+.1f}px "
            f"ratio={float(result.normalized_error):+.2f} comp={float(result.compensation_m):+.3f}m "
            f"reason={result.reject_reason}",
            flush=True,
        )
        self._write_fixed_insert_wrist_redline_debug_artifacts(
            image_rgb=image_rgb,
            result=result,
            reference=reference,
            wrist_redline=wrist_redline,
        )
        if float(result.compensation_m) == 0.0:
            return np.asarray(base_pose if base_pose is not None else current_pose, dtype=float).reshape(4, 4)

        desired_pose = wrist_redline.compensated_pose_from_base_axis(
            np.asarray(base_pose if base_pose is not None else current_pose, dtype=float).reshape(4, 4),
            compensation_m=float(result.compensation_m),
            base_down_axis_xy=self.config.fixed_insert_wrist_redline_base_down_axis_xy,
            compensation_sign=float(self.config.fixed_insert_wrist_redline_compensation_sign),
        )
        stage = f"wrist-redline-comp-{result.decision.value}"
        if self.config.fixed_insert_linear_motion_backend == "pilz":
            fixed_insert.execute_pilz_linear_pose(
                args=self._fixed_insert_pilz_settle_args(args),
                robot=robot,
                kinematics=kinematics,
                desired_pose=desired_pose,
                stage=f"{stage}-pilz",
                target_gripper_pos=target_gripper_pos,
            )
        else:
            fixed_insert.execute_until_locked_xy_pose(
                args=args,
                robot=robot,
                kinematics=kinematics,
                desired_pose=desired_pose,
                stage=stage,
                control_fps=float(getattr(args, "approach_control_fps", self.config.fps)),
                max_joint_step_deg=float(self.config.fixed_insert_wrist_redline_max_joint_step_deg),
                timeout_s=float(self.config.fixed_insert_wrist_redline_timeout_s),
                target_gripper_pos=target_gripper_pos,
                min_z_m=min_z_m,
            )
        return np.asarray(desired_pose, dtype=float).reshape(4, 4)

    def _execute_fixed_insert_primitive(self) -> None:
        from scripts.tools import run_phone_slot_fixed_pose_insert as fixed_insert
        import numpy as np

        args = self._make_fixed_insert_args()
        pose = fixed_insert.load_insert_ready_pose(Path(args.pose), side=args.side)
        kinematics = fixed_insert.make_execution_kinematics(args)
        preinsert_pose = np.asarray(kinematics.forward_kinematics(pose.joints_deg), dtype=float).reshape(4, 4)
        pre_comp_lift_m = float(self.config.fixed_insert_pre_comp_lift_m)
        nominal_insert_start_pose = preinsert_pose.copy()
        nominal_insert_start_pose[2, 3] += pre_comp_lift_m

        def log_stage(stage: str) -> None:
            try:
                observation, _joints, current_pose = fixed_insert.current_joints_pose_observation(
                    robot=self.robot,
                    kinematics=kinematics,
                    side=args.side,
                )
                gripper_key = f"{args.side}_gripper.pos"
                gripper = observation.get(gripper_key, observation.get("gripper.pos", "na"))
                self.logger.info(
                    "Fixed insert stage=%s link6_xyz=[%+.4f,%+.4f,%+.4f] gripper=%s",
                    stage,
                    float(current_pose[0, 3]),
                    float(current_pose[1, 3]),
                    float(current_pose[2, 3]),
                    gripper,
                )
            except Exception as exc:
                self.logger.warning("Fixed insert stage=%s pose diagnostic failed: %s", stage, exc)

        self.logger.info(
            "Fixed insert handoff: moving to %s, then inserting %.3fm.",
            pose.path,
            float(args.insert_distance_m),
        )
        self.logger.info(
            "Fixed insert target preinsert link6_xyz=[%+.4f,%+.4f,%+.4f] "
            "pre_comp_lift=%.3fm insert_start_z=%+.4f final_z=%+.4f return_to_start=%s",
            float(preinsert_pose[0, 3]),
            float(preinsert_pose[1, 3]),
            float(preinsert_pose[2, 3]),
            pre_comp_lift_m,
            float(nominal_insert_start_pose[2, 3]),
            float(nominal_insert_start_pose[2, 3]) - float(args.insert_distance_m),
            bool(self.config.fixed_insert_return_to_start_pose),
        )
        log_stage("before_approach")
        fixed_insert.execute_until_recorded_joints(
            args=args,
            robot=self.robot,
            target_joints_deg=pose.joints_deg,
            target_gripper_pos=pose.gripper_pos,
        )
        log_stage("after_approach")
        _observation, _joints, after_approach_pose = fixed_insert.current_joints_pose_observation(
            robot=self.robot,
            kinematics=kinematics,
            side=args.side,
        )
        insert_start_pose = np.asarray(after_approach_pose, dtype=float).reshape(4, 4).copy()
        self.logger.info(
            "Fixed insert link6-phone start pose: xyz=[%+.4f,%+.4f,%+.4f]. "
            "Subsequent Cartesian moves use backend=%s from this actual FK pose.",
            float(insert_start_pose[0, 3]),
            float(insert_start_pose[1, 3]),
            float(insert_start_pose[2, 3]),
            self.config.fixed_insert_linear_motion_backend,
        )

        def keep_locked_pose_with_candidate_xy(locked_pose: np.ndarray, candidate_pose: np.ndarray) -> np.ndarray:
            locked = np.asarray(locked_pose, dtype=float).reshape(4, 4)
            candidate = np.asarray(candidate_pose, dtype=float).reshape(4, 4)
            result = locked.copy()
            result[0, 3] = float(candidate[0, 3])
            result[1, 3] = float(candidate[1, 3])
            return result

        if pre_comp_lift_m > 0.0:
            lift_target_pose = insert_start_pose.copy()
            lift_target_pose[2, 3] += pre_comp_lift_m
            self.logger.info(
                "Fixed insert pre-comp lift: current_z=%+.4f target_z=%+.4f nominal_insert_start_z=%+.4f.",
                float(insert_start_pose[2, 3]),
                float(lift_target_pose[2, 3]),
                float(nominal_insert_start_pose[2, 3]),
            )
            if self.config.fixed_insert_linear_motion_backend == "pilz":
                try:
                    fixed_insert.execute_pilz_linear_pose(
                        args=self._fixed_insert_pilz_settle_args(args),
                        robot=self.robot,
                        kinematics=kinematics,
                        desired_pose=lift_target_pose,
                        stage="pre-comp-lift-pilz",
                        target_gripper_pos=pose.gripper_pos,
                    )
                except RuntimeError as exc:
                    if not self._is_fixed_insert_pilz_settle_timeout(exc):
                        raise
                    self.logger.warning(
                        "Fixed insert pre-comp lift PILZ settle timed out; "
                        "treating lift as best-effort and continuing: %s",
                        exc,
                    )
                    insert_start_pose = self._fixed_insert_current_fk_pose_or(
                        fixed_insert=fixed_insert,
                        robot=self.robot,
                        kinematics=kinematics,
                        args=args,
                        fallback_pose=insert_start_pose,
                        stage="pre-comp-lift",
                    )
                    log_stage("after_pre_comp_lift_timeout")
                else:
                    insert_start_pose = lift_target_pose
                    log_stage("after_pre_comp_lift")
            else:
                fixed_insert.execute_until_min_z(
                    args=args,
                    robot=self.robot,
                    kinematics=kinematics,
                    desired_pose=lift_target_pose,
                    target_z_m=float(lift_target_pose[2, 3]),
                    z_tol_m=float(self.config.fixed_insert_pre_comp_lift_tol_m),
                    stage="pre-comp-lift",
                    control_fps=float(getattr(args, "approach_control_fps", self.config.fps)),
                    max_joint_step_deg=float(args.approach_max_joint_step_deg),
                    timeout_s=float(args.approach_timeout_s),
                    target_gripper_pos=pose.gripper_pos,
                )
                insert_start_pose = lift_target_pose
                log_stage("after_pre_comp_lift")
        compensation_min_z_m = None
        if pre_comp_lift_m > 0.0:
            compensation_min_z_m = float(insert_start_pose[2, 3]) - float(self.config.fixed_insert_pre_comp_lift_tol_m)
        if self.config.fixed_insert_head_rgb_compensation:
            try:
                compensated_pose = self._run_fixed_insert_head_rgb_compensation(
                    fixed_insert=fixed_insert,
                    robot=self.robot,
                    kinematics=kinematics,
                    args=args,
                    target_gripper_pos=pose.gripper_pos,
                    base_pose=insert_start_pose,
                    min_z_m=compensation_min_z_m,
                )
            except RuntimeError as exc:
                if not self._is_fixed_insert_pilz_settle_timeout(exc):
                    raise
                self.logger.warning(
                    "Fixed insert head-RGB compensation PILZ settle timed out; "
                    "treating compensation as best-effort and continuing: %s",
                    exc,
                )
                insert_start_pose = self._fixed_insert_current_fk_pose_or(
                    fixed_insert=fixed_insert,
                    robot=self.robot,
                    kinematics=kinematics,
                    args=args,
                    fallback_pose=insert_start_pose,
                    stage="head-rgb-compensation",
                )
                log_stage("after_head_rgb_compensation_timeout")
            else:
                insert_start_pose = keep_locked_pose_with_candidate_xy(insert_start_pose, compensated_pose)
                log_stage("after_head_rgb_compensation")
        if self.config.fixed_insert_wrist_redline_compensation:
            try:
                compensated_pose = self._run_fixed_insert_wrist_redline_compensation(
                    fixed_insert=fixed_insert,
                    robot=self.robot,
                    kinematics=kinematics,
                    args=args,
                    target_gripper_pos=pose.gripper_pos,
                    base_pose=insert_start_pose,
                    min_z_m=compensation_min_z_m,
                )
            except RuntimeError as exc:
                if not self._is_fixed_insert_pilz_settle_timeout(exc):
                    raise
                self.logger.warning(
                    "Fixed insert wrist-redline compensation PILZ settle timed out; "
                    "treating compensation as best-effort and continuing: %s",
                    exc,
                )
                insert_start_pose = self._fixed_insert_current_fk_pose_or(
                    fixed_insert=fixed_insert,
                    robot=self.robot,
                    kinematics=kinematics,
                    args=args,
                    fallback_pose=insert_start_pose,
                    stage="wrist-redline-compensation",
                )
                log_stage("after_wrist_redline_compensation_timeout")
            else:
                insert_start_pose = keep_locked_pose_with_candidate_xy(insert_start_pose, compensated_pose)
                log_stage("after_wrist_redline_compensation")
        insert_start_pose_for_insert = np.asarray(insert_start_pose, dtype=float).reshape(4, 4).copy()
        if self.config.fixed_insert_linear_motion_backend == "pilz":
            final_insert_pose = insert_start_pose_for_insert.copy()
            final_insert_pose[2, 3] -= float(args.insert_distance_m)
            self.logger.info(
                "Fixed insert PILZ LIN insert: start_z=%+.4f final_z=%+.4f distance=%.3fm.",
                float(insert_start_pose_for_insert[2, 3]),
                float(final_insert_pose[2, 3]),
                float(args.insert_distance_m),
            )
            insert_args = self._fixed_insert_pilz_final_release_settle_args(args)
            try:
                fixed_insert.execute_pilz_linear_pose(
                    args=insert_args,
                    robot=self.robot,
                    kinematics=kinematics,
                    desired_pose=final_insert_pose,
                    stage="insert-z-pilz",
                    target_gripper_pos=pose.gripper_pos,
                )
            except RuntimeError as exc:
                if "timed out waiting for executed trajectory to settle" not in str(exc):
                    raise
                self.logger.warning(
                    "Fixed insert PILZ LIN insert settle failed after %.2fs; releasing gripper and returning: %s",
                    float(insert_args.approach_timeout_s),
                    exc,
                )
                log_stage("after_insert_stuck")
                self._send_fixed_insert_gripper_release()
                log_stage("after_release_after_insert_stuck")
                self._return_fixed_insert_to_start_pose()
                log_stage("after_return_after_insert_stuck")
                return
        else:
            fixed_insert.execute_straight_insert(
                args=args,
                robot=self.robot,
                kinematics=kinematics,
                preinsert_pose=insert_start_pose_for_insert,
                target_gripper_pos=pose.gripper_pos,
            )
        log_stage("after_insert")
        self._send_fixed_insert_gripper_release()
        log_stage("after_release")
        self._return_fixed_insert_to_start_pose()
        log_stage("after_return")

    def _run_handoff_fixed_insert_sequence(self) -> None:
        if self._handoff_fixed_insert_active or self._handoff_fixed_insert_done:
            return

        self._handoff_fixed_insert_active = True
        self._skip_return_home_on_stop_once = True
        repeat_handoff = bool(self.config.handoff_fixed_insert_repeat)
        self.logger.info(
            "VLA handoff detected; %s VLA action execution and starting fixed insert.",
            "pausing" if repeat_handoff else "stopping",
        )
        self._flush_action_queue()
        if not repeat_handoff:
            self.shutdown_event.set()
        try:
            self._execute_fixed_insert_primitive()
            self.logger.info("Fixed insert handoff sequence finished.")
        except Exception as exc:
            self.logger.exception("Fixed insert handoff sequence failed; holding current robot state: %s", exc)
        finally:
            self._flush_action_queue()
            if repeat_handoff:
                self._handoff_fixed_insert_detector.reset()
                self._handoff_vla_started_s = None
                self.must_go.set()
                resume_delay_s = float(self.config.handoff_fixed_insert_resume_delay_s)
                if resume_delay_s > 0:
                    self.logger.info(
                        "Fixed insert handoff repeat: waiting %.2fs before resuming VLA.",
                        resume_delay_s,
                    )
                    self._sleep_with_shutdown(resume_delay_s)
                self._handoff_fixed_insert_done = False
                self.logger.info("Fixed insert handoff repeat: VLA control resumed.")
            else:
                self._handoff_fixed_insert_done = True
            self._handoff_fixed_insert_active = False

    def _maybe_handoff_fixed_insert(self, raw_observation: RawObservation) -> bool:
        if (
            not self.config.handoff_fixed_insert
            or self._handoff_fixed_insert_active
            or self._handoff_fixed_insert_done
        ):
            return False

        if self._handoff_vla_started_s is None:
            self._handoff_fixed_insert_detector.reset()
            return False

        if self._handoff_fixed_insert_detector.update(raw_observation, now_s=time.perf_counter()):
            self._run_handoff_fixed_insert_sequence()
            return True

        return False

    def _maybe_handoff_fixed_insert_from_local_state(self) -> bool:
        if (
            not self.config.handoff_fixed_insert
            or self._handoff_fixed_insert_active
            or self._handoff_fixed_insert_done
        ):
            return False

        if self._handoff_vla_started_s is None:
            self._handoff_fixed_insert_detector.reset()
            return False

        return self._maybe_handoff_fixed_insert(self._get_action_safety_observation())

    def _maybe_stop_on_pose(self, raw_observation: RawObservation) -> None:
        if not self.config.stop_on_pose or not self._stop_pose_target:
            return

        runtime_s = time.perf_counter() - self._start_time_s
        min_runtime_reached = runtime_s >= self.config.stop_pose_min_runtime_s

        max_error = 0.0
        all_within_tolerance = True
        missing_keys: list[str] = []
        target_errors: dict[str, float] = {}
        for key, target in self._stop_pose_target.items():
            if key not in raw_observation:
                missing_keys.append(key)
                all_within_tolerance = False
                continue
            current = float(raw_observation[key])
            tolerance = (
                self.config.stop_pose_gripper_tolerance
                if "gripper" in key
                else self.config.stop_pose_tolerance_deg
            )
            error = abs(target - current)
            target_errors[key] = target - current
            max_error = max(max_error, error)
            if error > tolerance:
                all_within_tolerance = False

        if not min_runtime_reached:
            all_within_tolerance = False

        if missing_keys:
            self.logger.warning(
                "Stop-on-pose target has keys missing from robot observation: %s", missing_keys
            )
            self._stop_pose_stable_count = 0
            return

        if all_within_tolerance:
            self._stop_pose_stable_count += 1
        else:
            self._stop_pose_stable_count = 0

        tail_row: dict[str, float | int | bool] = {
            "elapsed_s": runtime_s,
            "stable_count": self._stop_pose_stable_count,
            "stable_required": self.config.stop_pose_stable_frames,
            "within_tolerance": all_within_tolerance,
            "would_stop": all_within_tolerance
            and self._stop_pose_stable_count >= self.config.stop_pose_stable_frames,
            "dry_run": self.config.stop_pose_dry_run,
            "max_error": max_error,
            "joint_tolerance": self.config.stop_pose_tolerance_deg,
            "gripper_tolerance": self.config.stop_pose_gripper_tolerance,
            "min_runtime_reached": min_runtime_reached,
        }
        for key, value in raw_observation.items():
            if key.startswith("right_") and key.endswith(".pos"):
                tail_row[f"{key}.current"] = float(value)
        for key, target in self._stop_pose_target.items():
            tail_row[f"{key}.target"] = target
            tail_row[f"{key}.error"] = target_errors.get(key, math.nan)
        self._stop_pose_tail_buffer.append(tail_row)

        if self._stop_pose_stable_count == 1 or self._stop_pose_stable_count % self.config.fps == 0:
            self.logger.info(
                "Stop-on-pose check: stable=%d/%d max_error=%.2f tolerance(joint=%.2f, gripper=%.2f) "
                "dry_run=%s.",
                self._stop_pose_stable_count,
                self.config.stop_pose_stable_frames,
                max_error,
                self.config.stop_pose_tolerance_deg,
                self.config.stop_pose_gripper_tolerance,
                self.config.stop_pose_dry_run,
            )

        if self._stop_pose_stable_count >= self.config.stop_pose_stable_frames:
            if self.config.stop_pose_dry_run:
                if (
                    self._stop_pose_stable_count == self.config.stop_pose_stable_frames
                    or self._stop_pose_stable_count % self.config.fps == 0
                ):
                    self.logger.info(
                        "Stop-on-pose would stop now after %.2fs: stable=%d frames, max_error=%.2f. "
                        "Dry-run is enabled, so inference continues.",
                        runtime_s,
                        self._stop_pose_stable_count,
                        max_error,
                    )
                return
            self.logger.info(
                "Stop-on-pose reached after %.2fs: stable=%d frames, max_error=%.2f. "
                "Stopping async inference.",
                runtime_s,
                self._stop_pose_stable_count,
                max_error,
            )
            self.shutdown_event.set()

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        action = self._action_tensor_to_action_dict(timed_action.get_action())
        if self.config.enable_action_safety_limits:
            safety_observation = self._get_action_safety_observation()
            action, safety_info = clip_piper_async_action(action, safety_observation, self.config.fps)
            if safety_info["num_abs_clipped"] or safety_info["num_speed_clipped"]:
                now = time.perf_counter()
                if now - self._last_action_safety_log_t >= self.config.action_safety_log_interval_s:
                    self.logger.warning(
                        "Async action safety clipped targets: absolute=%s speed=%s, max delta %.2f -> %.2f.",
                        safety_info["num_abs_clipped"],
                        safety_info["num_speed_clipped"],
                        safety_info["max_abs_delta_before_clip"],
                        safety_info["max_abs_delta_after_clip"],
                    )
                    self._last_action_safety_log_t = now

        self._trace_pipeline("execute_action", "executing VLA action step #%s", timed_action.get_timestep())
        _performed_action = self.robot.send_action(action)
        self._mark_vla_action_executed()
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _mark_vla_action_executed(self, now_s: float | None = None) -> None:
        if self._handoff_vla_started_s is not None:
            return

        started_s = time.perf_counter() if now_s is None else float(now_s)
        self._handoff_vla_started_s = started_s
        self._handoff_fixed_insert_detector.start_time_s = started_s
        self._handoff_fixed_insert_detector.reset()

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            self._trace_pipeline("capture_observation_start", "capturing robot observation")
            raw_observation: RawObservation = self.robot.get_observation()
            image_keys = [
                key
                for key, value in raw_observation.items()
                if hasattr(value, "shape") and len(getattr(value, "shape", ())) >= 2
            ]
            self._trace_pipeline(
                "capture_observation_done",
                "captured observation in %.1fms with %d image keys",
                (time.perf_counter() - start_time) * 1000,
                len(image_keys),
            )
            self._maybe_display_camera_views(raw_observation)
            self._maybe_stop_on_pose(raw_observation)
            auto_cycle_triggered = self._maybe_auto_cycle_on_pose(raw_observation)
            handoff_triggered = self._maybe_handoff_fixed_insert(raw_observation)
            if handoff_triggered or auto_cycle_triggered or self._auto_cycle_recovery_active or not self.running:
                return raw_observation
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            self._trace_pipeline(
                "send_observation_start",
                "sending observation timestep #%s to bridge",
                observation.get_timestep(),
            )
            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            if self._maybe_handoff_fixed_insert_from_local_state():
                time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))
                continue

            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
