# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bridge Evo-RL async inference clients to an OpenPI websocket policy server.

This module exposes the same gRPC ``AsyncInference`` service used by
``lerobot.async_inference.robot_client``. Internally it forwards the latest
robot observation to an OpenPI websocket policy server and converts the returned
``actions`` array back into Evo-RL ``TimedAction`` objects.

Example:

```shell
PYTHONPATH=/home/phl/workspace/Evo-RL/src:/home/phl/workspace/openpi_jax_phone_slot/packages/openpi-client/src \
python -m lerobot.async_inference.openpi_bridge_server \
  --host=127.0.0.1 \
  --port=8080 \
  --openpi_host=127.0.0.1 \
  --openpi_port=8000 \
  --fps=30
```
"""

import logging
import pickle  # nosec
import sys
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .constants import DEFAULT_FPS, DEFAULT_INFERENCE_LATENCY, DEFAULT_OBS_QUEUE_TIMEOUT
from .helpers import (
    FPSTracker,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    make_lerobot_observation,
)


@dataclass
class OpenPIBridgeServerConfig:
    """Configuration for the OpenPI bridge server."""

    host: str = field(default="127.0.0.1", metadata={"help": "gRPC host for Evo-RL robot clients."})
    port: int = field(default=8080, metadata={"help": "gRPC port for Evo-RL robot clients."})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Robot control FPS."})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY,
        metadata={"help": "Minimum apparent GetActions latency, in seconds."},
    )
    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT,
        metadata={"help": "Timeout waiting for a fresh observation, in seconds."},
    )

    openpi_host: str = field(default="127.0.0.1", metadata={"help": "OpenPI websocket policy host."})
    openpi_port: int = field(default=8000, metadata={"help": "OpenPI websocket policy port."})
    openpi_api_key: str | None = field(default=None, metadata={"help": "Optional OpenPI websocket API key."})
    openpi_client_path: str | None = field(
        default="/home/phl/workspace/openpi_jax_phone_slot/packages/openpi-client/src",
        metadata={"help": "Path containing the openpi_client package. Empty string disables path injection."},
    )

    # These defaults match the pi05_piper_phone_slot_insert OpenPI training config.
    openpi_front_image_key: str = field(default="cam_high")
    openpi_left_wrist_image_key: str = field(default="cam_left_wrist")
    openpi_right_wrist_image_key: str = field(default="cam_right_wrist")
    lerobot_front_image_key: str = field(default="observation.images.right_front")
    lerobot_left_wrist_image_key: str = field(default="observation.images.left_left_wrist")
    lerobot_right_wrist_image_key: str = field(default="observation.images.right_right_wrist")

    state_key: str = field(default="observation.state")
    prompt_key: str = field(default="prompt")
    task_key: str = field(default="task")
    action_response_key: str = field(default="actions")
    default_prompt: str = field(default="")
    action_dim: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional number of action dimensions to keep before sending to Evo-RL. "
                "Leave unset to keep all dimensions returned by OpenPI."
            )
        },
    )

    log_timing_every: int = field(default=1, metadata={"help": "Log detailed timing every N action chunks."})
    debug_pipeline_trace: bool = field(
        default=False,
        metadata={"help": "Print low-rate bridge pipeline trace logs for hardware debugging."},
    )

    @property
    def environment_dt(self) -> float:
        return 1.0 / float(self.fps)

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")
        if self.openpi_port < 1 or self.openpi_port > 65535:
            raise ValueError(f"openpi_port must be between 1 and 65535, got {self.openpi_port}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")
        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")
        if self.action_dim is not None and self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive when set, got {self.action_dim}")
        if self.log_timing_every <= 0:
            raise ValueError(f"log_timing_every must be positive, got {self.log_timing_every}")


class OpenPIBridgeServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "openpi_bridge_server"
    logger = get_logger(prefix)

    def __init__(self, config: OpenPIBridgeServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps: set[int] = set()
        self._openpi_client = None
        self._openpi_client_lock = threading.Lock()
        self._action_chunks_generated = 0
        self._pipeline_trace_last_t: dict[str, float] = {}

        self.lerobot_features: dict[str, Any] | None = None
        self.actions_per_chunk: int | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def _trace_pipeline(self, key: str, message: str, *args, min_interval_s: float = 1.0) -> None:
        if not self.config.debug_pipeline_trace:
            return

        now_s = time.perf_counter()
        if now_s - self._pipeline_trace_last_t.get(key, 0.0) < min_interval_s:
            return
        self._pipeline_trace_last_t[key] = now_s
        self.logger.info("[pipeline] " + message, *args)

    def _reset_server(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def _get_openpi_client(self):
        if self._openpi_client is not None:
            return self._openpi_client

        with self._openpi_client_lock:
            if self._openpi_client is not None:
                return self._openpi_client

            openpi_client_path = (self.config.openpi_client_path or "").strip()
            if openpi_client_path:
                path = str(Path(openpi_client_path).expanduser())
                if path not in sys.path:
                    sys.path.insert(0, path)

            try:
                from openpi_client import websocket_client_policy
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Could not import openpi_client. Add it to PYTHONPATH, for example: "
                    "PYTHONPATH=/home/phl/workspace/Evo-RL/src:"
                    "/home/phl/workspace/openpi_jax_phone_slot/packages/openpi-client/src"
                ) from exc

            self._trace_pipeline(
                "openpi_connect_start",
                "connecting to OpenPI websocket %s:%s",
                self.config.openpi_host,
                self.config.openpi_port,
                min_interval_s=5.0,
            )
            self.logger.info(
                "Connecting to OpenPI websocket policy server at %s:%s",
                self.config.openpi_host,
                self.config.openpi_port,
            )
            self._openpi_client = websocket_client_policy.WebsocketClientPolicy(
                host=self.config.openpi_host,
                port=self.config.openpi_port,
                api_key=self.config.openpi_api_key,
            )
            self._trace_pipeline("openpi_connect_done", "connected to OpenPI websocket", min_interval_s=5.0)
            self.logger.info("OpenPI server metadata: %s", self._openpi_client.get_server_metadata())
            return self._openpi_client

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info("Client %s connected and ready", client_id)
        self._reset_server()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        self.logger.info(
            "Received policy instructions from %s | policy_type=%s | requested_path=%s | "
            "actions_per_chunk=%s | device=%s. Local policy loading is skipped because "
            "OpenPI owns the model.",
            context.peer(),
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            self.actions_per_chunk,
            policy_specs.device,
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        receive_time = time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, self.logger)
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_ms = (time.perf_counter() - start_deserialize) * 1000

        if not isinstance(timed_observation, TimedObservation):
            raise TypeError(f"Expected TimedObservation, got {type(timed_observation)}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            "Received observation #%s | avg_fps=%.2f target=%.2f one_way_latency=%.2fms deserialize=%.2fms",
            obs_timestep,
            fps_metrics["avg_fps"],
            fps_metrics["target_fps"],
            (receive_time - obs_timestamp) * 1000,
            deserialize_ms,
        )
        self._trace_pipeline(
            "received_observation",
            "received observation #%s deserialize=%.1fms",
            obs_timestep,
            deserialize_ms,
        )

        self._enqueue_observation(timed_observation)
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        try:
            getactions_started = time.perf_counter()
            self._trace_pipeline("get_actions_wait", "GetActions waiting for observation")
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self._trace_pipeline("get_actions_obs", "GetActions dequeued observation #%s", obs.get_timestep())
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            action_chunk = self._predict_action_chunk(obs)
            actions_bytes = pickle.dumps(action_chunk)  # nosec

            elapsed = time.perf_counter() - getactions_started
            time.sleep(max(0.0, self.config.inference_latency - elapsed))
            return services_pb2.Actions(data=actions_bytes)
        except Empty:
            return services_pb2.Empty()
        except Exception:
            self.logger.exception("Error while producing actions")
            return services_pb2.Empty()

    def _enqueue_observation(self, obs: TimedObservation) -> None:
        if self.observation_queue.full():
            _ = self.observation_queue.get_nowait()
            self.logger.debug("Observation queue was full, removed oldest observation")
        self.observation_queue.put(obs)

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _prepare_aloha_image(self, value: Any) -> np.ndarray:
        """Return an uint8 image in CHW format for OpenPI AlohaInputs.

        Evo-RL camera observations normally arrive as HWC uint8 images, while
        OpenPI's Aloha input transform expects CHW and converts it to HWC
        internally before resizing.
        """

        image = self._to_numpy(value)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8, copy=False)

        if image.ndim == 2:
            image = image[:, :, None]

        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")

        if image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] in (1, 3, 4):
            pass
        else:
            raise ValueError(
                f"Cannot infer image layout for shape {image.shape}; expected HWC or CHW with 1/3/4 channels"
            )

        if image.shape[0] == 4:
            image = image[:3]
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)

        return np.ascontiguousarray(image)

    def _prepare_openpi_observation(self, observation_t: TimedObservation) -> dict[str, Any]:
        if self.lerobot_features is None:
            raise RuntimeError("No policy instructions received yet; lerobot_features are unavailable.")

        raw_observation = observation_t.get_observation()
        lerobot_obs = make_lerobot_observation(raw_observation, self.lerobot_features)

        prompt = raw_observation.get(self.config.task_key) or raw_observation.get(self.config.prompt_key)
        if prompt is None:
            prompt = self.config.default_prompt
        if not prompt:
            self.logger.warning("No task/prompt found in observation and default_prompt is empty.")

        request: dict[str, Any] = {
            "images": {
                self.config.openpi_front_image_key: self._prepare_aloha_image(
                    lerobot_obs[self.config.lerobot_front_image_key]
                ),
                self.config.openpi_left_wrist_image_key: self._prepare_aloha_image(
                    lerobot_obs[self.config.lerobot_left_wrist_image_key]
                ),
                self.config.openpi_right_wrist_image_key: self._prepare_aloha_image(
                    lerobot_obs[self.config.lerobot_right_wrist_image_key]
                ),
            },
            "state": self._to_numpy(lerobot_obs[self.config.state_key]).astype(np.float32),
            "prompt": str(prompt),
        }

        return request

    def _time_action_chunk(self, t_0: float, action_chunk: np.ndarray, i_0: int) -> list[TimedAction]:
        return [
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                action=torch.as_tensor(action, dtype=torch.float32),
            )
            for i, action in enumerate(action_chunk)
        ]

    def _normalize_openpi_actions(self, response: dict[str, Any]) -> np.ndarray:
        if self.config.action_response_key not in response:
            raise KeyError(
                f"OpenPI response missing {self.config.action_response_key!r}. "
                f"Available keys: {sorted(response.keys())}"
            )

        actions = np.asarray(response[self.config.action_response_key], dtype=np.float32)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"Expected batched OpenPI actions with batch size 1, got {actions.shape}")
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"Expected OpenPI actions with shape [horizon, dim], got {actions.shape}")

        if self.config.action_dim is not None:
            if actions.shape[-1] < self.config.action_dim:
                raise ValueError(
                    f"OpenPI returned action dim {actions.shape[-1]}, smaller than requested {self.config.action_dim}"
                )
            actions = actions[:, : self.config.action_dim]

        if self.actions_per_chunk is not None:
            actions = actions[: self.actions_per_chunk]

        return actions

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        start_prepare = time.perf_counter()
        openpi_observation = self._prepare_openpi_observation(observation_t)
        prepare_ms = (time.perf_counter() - start_prepare) * 1000

        client = self._get_openpi_client()

        start_openpi = time.perf_counter()
        self._trace_pipeline("openpi_infer_start", "calling OpenPI infer for obs #%s", observation_t.get_timestep())
        response = client.infer(openpi_observation)
        openpi_ms = (time.perf_counter() - start_openpi) * 1000
        self._trace_pipeline(
            "openpi_infer_done",
            "OpenPI infer returned for obs #%s in %.1fms",
            observation_t.get_timestep(),
            openpi_ms,
        )

        start_post = time.perf_counter()
        actions = self._normalize_openpi_actions(response)
        timed_actions = self._time_action_chunk(
            observation_t.get_timestamp(),
            actions,
            observation_t.get_timestep(),
        )
        post_ms = (time.perf_counter() - start_post) * 1000

        self._action_chunks_generated += 1
        if self._action_chunks_generated % self.config.log_timing_every == 0:
            server_timing = response.get("server_timing", {}) if isinstance(response, dict) else {}
            policy_timing = response.get("policy_timing", {}) if isinstance(response, dict) else {}
            self.logger.info(
                "OpenPI bridge action chunk #%s for obs #%s | shape=%s | prepare=%.2fms "
                "openpi_roundtrip=%.2fms post=%.2fms | openpi_server=%s policy=%s",
                self._action_chunks_generated,
                observation_t.get_timestep(),
                tuple(actions.shape),
                prepare_ms,
                openpi_ms,
                post_ms,
                server_timing,
                policy_timing,
            )

        return timed_actions


@draccus.wrap()
def serve(cfg: OpenPIBridgeServerConfig) -> None:
    logging.info(pformat(asdict(cfg)))

    bridge_server = OpenPIBridgeServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(bridge_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    bridge_server.logger.info(
        "OpenPIBridgeServer started on %s:%s; forwarding to OpenPI websocket %s:%s",
        cfg.host,
        cfg.port,
        cfg.openpi_host,
        cfg.openpi_port,
    )
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
