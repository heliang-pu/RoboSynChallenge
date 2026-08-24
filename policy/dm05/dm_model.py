# -- coding: UTF-8
"""HTTP client for the OpenDM DM0.5 inference service.

The DM0.5 model runs as a standalone HTTP service (see policy/dm05/README.md
and launch/run_dm05_server.sh). This client posts the current observation
(3 camera images + joint state + instruction) to `POST /v1/infer` and receives
an absolute joint-position action chunk.

Keeping the model out-of-process means the simulator environment does not need
any of opendm's heavy dependencies (TensorRT / Triton / transformers, etc.).
"""

import base64
import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np


class DM05:

    def __init__(
        self,
        server_url="http://127.0.0.1:7891",
        dm_step=30,
        robot_type=None,
        control_mode=None,
        speed=None,
        state_indices=None,
        timeout=60.0,
        jpeg_quality=95,
        seed=None,
    ):
        self.server_url = server_url.rstrip("/")
        self.dm_step = dm_step
        self.robot_type = robot_type
        self.control_mode = control_mode
        self.speed = speed
        # Optional subset/reorder of qpos dims so the state matches the
        # checkpoint's norm_stats (e.g. pick 14 of 16 sim joints).
        self.state_indices = list(state_indices) if state_indices else None
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        self.seed = seed

        self.instruction = None
        self.observation_window = None
        self._check_server()

    def _check_server(self):
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(self.server_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        try:
            with socket.create_connection((host, port), timeout=5):
                pass
        except OSError as e:
            raise ConnectionError(
                f"Cannot reach DM05 inference service at {self.server_url} ({e}).\n"
                "Start it first, e.g.:\n"
                "  bash launch/run_dm05_server.sh <checkpoint_dir>\n"
                "See policy/dm05/README.md for details."
            ) from e
        print(f"DM05 inference service reachable at {self.server_url}")

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"\nsuccessfully set instruction: {instruction}")

    def _encode_image(self, img):
        """RGB (H, W, 3) uint8 -> base64 JPEG string."""
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed while preparing DM05 request")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def update_observation_window(self, img_arr, state):
        """img_arr: [head, left_wrist, right_wrist] RGB arrays; state: qpos."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.state_indices is not None:
            state = state[self.state_indices]
        self.observation_window = {
            "images": img_arr,
            "state": state,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"

        observation = {
            "prompt": self.instruction or "",
            "state": [float(x) for x in self.observation_window["state"]],
            "images": {
                str(i + 1): self._encode_image(img)
                for i, img in enumerate(self.observation_window["images"])
            },
        }
        if self.robot_type is not None:
            observation["robot_type"] = self.robot_type
        if self.control_mode is not None:
            observation["control_mode"] = self.control_mode
        if self.speed is not None:
            observation["speed"] = str(self.speed)

        payload = {"observation": observation}
        if self.seed is not None:
            payload["sampling"] = {"seed": int(self.seed)}

        request = urllib.request.Request(
            f"{self.server_url}/v1/infer",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"DM05 service returned HTTP {e.code}: {detail}\n"
                "Check that chunk_size / action dim / image count of the running "
                "service match the checkpoint (see policy/dm05/README.md)."
            ) from e

        actions = np.asarray(body["actions"], dtype=np.float32)
        return actions

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language instruction")
