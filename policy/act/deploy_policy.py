# ----------------------------------------------------------------------------
# LeRobot ACT Policy Adapter for RoboSynChallenge
#
# Follows the same unified evaluation interface as policy/pi0:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info, truncated
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import atexit
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from policy.inference_timing import finish_inference, start_inference


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class ACTWorkerClient:
    """Run ACT in its training-time LeRobot environment over line-delimited RPC."""

    def __init__(self, usr_args):
        self.checkpoint_path = Path(usr_args["checkpoint_path"]).resolve()
        self.python_bin = (
            usr_args.get("act_python")
            or os.environ.get("ACT_PYTHON")
            or sys.executable
        )
        self.device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
        self.act_step = int(usr_args.get("act_step", 8))
        self.state_obs_path = usr_args.get("state_obs_path", "robot/qpos")
        self.strict_action_dim = bool(usr_args.get("strict_action_dim", True))

        config = json.loads((self.checkpoint_path / "config.json").read_text())
        input_features = config.get("input_features", {})
        self.act_image_keys = [
            key
            for key, feature in input_features.items()
            if feature.get("type") == "VISUAL"
        ]
        default_map = {
            "observation.images.cam_high": "cam_high",
            "observation.images.cam_right_wrist": "cam_right_wrist",
            "observation.images.cam_left_wrist": "cam_left_wrist",
        }
        default_map.update(usr_args.get("image_key_map") or {})
        self.image_key_map = default_map

        worker_path = Path(__file__).resolve().parent / "act_worker.py"
        worker_env = os.environ.copy()
        worker_cmd = [
            str(self.python_bin),
            str(worker_path),
            "--checkpoint-dir",
            str(self.checkpoint_path),
            "--device",
            str(self.device),
        ]
        self.proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        atexit.register(self.close)

    def close(self):
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            return
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def _rpc(self, payload):
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("ACT worker pipes are unavailable")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("ACT worker exited unexpectedly")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "ACT worker error"))
            return response

    @staticmethod
    def _select_first_env(value):
        array = _to_numpy(value)
        if array.ndim >= 2 and array.shape[0] == 1:
            array = array[0]
        return np.ascontiguousarray(array)

    def infer(self, obs):
        state = obs
        for key in str(self.state_obs_path).split("/"):
            if key:
                state = state[key]
        encoded = {
            "observation.state": self._select_first_env(state).astype(
                np.float32, copy=False
            )
        }
        for image_key in self.act_image_keys:
            camera_name = self.image_key_map.get(
                image_key, image_key.removeprefix("observation.images.")
            )
            encoded[image_key] = self._select_first_env(
                obs["sensor"][camera_name]["color"]
            )

        fd, obs_path = tempfile.mkstemp(prefix="act_obs_", suffix=".npz")
        os.close(fd)
        try:
            np.savez(obs_path, **encoded)
            response = self._rpc(
                {
                    "cmd": "infer",
                    "obs_path": obs_path,
                    "n_action_steps": self.act_step,
                }
            )
        finally:
            if os.path.exists(obs_path):
                os.remove(obs_path)
        return [np.asarray(action, dtype=np.float32) for action in response["actions"]]

    def reset(self):
        self._rpc({"cmd": "reset"})


def get_model(usr_args):
    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in usr_args.")

    worker_python = usr_args.get("act_python") or os.environ.get("ACT_PYTHON")
    if worker_python:
        return ACTWorkerClient(usr_args)

    device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
    cli_overrides = [f"--device={device}"]
    try:
        policy = ACTPolicy.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
    except TypeError as exc:
        if "cli_overrides" not in str(exc):
            raise
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
        policy = ACTPolicy.from_pretrained(checkpoint_path, config=config)
    policy.eval()

    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(usr_args.get("image_key_map") or {})
    image_keys = list(policy.config.image_features.keys())
    for image_key in image_keys:
        image_key_map.setdefault(
            image_key,
            image_key.removeprefix("observation.images."),
        )

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 8))
    policy.state_obs_path = usr_args.get("state_obs_path", "robot/qpos")
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.act_image_keys = image_keys
    policy.image_key_map = image_key_map

    if policy.act_step <= 0:
        raise ValueError(f"act_step must be positive, got {policy.act_step}.")

    return policy


def eval(env, model, obs):
    if isinstance(model, ACTWorkerClient):
        started_at = time.perf_counter()
        actions = model.infer(obs)
        inference_times_s = [time.perf_counter() - started_at]
        final_obs = obs
        info = None
        truncated = False
        for action in actions:
            action_tensor = torch.as_tensor(
                action,
                dtype=torch.float32,
                device=env.unwrapped.device,
            ).reshape(1, -1)
            env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
            policy_action_dim = int(action_tensor.shape[-1])
            if policy_action_dim != env_action_dim:
                message = (
                    f"Policy action has dim {policy_action_dim}, "
                    f"but env expects {env_action_dim}."
                )
                if model.strict_action_dim or policy_action_dim < env_action_dim:
                    raise ValueError(message)
                action_tensor = action_tensor[:, :env_action_dim]
            final_obs, reward, terminated, truncated, info = env.step(action_tensor)
            if env.get_wrapper_attr("is_task_success")():
                break
            if isinstance(truncated, torch.Tensor):
                is_truncated = truncated.any().item()
            elif isinstance(truncated, np.ndarray):
                is_truncated = truncated.any()
            else:
                is_truncated = bool(truncated)
            if is_truncated:
                break
        return final_obs, info, truncated, inference_times_s

    final_obs = obs
    info = None
    truncated = False
    inference_times_s = []

    for _ in range(model.act_step):
        runs_model_inference = (
            model.config.temporal_ensemble_coeff is not None
            or len(model._action_queue) == 0
        )
        started_at = start_inference(model.act_device) if runs_model_inference else None
        state = final_obs
        for key in str(model.state_obs_path).split("/"):
            if key:
                state = state[key]
        state_tensor = state.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=torch.float32, device=model.act_device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        batch = {"observation.state": state_tensor}
        for image_key in model.act_image_keys:
            camera_name = model.image_key_map.get(
                image_key,
                image_key.removeprefix("observation.images."),
            )
            image = final_obs["sensor"][camera_name]["color"]
            image_tensor = image.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(image, torch.Tensor) else torch.as_tensor(image, dtype=torch.float32, device=model.act_device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
            if torch.max(image_tensor) > 1.5:
                image_tensor = image_tensor / 255.0
            batch[image_key] = image_tensor

        action = model.select_action(batch)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(f"Expected policy action shape [B, D], but got {tuple(action.shape)}.")

        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        policy_action_dim = int(action.shape[-1])
        if policy_action_dim != env_action_dim:
            message = f"Policy action has dim {policy_action_dim}, but env expects {env_action_dim}."
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        action_tensor = action.detach().to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        if runs_model_inference:
            finish_inference(started_at, inference_times_s, model.act_device)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if env.get_wrapper_attr("is_task_success")():
            break
        if isinstance(truncated, torch.Tensor):
            is_truncated = truncated.any().item()
        elif isinstance(truncated, np.ndarray):
            is_truncated = truncated.any()
        else:
            is_truncated = bool(truncated)
        if is_truncated:
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    model.reset()


def close_model(model):
    if isinstance(model, ACTWorkerClient):
        model.close()
