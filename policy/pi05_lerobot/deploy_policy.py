# ----------------------------------------------------------------------------
# LeRobot PI0.5 (PyTorch) Policy Adapter for RoboSynChallenge
#
# This is the LeRobot `PI05Policy` port, not the JAX openpi one in policy/pi05.
# It targets the upstream revision that added MEM (short-horizon visual and
# proprioceptive memory), so a MEM checkpoint evaluates without extra glue.
#
# Unified eval interface:
#   get_model(usr_args) -> model
#   eval(env, model, obs) -> (obs, info, truncated, inference_times)
#   reset_model(model) -> None
# ----------------------------------------------------------------------------

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from policy.inference_timing import finish_inference, start_inference

SENSOR_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_IMAGE_KEYS = {f"observation.images.{sensor}": sensor for sensor in SENSOR_ORDER}


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _select_env(value: np.ndarray, env_index: int = 0) -> np.ndarray:
    if value.ndim == 0:
        return value
    return value[env_index]


def _extract_image(obs, sensor_name: str, env_index: int = 0) -> np.ndarray:
    """Return one camera as a contiguous uint8 CHW array."""
    image = _to_numpy(obs["sensor"][sensor_name]["color"])
    image = _select_env(image, env_index=env_index)
    if image.ndim != 3:
        raise ValueError(f"Expected image for {sensor_name} to be 3D, got {image.shape}")

    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        chw = image
    else:
        chw = np.moveaxis(image, -1, 0)

    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)
    elif chw.shape[0] > 3:
        chw = chw[:3]

    if np.issubdtype(chw.dtype, np.floating):
        if float(np.nanmax(chw)) <= 1.0:
            chw = chw * 255.0
        chw = np.clip(chw, 0, 255).astype(np.uint8)
    elif chw.dtype != np.uint8:
        chw = np.clip(chw, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(chw)


def _extract_state(obs, env_index: int = 0) -> np.ndarray:
    value = _to_numpy(obs["robot"]["qpos"])
    value = _select_env(value, env_index=env_index).astype(np.float32, copy=False).reshape(-1)
    return np.ascontiguousarray(value)


def encode_obs(obs, image_keys: dict[str, str] | None = None):
    """Convert an EmbodiChain observation into the LeRobot PI0.5 batch layout.

    Only the keys the policy consumes are sent: every extra array would be
    cloned into the MEM ring buffer once per step for nothing.
    """
    encoded = {"observation.state": _extract_state(obs)}
    for obs_key, sensor_name in (image_keys or DEFAULT_IMAGE_KEYS).items():
        encoded[obs_key] = _extract_image(obs, sensor_name)
    return encoded


def resolve_image_keys(input_feature_keys: list[str]) -> dict[str, str]:
    """Map a checkpoint's camera features onto EmbodiChain sensor names.

    Checkpoints reach us under at least three naming schemes: RoboSyn's own
    (cam_high/cam_left_wrist/cam_right_wrist), LeRobot's generic camera1/2/3,
    and openpi's base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb. Match on the
    name where it is unambiguous, else fall back to the order the features were
    declared in during training.
    """
    image_keys = [key for key in input_feature_keys if key.startswith("observation.image")]
    if not image_keys:
        raise ValueError(f"No image features among {input_feature_keys}")

    def match(key: str) -> str | None:
        name = key.rsplit(".", 1)[-1].lower()
        if "left" in name:
            return "cam_left_wrist"
        if "right" in name:
            return "cam_right_wrist"
        if any(token in name for token in ("high", "top", "base", "front", "head")):
            return "cam_high"
        return None

    mapping = {key: match(key) for key in image_keys}
    sensors = [sensor for sensor in mapping.values() if sensor is not None]
    if len(sensors) == len(image_keys) == len(set(sensors)):
        return mapping

    if len(image_keys) > len(SENSOR_ORDER):
        raise ValueError(
            f"Checkpoint declares {len(image_keys)} cameras {image_keys}, but the env "
            f"only exposes {SENSOR_ORDER}."
        )
    positional = dict(zip(image_keys, SENSOR_ORDER, strict=False))
    print(
        f"[pi05_lerobot] camera names are not self-describing; mapping by declaration "
        f"order: {positional}",
        flush=True,
    )
    return positional


def _resolve_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"auto", "none", ""}:
            return None
        return normalized in {"1", "true", "yes", "y", "on"}
    return bool(value)


def encode_action(
    action,
    env,
    rescale_gripper: bool = True,
    gripper_indices: tuple[int, ...] = (6, 13),
    gripper_source_range: tuple[float, float] = (0.0, 1.0),
):
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.shape[0] < env_action_dim:
        raise ValueError(
            f"Policy action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
        )
    action_array = action_array[:env_action_dim]
    if rescale_gripper:
        action_array = action_array.copy()
        action_space = env.unwrapped.single_action_space
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        src_low, src_high = gripper_source_range
        src_span = max(src_high - src_low, np.finfo(np.float32).eps)
        for gripper_idx in gripper_indices:
            if gripper_idx >= env_action_dim:
                continue
            gripper_01 = np.clip((action_array[gripper_idx] - src_low) / src_span, 0.0, 1.0)
            action_array[gripper_idx] = low[gripper_idx] + gripper_01 * (
                high[gripper_idx] - low[gripper_idx]
            )
    action_tensor = torch.as_tensor(action_array, dtype=torch.float32, device=env.unwrapped.device)
    return action_tensor.unsqueeze(0)


class Pi05LeRobotClient:
    """Drives policy/pi05_lerobot/pi05_worker.py over a stdio JSON pipe."""

    def __init__(self, usr_args: dict):
        policy_root = Path(__file__).resolve().parent
        self.checkpoint_dir = Path(usr_args["checkpoint_dir"]).resolve()
        lerobot_root = (
            usr_args.get("lerobot_root")
            or os.environ.get("PI05_LEROBOT_ROOT")
            or os.environ.get("LEROBOT_ROOT")
            or ""
        )
        self.lerobot_root = Path(lerobot_root).expanduser().resolve() if lerobot_root else None
        self.python_bin = (
            usr_args.get("pi05_lerobot_python")
            or os.environ.get("PI05_LEROBOT_PYTHON")
            or sys.executable
        )
        self.device = usr_args.get("pytorch_device", "cuda")
        self.steps = int(usr_args.get("pi05_lerobot_steps", 10))
        self.memory_stride = int(usr_args.get("pi05_lerobot_memory_stride", 0))
        self.tokenizer = str(usr_args.get("pi05_lerobot_tokenizer", "") or "")
        self.gripper_indices = tuple(
            int(idx) for idx in usr_args.get("pi05_lerobot_gripper_indices", [6, 13])
        )
        source_range = usr_args.get("pi05_lerobot_gripper_source_range", [0.0, 1.0])
        self.gripper_source_range = (float(source_range[0]), float(source_range[1]))
        self.rescale_gripper = self._resolve_gripper_rescale(
            usr_args.get("pi05_lerobot_rescale_gripper", "auto")
        )
        self.cuda_visible_devices = str(
            usr_args.get("pi05_lerobot_cuda_visible_devices", usr_args.get("gpu_id", 0))
        )
        self.image_keys = self._resolve_image_keys()

        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        worker_cmd = [
            self.python_bin,
            str(policy_root / "pi05_worker.py"),
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--device",
            self.device,
            "--n-action-steps",
            str(self.steps),
            "--memory-stride",
            str(self.memory_stride),
        ]
        if self.tokenizer:
            worker_cmd.extend(["--tokenizer", self.tokenizer])
        if self.lerobot_root is not None:
            worker_cmd.extend(["--lerobot-root", str(self.lerobot_root)])
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

        self.info = self._rpc({"cmd": "info"})["info"]
        self.step_mode = self._resolve_step_mode(usr_args.get("pi05_lerobot_step_mode", "auto"))
        self._describe()

    # -- setup helpers ----------------------------------------------------

    def _resolve_step_mode(self, requested) -> str:
        requested = str(requested or "auto").strip().lower()
        if requested not in {"auto", "per_step", "chunk"}:
            raise ValueError(
                f"pi05_lerobot_step_mode must be auto/per_step/chunk, got {requested!r}"
            )
        if requested == "auto":
            # MEM 的历史是按「每次入队一帧」对齐训练时的 delta_indices 的。
            # chunk 模式一个执行周期只喂一帧观测,历史间隔会被拉长 steps 倍,
            # 所以带 MEM 的 checkpoint 默认走逐步模式。
            requested = "per_step" if self.info["memory_enabled"] else "chunk"
        if requested == "per_step" and self.info["rtc_enabled"]:
            # select_action asserts RTC is off; RTC checkpoints must go through
            # predict_action_chunk.
            print(
                "[pi05_lerobot] RTC checkpoint detected: forcing step_mode=chunk. "
                "MEM history spacing will not match training if MEM is also enabled."
            )
            return "chunk"
        return requested

    def _resolve_gripper_rescale(self, override) -> bool:
        override_value = _resolve_optional_bool(override)
        if override_value is not None:
            return override_value

        try:
            from safetensors.torch import load_file

            stats_files = sorted(self.checkpoint_dir.glob("*unnormalizer*.safetensors"))
            if not stats_files:
                raise FileNotFoundError("no unnormalizer stats in checkpoint")
            stats = load_file(str(stats_files[0]))
            # PI0.5 normalizes with QUANTILES, so the upper bound is q99; MIN_MAX
            # checkpoints keep action.max instead.
            upper = stats.get("action.q99")
            if upper is None:
                upper = stats["action.max"]
            action_upper = _to_numpy(upper).reshape(-1)
        except Exception:
            # Most RoboSyn CobotMagic tasks train on 0-1 gripper data while the env
            # expects physical qpos. Prefer the safer conversion if stats are missing.
            return True

        valid_indices = [idx for idx in self.gripper_indices if idx < action_upper.shape[0]]
        gripper_upper = action_upper[valid_indices]
        return bool(gripper_upper.size and float(np.nanmax(gripper_upper)) > 0.5)

    def _resolve_image_keys(self) -> dict[str, str]:
        config = json.loads((self.checkpoint_dir / "config.json").read_text())
        return resolve_image_keys(list(config.get("input_features", {})))

    def _describe(self) -> None:
        info = self.info
        memory = (
            f"visual={info['use_visual_memory']} proprio={info['use_proprioceptive_memory']} "
            f"frames={info['memory_frames']} stride={info['memory_stride']}"
            if info["memory_enabled"]
            else "disabled"
        )
        print(
            f"[pi05_lerobot] step_mode={self.step_mode} steps={self.steps} "
            f"chunk_size={info['chunk_size']} n_action_steps={info['n_action_steps']} "
            f"MEM({memory}) rescale_gripper={self.rescale_gripper}",
            flush=True,
        )

    # -- RPC --------------------------------------------------------------

    def close(self) -> None:
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

    def _rpc(self, payload: dict) -> dict:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("PI0.5 worker pipes are not available.")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    "PI0.5 worker exited unexpectedly; its traceback is on stderr above. "
                    "A checkpoint written by an older LeRobot often fails here because "
                    "config.json still carries fields the current PI05Config has dropped."
                )
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "PI0.5 worker error"))
            return response

    def _infer(self, cmd: str, obs: dict[str, np.ndarray], task: str, **extra) -> dict:
        fd, obs_path = tempfile.mkstemp(prefix="pi05_lerobot_obs_", suffix=".npz")
        os.close(fd)
        try:
            # Uncompressed: one round trip per env step in per_step mode, so the
            # zlib pass would cost more than the pipe it saves.
            np.savez(obs_path, **obs, task=np.asarray(task))
            return self._rpc({"cmd": cmd, "obs_path": obs_path, **extra})
        finally:
            if os.path.exists(obs_path):
                os.remove(obs_path)

    def select_action(self, obs: dict[str, np.ndarray], task: str) -> tuple[np.ndarray, bool]:
        response = self._infer("select_action", obs, task)
        return np.asarray(response["actions"][0], dtype=np.float32), bool(response["planned"])

    def predict_chunk(self, obs: dict[str, np.ndarray], task: str) -> list[np.ndarray]:
        response = self._infer("predict_chunk", obs, task, n_action_steps=self.steps)
        return [np.asarray(action, dtype=np.float32) for action in response["actions"]]

    def reset(self) -> None:
        self._rpc({"cmd": "reset"})


def get_model(usr_args):
    if not usr_args.get("checkpoint_dir"):
        raise ValueError("checkpoint_dir must be provided in deploy_policy.yml or overrides.")
    return Pi05LeRobotClient(usr_args)


def _step_env(env, model, action):
    action_tensor = encode_action(
        action,
        env,
        rescale_gripper=model.rescale_gripper,
        gripper_indices=model.gripper_indices,
        gripper_source_range=model.gripper_source_range,
    )
    return env.step(action_tensor)


def _is_truncated(truncated) -> bool:
    return bool(_to_numpy(truncated).reshape(-1)[0])


def eval(env, model, obs):
    """Advance the environment by one execution horizon.

    ``per_step`` mirrors ``lerobot-eval``: every environment step feeds a fresh
    observation to ``select_action``, so MEM sees history at the same cadence it
    was trained on, and the policy replans on its own ``n_action_steps`` queue.
    ``chunk`` samples one action chunk per horizon, matching the openpi adapter
    in policy/pi05 — cheaper, but only correct when the checkpoint has no MEM.
    """
    instruction = getattr(env, "_current_instruction", None) or ""
    inference_times: list[float] = []
    info: dict = {}
    truncated = False
    final_obs = obs

    if model.step_mode == "per_step":
        for _ in range(model.steps):
            started_at = start_inference(model.device)
            action, planned = model.select_action(
                encode_obs(final_obs, image_keys=model.image_keys), task=instruction
            )
            if planned:
                finish_inference(started_at, inference_times, model.device)
            final_obs, _reward, _terminated, truncated, info = _step_env(env, model, action)
            if _is_truncated(truncated):
                break
        return final_obs, info, truncated, inference_times

    started_at = start_inference(model.device)
    actions = model.predict_chunk(encode_obs(obs, image_keys=model.image_keys), task=instruction)
    finish_inference(started_at, inference_times, model.device)

    for action in actions:
        final_obs, _reward, _terminated, truncated, info = _step_env(env, model, action)
        if _is_truncated(truncated):
            break

    return final_obs, info, truncated, inference_times


def reset_model(model):
    model.reset()


def close_model(model):
    model.close()
