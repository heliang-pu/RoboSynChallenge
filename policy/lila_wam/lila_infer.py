"""LiLa-WAM inference for RoboSynChallenge (EmbodiChain) environments.

Mirrors upstream ``robotwin_infer.RobotWinInference`` -- flow-matching ODE
sampling, receding-horizon action queue, optional B-spline smoothing -- but
takes its observations from an EmbodiChain gym observation dict and feeds every
configured camera.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from policy.lila_wam._bootstrap import resolve_path
from policy.lila_wam.lerobot_v21 import feature_key_to_camera
from policy.lila_wam.lila_dataset import normalize_image
from policy.lila_wam.lila_model import build_model

logger = logging.getLogger(__name__)


def bspline_smooth(action_seq: np.ndarray, degree: int = 3, num_ctrl_pts: int = 8) -> np.ndarray:
    """Least-squares B-spline smoothing of an ``(N, D)`` action chunk.

    Reimplemented from upstream ``robotwin_infer.bspline_smooth`` (identical
    knot construction) rather than imported: that module resolves
    ``from models.model_runner import ...`` absolutely, which does not work
    under the ``lila_upstream`` namespace, and it pulls in matplotlib.
    """
    from scipy.interpolate import make_lsq_spline

    num_steps = action_seq.shape[0]
    if num_steps <= num_ctrl_pts:
        return action_seq

    x = np.arange(num_steps)
    internal_knots = np.linspace(0, num_steps - 1, (num_ctrl_pts - degree) + 2)[1:-1]
    knots = np.concatenate(
        [[0] * (degree + 1), internal_knots, [num_steps - 1] * (degree + 1)]
    )
    return make_lsq_spline(x, action_seq, knots, k=degree)(x)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _select_env(value: np.ndarray, env_index: int = 0) -> np.ndarray:
    return value if value.ndim == 0 else value[env_index]


def extract_rgb(obs, camera: str, env_index: int = 0) -> np.ndarray:
    """EmbodiChain ``sensor/<camera>/color`` -> ``(H, W, 3)`` uint8 RGB."""
    image = _select_env(_to_numpy(obs["sensor"][camera]["color"]), env_index)
    if image.ndim != 3:
        raise ValueError(f"camera '{camera}' produced a {image.shape} tensor, expected HxWxC")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if float(np.nanmax(image)) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255)
    return np.ascontiguousarray(image.astype(np.uint8))


def extract_state(obs, state_obs_path: str = "robot/qpos", env_index: int = 0) -> np.ndarray:
    node: Any = obs
    for key in str(state_obs_path).split("/"):
        if key:
            node = node[key]
    state = _select_env(_to_numpy(node), env_index).astype(np.float32).reshape(-1)
    return np.ascontiguousarray(state)


class LilaWamInference:
    """Stateful policy handle used by ``deploy_policy.eval``."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        norm_stats_path: str | Path,
        task_cond_dir: str | Path | None = None,
        task_name: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        state_obs_path: str = "robot/qpos",
        action_execution_horizon: int | None = None,
        num_inference_steps: int | None = None,
        exposure_match: dict | None = None,
    ):
        from omegaconf import OmegaConf

        config_path = resolve_path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        self.config = OmegaConf.load(config_path)

        if not torch.cuda.is_available():
            device, dtype = "cpu", torch.float32
        elif not torch.cuda.is_bf16_supported():
            dtype = torch.float32
        self.device = device
        self.dtype = dtype
        self.inference_device = device
        self.state_obs_path = state_obs_path

        self.camera_names = [
            feature_key_to_camera(str(c)) for c in self.config.dataset.camera_names
        ]
        self.image_size = tuple(int(v) for v in self.config.dataset.image_size)
        self.action_len = int(self.config.common.action_chunk_size)
        self.action_dim = int(self.config.common.action_dim)
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else self.config.common.num_inference_steps
        )
        self.action_execution_horizon = int(
            action_execution_horizon
            if action_execution_horizon is not None
            else self.config.common.action_execution_horizon
        )
        inference_cfg = self.config.get("inference", {}) or {}
        self.smooth_actions = bool(inference_cfg.get("smooth_actions", False))
        self._bspline_smooth = bspline_smooth if self.smooth_actions else None

        self.model, self.action_model = build_model(
            self.config,
            norm_stats_path=norm_stats_path,
            device=device,
            dtype=dtype,
            train_config=None,
        )
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.action_model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        # No wrapper-wide .to(dtype) here: the encoder and action model are
        # already in `dtype` from build_model, and casting would drag the
        # fp32 min/max normalization buffers down to bf16 with them.
        logger.info(
            "LiLa-WAM ready: %s (epoch %s), cameras=%s, chunk=%d, horizon=%d, steps=%d",
            checkpoint_path,
            checkpoint.get("epoch", "?"),
            self.camera_names,
            self.action_len,
            self.action_execution_horizon,
            self.num_inference_steps,
        )

        self.use_task_cond = bool(self.config.model.get("use_task_cond", False))
        self.task_cond = None
        configured_task_cond_dir = task_cond_dir or self.config.dataset.get("task_cond_dir", None)
        self.task_cond_dir = resolve_path(configured_task_cond_dir) if configured_task_cond_dir else None
        if self.use_task_cond:
            if task_name is None:
                raise ValueError(
                    "model.use_task_cond=true, so get_model() needs a task_name to look up the VTT vector"
                )
            self.set_task(task_name)

        # 诊断用的曝光匹配(默认关闭)。评测场景可能比训练数据亮得多——
        # 本任务实测 cam_high 训练中位数 105.6、环境 227.9,腕部相机 50 vs 207。
        # 冻结的 DINOv3 从没见过这种输入。这个开关把每帧线性缩放到训练亮度,
        # 用来判断"分布外曝光"是不是失败主因。它**不是**正式方案:
        # 已经饱和的像素信息已丢失,缩放只能恢复统计量,恢复不了细节。
        self.exposure_match = dict(exposure_match) if exposure_match else None
        if self.exposure_match:
            logger.info("曝光匹配已启用: %s", self.exposure_match)

        self.action_queue: deque[np.ndarray] = deque()

    # ------------------------------------------------------------------ setup
    def set_task(self, task_name: str):
        if not self.use_task_cond:
            return
        if self.task_cond_dir is None:
            raise ValueError("model.use_task_cond=true but no task_cond_dir is configured")
        path = self.task_cond_dir / task_name / "task_cond.npy"
        if not path.exists():
            available = sorted(p.name for p in self.task_cond_dir.glob("*/task_cond.npy"))
            raise FileNotFoundError(
                f"no VTT vector for task '{task_name}': {path}\n"
                f"available tasks in {self.task_cond_dir}: "
                + (", ".join(p for p in available) if available else "<none>")
            )
        vector = np.load(path).astype(np.float32)
        self.task_cond = torch.from_numpy(vector).to(self.device, self.dtype).unsqueeze(0)
        logger.info("task condition '%s' loaded (dim=%d)", task_name, vector.shape[0])

    def reset(self):
        self.action_queue.clear()

    # -------------------------------------------------------------- inference
    def encode_obs(self, obs, env_index: int = 0) -> dict[str, torch.Tensor]:
        import cv2

        width, height = self.image_size
        frames = []
        for camera in self.camera_names:
            rgb = extract_rgb(obs, camera, env_index=env_index)
            if (rgb.shape[1], rgb.shape[0]) != (width, height):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            rgb = self._match_exposure(rgb, camera)
            frames.append(normalize_image(rgb))
        pixel_values = torch.from_numpy(np.stack(frames, axis=0)[None]).to(self.device, self.dtype)

        state = extract_state(obs, self.state_obs_path, env_index=env_index)
        expected = int(self.config.common.state_dim)
        if state.shape[0] != expected:
            raise ValueError(
                f"observation '{self.state_obs_path}' has dim {state.shape[0]}, "
                f"but the model was trained with common.state_dim={expected}"
            )
        state_tensor = torch.from_numpy(state).to(self.device, self.dtype).view(1, 1, -1)
        return {"pixel_values": pixel_values, "state": state_tensor}

    def _match_exposure(self, rgb: np.ndarray, camera: str) -> np.ndarray:
        """把单帧线性缩放到该相机的训练亮度(诊断开关,默认关闭)。"""
        if not self.exposure_match:
            return rgb
        target = self.exposure_match.get(camera)
        if target is None:
            return rgb
        current = float(rgb.mean())
        if current < 1e-3:
            return rgb
        scaled = rgb.astype(np.float32) * (float(target) / current)
        return np.clip(scaled, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def predict_chunk(self, obs, env_index: int = 0) -> np.ndarray:
        """One flow-matching rollout -> ``(action_chunk_size, action_dim)`` numpy."""
        batch = self.encode_obs(obs, env_index=env_index)
        qpos = self.model.normalize_state(batch["state"])
        dino_features = self.model.get_vision_features(batch["pixel_values"])

        x_t = torch.randn(
            (1, self.action_len, self.action_dim), device=self.device, dtype=self.dtype
        )
        steps = torch.linspace(
            0, 1, self.num_inference_steps + 1, device=self.device, dtype=self.dtype
        )
        for i in range(self.num_inference_steps):
            dt = steps[i + 1] - steps[i]
            preds = self.action_model(
                t=steps[i].unsqueeze(0),
                noisy_actions=x_t,
                qpos_history=qpos,
                dino_features_list=dino_features,
                task_cond=self.task_cond,
            )
            x_t = x_t + preds["final_pred"] * dt

        actions = self.model.denormalize_action(x_t)[0].float().cpu().numpy()
        if self._bspline_smooth is not None:
            actions = self._bspline_smooth(actions, degree=3, num_ctrl_pts=8)
        return actions

    def get_action(self, obs, env_index: int = 0) -> np.ndarray:
        """Return the next ``action_execution_horizon`` actions to execute."""
        chunk = self.predict_chunk(obs, env_index=env_index)
        return chunk[: self.action_execution_horizon]

    def step(self, obs, env_index: int = 0) -> np.ndarray:
        """Single-action receding-horizon interface (upstream ``step`` semantics)."""
        if not self.action_queue:
            self.action_queue.extend(self.get_action(obs, env_index=env_index))
        return self.action_queue.popleft()


def encode_action(action: Sequence[float] | np.ndarray, env, strict: bool = True) -> torch.Tensor:
    """Policy action -> the ``(1, action_dim)`` tensor EmbodiChain's env.step wants."""
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if array.shape[0] != env_action_dim and (strict or array.shape[0] < env_action_dim):
        raise ValueError(
            f"policy action has dim {array.shape[0]}, but the env expects {env_action_dim}"
        )
    array = array[:env_action_dim]
    return torch.as_tensor(array, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
