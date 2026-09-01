"""Model construction shared by LiLa-WAM training and deployment.

Everything structural comes from the pinned upstream checkout; this module only
adds what RoboSynChallenge needs on top:

* a vision-encoder loader that reads ``num_register_tokens`` from the encoder
  config instead of assuming 4 (upstream's default is right for DINOv3 but
  silently wrong for encoders without registers, e.g. DINOv2);
* multi-camera support -- the DINOv3 tokens of every configured camera are
  concatenated along the token axis, with a learnable per-camera embedding so
  the perceiver adapter can tell the views apart.  With one camera the model is
  bit-for-bit upstream (no camera embedding is created at all).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from policy.lila_wam._bootstrap import resolve_path
from policy.lila_wam._upstream import ensure_upstream, upstream_models

logger = logging.getLogger(__name__)


def load_vision_encoder(checkpoint_path: str, dtype: torch.dtype, device: str | torch.device):
    """Load the frozen ViT encoder; returns ``(model, hidden_size, registers, patch)``."""
    ModelFactory, _, _ = upstream_models()
    from transformers import AutoConfig

    model, hidden_size, _registers, patch_size = ModelFactory.create_vision_encoder(
        checkpoint_path, dtype, device
    )
    # Upstream falls back to 4 registers when the config is silent; that is only
    # correct for DINOv3. Trust the config, default to 0 otherwise.
    encoder_config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)
    num_register_tokens = int(getattr(encoder_config, "num_register_tokens", 0) or 0)
    logger.info(
        "vision encoder: hidden=%d patch=%d registers=%d", hidden_size, patch_size, num_register_tokens
    )
    return model, hidden_size, num_register_tokens, patch_size


def _make_wrapper_class():
    _, VLAWrapper, _ = upstream_models()

    class _MultiCameraVLAWrapper(VLAWrapper):
        """VLAWrapper that accepts ``pixel_values`` of shape ``(B, C, 3, H, W)``."""

        def __init__(self, *args, camera_names: Sequence[str], **kwargs):
            super().__init__(*args, **kwargs)
            if not camera_names:
                raise ValueError("camera_names must list at least one camera")
            self.camera_names = list(camera_names)

        @property
        def camera_emb(self):
            return getattr(self.action_model, "camera_emb", None)

        def get_vision_features(self, pixel_values):
            """Frozen DINO features for every camera, concatenated over tokens.

            Returns one ``(B, C * N, D)`` tensor per entry of ``feat_layers``.
            """
            if pixel_values.dim() == 4:          # (B, 3, H, W) -> single camera
                pixel_values = pixel_values.unsqueeze(1)
            if pixel_values.dim() != 5:
                raise ValueError(
                    f"pixel_values must be (B, 3, H, W) or (B, C, 3, H, W), got {tuple(pixel_values.shape)}"
                )
            batch, cams = pixel_values.shape[:2]
            if cams != len(self.camera_names):
                raise ValueError(
                    f"batch carries {cams} cameras but the model was built for "
                    f"{len(self.camera_names)} ({self.camera_names})"
                )

            flat = pixel_values.reshape(batch * cams, *pixel_values.shape[2:])
            # The encoder itself stays frozen and runs under no_grad (upstream).
            per_layer = super().get_vision_features(flat)

            camera_emb = self.camera_emb
            out = []
            for feats in per_layer:
                # reshape rather than view: with include_cls_register=false the
                # upstream token slice leaves a non-contiguous tensor.
                tokens, dim = feats.shape[-2], feats.shape[-1]
                feats = feats.reshape(batch, cams, tokens, dim)
                if camera_emb is not None:
                    # Applied outside the frozen no_grad block so it stays trainable.
                    feats = feats + camera_emb.to(device=feats.device, dtype=feats.dtype).reshape(
                        1, cams, 1, dim
                    )
                out.append(feats.reshape(batch, cams * tokens, dim))
            return out

    return _MultiCameraVLAWrapper


def build_action_model(
    config,
    dino_hidden_size: int,
    num_dino_layers: int,
    task_cond_dim: int | None,
    patch_size: int,
    num_cameras: int,
):
    ModelFactory, _, _ = upstream_models()
    action_model = ModelFactory.create_action_model(
        config,
        dino_hidden_size=dino_hidden_size,
        num_dino_layers=num_dino_layers,
        task_cond_dim=task_cond_dim,
        patch_size=patch_size,
    )
    if num_cameras > 1:
        # Zero-init: at step 0 the model behaves exactly like the single-camera
        # one, and the embedding is learned from there. Registered on the action
        # model so it is optimized and saved with the rest of the weights.
        action_model.register_parameter(
            "camera_emb", nn.Parameter(torch.zeros(num_cameras, dino_hidden_size))
        )
        logger.info("multi-camera mode: %d cameras, learnable camera embedding added", num_cameras)
    return action_model


def build_model(
    config,
    norm_stats_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    train_config: dict | None = None,
):
    """Build ``(wrapper, action_model)`` from an OmegaConf config.

    The same call is used by ``train_lila_wam.py`` and by ``deploy_policy.py``,
    so training and evaluation cannot drift apart.
    """
    ensure_upstream()

    camera_names = [str(c) for c in config.dataset.camera_names]
    encoder_path = resolve_path(config.model.vision_encoder.checkpoint_path)
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"vision encoder weights not found: {encoder_path}\n"
            f"Download them once: bash policy/lila_wam/setup_env.sh --download-encoder"
        )
    vision_encoder, dino_hidden_size, num_register_tokens, patch_size = load_vision_encoder(
        str(encoder_path), dtype, device
    )

    width, height = (int(v) for v in config.dataset.image_size)
    if width % patch_size or height % patch_size:
        raise ValueError(
            f"dataset.image_size {(width, height)} must be a multiple of the encoder patch size {patch_size}"
        )

    feat_layers = list(config.model.vision_encoder.feat_layers)
    use_task_cond = bool(config.model.get("use_task_cond", False))
    task_cond_dim = dino_hidden_size if use_task_cond else None

    action_model = build_action_model(
        config,
        dino_hidden_size=dino_hidden_size,
        num_dino_layers=len(feat_layers),
        task_cond_dim=task_cond_dim,
        patch_size=patch_size,
        num_cameras=len(camera_names),
    )
    action_model.to(device, dtype=dtype)

    future_cfg = config.model.get("future_feat", {}) or {}
    wrapper_cls = _make_wrapper_class()
    wrapper = wrapper_cls(
        vision_encoder=vision_encoder,
        action_model=action_model,
        time_sampler=config.training.time_sampler,
        feat_layers=feat_layers,
        include_cls_register=bool(config.model.vision_encoder.include_cls_register),
        num_register_tokens=num_register_tokens,
        device=device,
        dtype=dtype,
        norm_stats_path=str(norm_stats_path),
        train_config=train_config,
        future_feat_target_layer=int(future_cfg.get("target_layer", -1)),
        camera_names=camera_names,
    )
    return wrapper, action_model
