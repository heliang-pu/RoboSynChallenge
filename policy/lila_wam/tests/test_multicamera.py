"""Multi-camera wrapper tests.

The camera axis is the one piece of modelling this adapter adds on top of
upstream, so it gets its own tests. The DINOv3 encoder is stubbed out (a tiny
module returning fake hidden states), which keeps these runnable without the
gated weights -- and without `transformers`, which upstream imports at module
scope purely for the encoder loader.

    python -m pytest policy/lila_wam/tests/test_multicamera.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HIDDEN = 32
PATCHES = 6          # tokens per image before CLS/registers
REGISTERS = 4
ACTION_DIM = 14
STATE_DIM = 14
CHUNK = 4
FEAT_LAYERS = [-2, -1]


def _stub_transformers():
    """Upstream imports transformers for the encoder loader we do not use here."""
    if "transformers" in sys.modules:
        return
    module = types.ModuleType("transformers")

    class _Unavailable:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("stubbed out in tests")

    module.AutoModel = _Unavailable
    module.AutoConfig = _Unavailable
    sys.modules["transformers"] = module


_stub_transformers()

from policy.lila_wam.lila_model import build_action_model, _make_wrapper_class  # noqa: E402


class StubEncoder(nn.Module):
    """Stands in for DINOv3: emits deterministic per-image hidden states."""

    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.calls = 0

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        self.calls += 1
        batch = pixel_values.shape[0]
        tokens = 1 + REGISTERS + PATCHES
        # Encode the image's own mean into the features so we can tell images apart.
        signature = pixel_values.reshape(batch, -1).mean(dim=1).view(batch, 1, 1)
        base = torch.ones(batch, tokens, HIDDEN, device=pixel_values.device, dtype=pixel_values.dtype)
        hidden_states = tuple(base * signature * (i + 1) for i in range(self.num_layers + 1))
        return types.SimpleNamespace(hidden_states=hidden_states, last_hidden_state=hidden_states[-1])


class Cfg(dict):
    """Minimal stand-in for OmegaConf: attribute access plus dict access.

    Keeps this test runnable in any environment that has torch, which matters
    because it is the only check of the camera axis that does not need the
    gated DINOv3 weights.
    """

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Cfg(value) if isinstance(value, dict) else value

    def get(self, name, default=None):
        value = dict.get(self, name, default)
        return Cfg(value) if isinstance(value, dict) else value


def _config(camera_names):
    return Cfg(
        {
            "common": {
                "action_dim": ACTION_DIM,
                "state_dim": STATE_DIM,
                "action_chunk_size": CHUNK,
                "proprio_len": 1,
                "num_inference_steps": 2,
                "action_execution_horizon": 2,
            },
            "dataset": {
                "camera_names": list(camera_names),
                "image_size": [PATCHES * 16, 16],
                "indices_config": {
                    "state_indices": [0],
                    "camera_indices": [0],
                    "action_indices": list(range(CHUNK)),
                },
            },
            "model": {
                "num_registers": 2,
                "use_task_cond": False,
                "vision_encoder": {
                    "checkpoint_path": "unused",
                    "feat_layers": FEAT_LAYERS,
                    "include_cls_register": True,
                    "fusion_mode": "concat",
                    "concat": {"proj_type": "linear", "pre_norm": True, "out_dim": None},
                },
                "future_feat": {
                    "enabled": False,
                    "target_layer": -1,
                    "num_decoder_layers": 1,
                    "num_heads": 2,
                },
                "action_expert": {
                    "depth": 1,
                    "hidden_size": 32,
                    "num_heads": 2,
                    "vlm_adapter_num_queries": 4,
                    "adapter_depth": 1,
                },
            },
            "training": {"time_sampler": "uniform"},
        }
    )


def _norm_stats(tmp_path: Path) -> Path:
    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "robotwin2": {
                    "action": {"min": [-1.0] * ACTION_DIM, "max": [1.0] * ACTION_DIM},
                    "state": {"min": [-1.0] * STATE_DIM, "max": [1.0] * STATE_DIM},
                }
            }
        )
    )
    return path


def _build(camera_names, tmp_path):
    config = _config(camera_names)
    action_model = build_action_model(
        config,
        dino_hidden_size=HIDDEN,
        num_dino_layers=len(FEAT_LAYERS),
        task_cond_dim=None,
        patch_size=16,
        num_cameras=len(camera_names),
    )
    wrapper_cls = _make_wrapper_class()
    wrapper = wrapper_cls(
        vision_encoder=StubEncoder(),
        action_model=action_model,
        time_sampler="uniform",
        feat_layers=FEAT_LAYERS,
        include_cls_register=True,
        num_register_tokens=REGISTERS,
        device="cpu",
        dtype=torch.float32,
        norm_stats_path=str(_norm_stats(tmp_path)),
        train_config={
            "time_mu": 0.0,
            "time_sigma": 1.0,
            "use_vel_weight": False,
            "vel_weight_alpha": 0.2,
            "vel_weight_sigma": 0.01,
            "use_future_feat": False,
            "lambda_future_feat": 0.0,
        },
        camera_names=list(camera_names),
    )
    return wrapper, action_model


def _batch(num_cameras, batch_size=2):
    height, width = 16, PATCHES * 16
    pixel_values = torch.arange(
        batch_size * num_cameras * 3 * height * width, dtype=torch.float32
    ).reshape(batch_size, num_cameras, 3, height, width)
    pixel_values = pixel_values / pixel_values.max()
    return {
        "pixel_values": pixel_values,
        "state": torch.zeros(batch_size, 1, STATE_DIM),
        "action_sequence": torch.zeros(batch_size, CHUNK, ACTION_DIM),
    }


def test_single_camera_model_has_no_camera_embedding(tmp_path):
    _, action_model = _build(["cam_high"], tmp_path)
    assert not hasattr(action_model, "camera_emb")
    assert "camera_emb" not in action_model.state_dict()


def test_camera_embedding_is_zero_init_and_checkpointed(tmp_path):
    _, action_model = _build(["cam_high", "cam_left_wrist", "cam_right_wrist"], tmp_path)
    assert action_model.camera_emb.shape == (3, HIDDEN)
    # Zero init: step 0 behaves exactly like the single-camera model.
    assert torch.count_nonzero(action_model.camera_emb) == 0
    assert "camera_emb" in action_model.state_dict()
    # It must reach the optimizer through the usual parameter list.
    assert any(p is action_model.camera_emb for p in action_model.parameters())


@pytest.mark.parametrize("num_cameras", [1, 3])
def test_token_count_scales_with_cameras(num_cameras, tmp_path):
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"][:num_cameras]
    wrapper, _ = _build(cameras, tmp_path)
    batch = _batch(num_cameras)

    features = wrapper.get_vision_features(batch["pixel_values"])
    assert len(features) == len(FEAT_LAYERS)
    tokens_per_image = 1 + REGISTERS + PATCHES
    for layer in features:
        assert layer.shape == (2, num_cameras * tokens_per_image, HIDDEN)
    # One encoder call per forward, batching all cameras together.
    assert wrapper.vision_encoder.calls == 1


def test_a_4d_batch_is_accepted_as_a_single_camera(tmp_path):
    wrapper, _ = _build(["cam_high"], tmp_path)
    batch = _batch(1)
    squeezed = wrapper.get_vision_features(batch["pixel_values"].squeeze(1))
    kept = wrapper.get_vision_features(batch["pixel_values"])
    for a, b in zip(squeezed, kept):
        assert torch.equal(a, b)


def test_camera_count_mismatch_is_reported(tmp_path):
    wrapper, _ = _build(["cam_high", "cam_left_wrist"], tmp_path)
    with pytest.raises(ValueError, match="batch carries 3 cameras"):
        wrapper.get_vision_features(_batch(3)["pixel_values"])


def test_camera_tokens_stay_distinct_and_ordered(tmp_path):
    """Camera k's tokens must land in slot k, carrying camera k's embedding."""
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    wrapper, action_model = _build(cameras, tmp_path)
    with torch.no_grad():
        action_model.camera_emb.copy_(
            torch.tensor([[1.0], [2.0], [3.0]]).expand(3, HIDDEN).contiguous()
        )

    # Feed the SAME image to every camera, so any difference between the token
    # blocks can only come from the camera embedding.
    one_image = _batch(1, batch_size=1)["pixel_values"][:, 0]
    pixel_values = one_image.unsqueeze(1).expand(1, 3, *one_image.shape[1:]).contiguous()

    features = wrapper.get_vision_features(pixel_values)[0]
    tokens = 1 + REGISTERS + PATCHES
    assert features.shape == (1, 3 * tokens, HIDDEN)

    blocks = [features[:, k * tokens : (k + 1) * tokens, :] for k in range(3)]
    for k in (1, 2):
        # camera_emb[k] - camera_emb[0] == (k + 1) - 1 == k
        assert torch.allclose(blocks[k] - blocks[0], torch.full_like(blocks[0], float(k)), atol=1e-5)


def test_gradients_reach_the_camera_embedding(tmp_path):
    cameras = ["cam_high", "cam_left_wrist"]
    wrapper, action_model = _build(cameras, tmp_path)

    # The DiT uses adaLN-zero: every block's gates start at exactly 0, so at
    # initialization NOTHING downstream of the vision tokens receives gradient
    # (that is upstream's design, not a wiring bug). Open the gates so this test
    # measures what it means to measure.
    for block in action_model.blocks:
        nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)

    loss, _info = wrapper(_batch(2))
    loss.backward()
    assert action_model.camera_emb.grad is not None
    assert torch.count_nonzero(action_model.camera_emb.grad) > 0


def test_forward_runs_end_to_end_for_both_camera_counts(tmp_path):
    for cameras in (["cam_high"], ["cam_high", "cam_left_wrist", "cam_right_wrist"]):
        wrapper, _ = _build(cameras, tmp_path)
        loss, info = wrapper(_batch(len(cameras)))
        assert torch.isfinite(loss)
        assert info["pred_v"].shape == (2, CHUNK, ACTION_DIM)
