# ----------------------------------------------------------------------------
# Motus (thu-ml) policy wrapper for RoboSynChallenge.
#
# Wraps the upstream in-process inference stack (policy/motus/Motus/inference/
# robotwin/Motus) behind the observation-window API the challenge adapters use:
#
#     set_language(instruction)
#     update_observation_window(img_arr, state)
#     get_action() -> np.ndarray [n_steps, 14] absolute joint positions
#     reset_obsrvationwindows()
#
# Everything runs in-process; there is no server/client split.
# ----------------------------------------------------------------------------

from __future__ import annotations

import gc
import hashlib
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

POLICY_DIR = Path(__file__).resolve().parent
# Upstream ships a self-contained inference tree; we import it rather than vendor it.
MOTUS_INFER_ROOT = POLICY_DIR / "Motus" / "inference" / "robotwin" / "Motus"

# Prompt prefix baked into the RoboTwin2 training metas by
# Motus/data/robotwin2/robotwin_data_convert/robotwin_converter.py. The upstream
# deploy script prepends the identical string, so train/deploy stay aligned.
SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)


# ---------------------------------------------------------------------------
# Upstream import plumbing
# ---------------------------------------------------------------------------
def _ensure_cuda_home():
    """Point CUDA_HOME at the venv's nvcc shim when no real toolkit is present.

    Motus' utils/common.py does ``import deepspeed.comm.comm``, and importing
    deepspeed runs ``builder.is_compatible()``, which shells out to
    ``$CUDA_HOME/bin/nvcc -V`` just to read a version number. This box has no
    CUDA toolkit (torch ships only runtime libs), so deepspeed raises
    MissingCUDAException at import time. setup_env.sh installs a version-only
    shim at .venv/.cuda-shim; we select it here so evaluation works no matter
    how the process was launched. No CUDA op is ever compiled at inference.
    """
    def _publish(path: str):
        os.environ["CUDA_HOME"] = path
        # torch.utils.cpp_extension.CUDA_HOME is a module-level constant computed
        # at first import, and deepspeed reads *that*, not the environment. If
        # anything already imported it (e.g. a previous failed `import
        # deepspeed`), setting the env var alone is a no-op — patch it directly.
        mod = sys.modules.get("torch.utils.cpp_extension")
        if mod is not None and getattr(mod, "CUDA_HOME", None) != path:
            mod.CUDA_HOME = path
        # A failed `import deepspeed` leaves half-initialised submodules behind;
        # drop them so the retry re-executes with CUDA_HOME visible.
        for key in [k for k in sys.modules if k == "deepspeed" or k.startswith("deepspeed.")]:
            sys.modules.pop(key, None)

    candidates = []
    if os.environ.get("CUDA_HOME"):
        candidates.append(os.environ["CUDA_HOME"])
    candidates += ["/usr/local/cuda", "/usr/local/cuda-12", str(Path(sys.prefix) / ".cuda-shim")]

    for candidate in candidates:
        if Path(candidate, "bin", "nvcc").exists():
            _publish(candidate)
            logger.debug("CUDA_HOME -> %s", candidate)
            return

    logger.warning(
        "No CUDA toolkit and no nvcc shim at %s; importing deepspeed (a hard "
        "dependency of Motus' utils/common.py) will likely fail. "
        "Re-run policy/motus/setup_env.sh.", Path(sys.prefix) / ".cuda-shim",
    )


def _sdpa_flash_attention(
    q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None,
    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
    dtype=None, version=None,
):
    """Drop-in replacement for wan.modules.attention.flash_attention using SDPA.

    Upstream's ``attention()`` dispatcher already falls back to
    ``scaled_dot_product_attention``, but Motus never calls it: model.py,
    action_expert.py and und_expert.py all import and call ``flash_attention``
    directly, and that function ends in a bare ``assert FLASH_ATTN_2_AVAILABLE``.
    Without the flash-attn package that assert fires on the first denoising step.

    Layout matches upstream: q/k/v are [B, L, N, C]; the return is [B, Lq, N, C]
    in q's original dtype. Unlike upstream's fallback we also honour
    ``softmax_scale``, ``q_scale`` and ``k_lens`` padding, so results match the
    flash-attn path rather than merely approximating it.
    """
    import torch
    import torch.nn.functional as F

    if dtype is None:
        dtype = torch.bfloat16
    out_dtype = q.dtype
    lk = k.shape[1]

    if q_scale is not None:
        q = q * q_scale

    qt = q.transpose(1, 2).to(dtype)   # [B, N, Lq, C]
    kt = k.transpose(1, 2).to(dtype)
    vt = v.transpose(1, 2).to(dtype)

    attn_mask = None
    if k_lens is not None:
        kl = k_lens.to(device=kt.device).reshape(-1).to(torch.long)
        if bool((kl != lk).any()):
            keep = torch.arange(lk, device=kt.device)[None, :] < kl[:, None]
            attn_mask = keep[:, None, None, :]          # [B, 1, 1, Lk]
            if causal:
                lq = qt.shape[2]
                tri = torch.ones(lq, lk, dtype=torch.bool, device=kt.device).tril(lk - lq)
                attn_mask = attn_mask & tri[None, None, :, :]

    if window_size not in ((-1, -1), [-1, -1], None):
        logger.warning("sdpa fallback ignores sliding window_size=%s", window_size)

    out = F.scaled_dot_product_attention(
        qt, kt, vt,
        attn_mask=attn_mask,
        is_causal=bool(causal) and attn_mask is None,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous().type(out_dtype)


def _install_sdpa_attention_fallback():
    """Swap flash_attention for the SDPA version in every module that bound it.

    ``from .attention import flash_attention`` copies the reference, so patching
    only wan.modules.attention would leave the real call sites untouched.
    """
    try:
        import flash_attn  # noqa: F401
        return False
    except ImportError:
        pass

    patched = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        fn = getattr(module, "flash_attention", None)
        if fn is None or not callable(fn) or getattr(fn, "_motus_sdpa", False):
            continue
        if getattr(fn, "__module__", "").endswith("wan.modules.attention"):
            setattr(module, "flash_attention", _sdpa_flash_attention)
            patched.append(name)
    _sdpa_flash_attention._motus_sdpa = True
    if patched:
        logger.warning(
            "flash-attn not installed: routed flash_attention -> SDPA in %s. "
            "Same math (PyTorch picks its own fused kernel), slightly slower.",
            ", ".join(sorted(patched)),
        )
    return bool(patched)


def _import_motus():
    """Import the upstream Motus modules and return them.

    The upstream tree uses bare top-level package names (``models``, ``utils``,
    ``wan``). If the host process already bound those names to something else we
    evict them first, otherwise ``from utils.image_utils import ...`` resolves
    against the wrong package.
    """
    if not MOTUS_INFER_ROOT.is_dir():
        raise FileNotFoundError(
            f"Motus source tree not found at {MOTUS_INFER_ROOT}. "
            "Clone https://github.com/thu-ml/Motus into policy/motus/Motus."
        )

    _ensure_cuda_home()

    bak_root = MOTUS_INFER_ROOT / "bak"
    for p in (str(bak_root), str(MOTUS_INFER_ROOT)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    for name in ("models", "utils", "wan"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        origin = getattr(mod, "__file__", None) or ""
        if str(MOTUS_INFER_ROOT) not in str(origin):
            logger.warning(
                "Evicting conflicting top-level module %r (%s) so Motus can import its own.",
                name, origin or "<namespace>",
            )
            for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
                sys.modules.pop(key, None)

    import importlib

    motus_models = importlib.import_module("models.motus")
    image_utils = importlib.import_module("utils.image_utils")
    t5_module = importlib.import_module("wan.modules.t5")

    # Must run after the Motus modules are imported, since they are the ones
    # that bind `flash_attention` into their own namespaces.
    _install_sdpa_attention_fallback()

    return motus_models, image_utils, t5_module


# ---------------------------------------------------------------------------
# Image assembly
# ---------------------------------------------------------------------------
def build_three_view(cam_high, cam_left, cam_right) -> np.ndarray:
    """Compose the T-shaped three-view frame Motus was trained on.

    Layout (see Motus/data/lerobot/add_cam_concatenated_to_lerobot_dataset.py
    :func:`_stitch_frames` and data/utils/multi_camera_concat.py):

        +-----------------------+   top    = cam_high at native size (H x W)
        |       cam_high        |   bottom = each wrist at (H//2) x (W//2)
        +-----------+-----------+   total  = 1.5H x W
        | cam_left  | cam_right |
        +-----------+-----------+

    Because the downstream resize preserves aspect ratio, the result is
    identical for 480x640 challenge frames and for 240x320 RoboTwin frames.
    """
    import cv2

    cam_high = np.ascontiguousarray(cam_high)
    top_h, target_w = cam_high.shape[:2]
    bottom_h = top_h // 2
    split_w = target_w // 2
    right_w = target_w - split_w

    left_resized = cv2.resize(np.ascontiguousarray(cam_left), (split_w, bottom_h))
    right_resized = cv2.resize(np.ascontiguousarray(cam_right), (right_w, bottom_h))

    out = np.zeros((top_h + bottom_h, target_w, 3), dtype=np.uint8)
    out[:top_h] = cam_high[..., :3]
    out[top_h:, :split_w] = left_resized[..., :3]
    out[top_h:, split_w:] = right_resized[..., :3]
    return out


def _to_uint8_rgb(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:  # leading env dimension
        arr = arr[0]
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        peak = float(np.nanmax(arr)) if arr.size else 0.0
        if peak <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class MotusPolicy:
    """In-process Motus inference for the RoboSynChallenge evaluator."""

    def __init__(
        self,
        checkpoint_path: str,
        model_config_path: str,
        wan_path: str,
        vlm_path: str,
        device: str = "cuda",
        num_inference_steps: Optional[int] = None,
        execute_steps: Optional[int] = None,
        action_repeat: int = 1,
        action_normalization: str = "none",
        stat_path: Optional[str] = None,
        embodiment_type: str = "robosyn",
        gripper_indices: Sequence[int] = (6, 13),
        gripper_scale: float = 1.0,
        gripper_limits: Optional[Sequence[float]] = None,
        action_clip: Optional[Sequence[Sequence[float]]] = None,
        t5_mode: str = "auto",
        t5_cache_dir: Optional[str] = None,
        scene_prefix: Optional[str] = None,
        save_video_debug: bool = False,
        debug_dir: Optional[str] = None,
    ):
        import torch
        import yaml

        self.torch = torch
        self.device = device
        self.checkpoint_path = str(checkpoint_path)
        self.wan_path = str(wan_path)
        self.vlm_path = str(vlm_path)

        motus_models, image_utils, t5_module = _import_motus()
        self._Motus = motus_models.Motus
        self._MotusConfig = motus_models.MotusConfig
        self._resize_with_padding = image_utils.resize_with_padding
        self._T5EncoderModel = t5_module.T5EncoderModel

        with open(model_config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)
        common = self.config_dict["common"]
        self.video_size = (int(common["video_height"]), int(common["video_width"]))
        self.chunk_size = int(common["num_video_frames"]) * int(common["video_action_freq_ratio"])
        self.global_downsample_rate = int(common.get("global_downsample_rate", 1))

        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else self.config_dict["model"]["inference"]["num_inference_timesteps"]
        )
        self.execute_steps = int(execute_steps) if execute_steps else max(1, self.chunk_size // 2)
        self.execute_steps = min(self.execute_steps, self.chunk_size)
        self.action_repeat = max(1, int(action_repeat))

        self.action_normalization = str(action_normalization).lower()
        if self.action_normalization not in ("none", "minmax"):
            raise ValueError(f"action_normalization must be 'none' or 'minmax', got {action_normalization!r}")
        self.action_min = None
        self.action_max = None
        if self.action_normalization == "minmax":
            self._load_normalization_stats(stat_path, embodiment_type)

        self.gripper_indices = tuple(int(i) for i in gripper_indices)
        self.gripper_scale = float(gripper_scale)
        self.gripper_limits = tuple(float(v) for v in gripper_limits) if gripper_limits else None
        self.action_clip = np.asarray(action_clip, dtype=np.float32) if action_clip else None

        self.t5_mode = str(t5_mode).lower()
        self.t5_cache_dir = Path(t5_cache_dir) if t5_cache_dir else (POLICY_DIR / "cache" / "t5")
        self.scene_prefix = SCENE_PREFIX if scene_prefix is None else scene_prefix
        self.save_video_debug = bool(save_video_debug)
        self.debug_dir = Path(debug_dir) if debug_dir else (POLICY_DIR / "cache" / "debug")

        # ---- runtime state ----
        self.observation_window = None      # deque of CHW float tensors, or None before first obs
        self.current_state = None           # torch [1, 14]
        self.instruction: Optional[str] = None
        self._t5_embedding = None           # torch [S, D] on CPU
        self._t5_encoder = None
        self._vlm_processor = None
        self.episode_count = 0
        self.step_count = 0

        self.model = self._load_model()
        self._vlm_processor = self._load_vlm_processor()
        logger.info(
            "MotusPolicy ready | chunk=%d execute=%d repeat=%d steps=%d norm=%s video=%s",
            self.chunk_size, self.execute_steps, self.action_repeat,
            self.num_inference_steps, self.action_normalization, self.video_size,
        )

    # -- construction helpers ------------------------------------------------
    def _load_normalization_stats(self, stat_path, embodiment_type):
        import json

        if not stat_path:
            raise ValueError("action_normalization='minmax' requires stat_path")
        with open(stat_path, "r") as f:
            stats = json.load(f)
        if embodiment_type not in stats:
            raise KeyError(
                f"embodiment '{embodiment_type}' not in {stat_path}; available: {sorted(stats)}"
            )
        entry = stats[embodiment_type]
        self.action_min = np.asarray(entry["min"], dtype=np.float32)
        self.action_max = np.asarray(entry["max"], dtype=np.float32)

    def _resolve_checkpoint_dir(self) -> str:
        """Motus.load_checkpoint wants the directory holding mp_rank_00_model_states.pt."""
        p = Path(self.checkpoint_path)
        if p.is_file():
            return str(p)
        if (p / "mp_rank_00_model_states.pt").exists():
            return str(p)
        matches = sorted(p.glob("**/mp_rank_00_model_states.pt"))
        if matches:
            logger.info("Resolved checkpoint to %s", matches[0].parent)
            return str(matches[0].parent)
        raise FileNotFoundError(
            f"No mp_rank_00_model_states.pt under {p}. Point ckpt_path at the directory "
            "containing it (e.g. .../Motus_robotwin2 or .../checkpoint_step_XXXX/pytorch_model)."
        )

    def _build_model_config(self):
        common = self.config_dict["common"]
        model_cfg = self.config_dict["model"]
        und = model_cfg.get("und_expert", {})
        und_vlm = und.get("vlm", {})

        return self._MotusConfig(
            wan_checkpoint_path=self.wan_path,
            vae_path=os.path.join(self.wan_path, "Wan2.2_VAE.pth"),
            wan_config_path=self.wan_path,
            video_precision=model_cfg.get("wan", {}).get("precision", "bfloat16"),
            vlm_checkpoint_path=self.vlm_path,
            und_expert_hidden_size=int(und.get("hidden_size", 512)),
            und_expert_ffn_dim_multiplier=int(und.get("ffn_dim_multiplier", 4)),
            und_expert_norm_eps=float(und.get("norm_eps", 1e-5)),
            und_layers_to_extract=None,
            vlm_adapter_input_dim=int(und_vlm.get("input_dim", 2048)),
            vlm_adapter_projector_type=und_vlm.get("projector_type", "mlp3x_silu"),
            num_layers=30,
            action_state_dim=int(common["state_dim"]),
            action_dim=int(common["action_dim"]),
            action_expert_dim=int(model_cfg["action_expert"]["hidden_size"]),
            action_expert_ffn_dim_multiplier=int(model_cfg["action_expert"]["ffn_dim_multiplier"]),
            action_expert_norm_eps=1e-6,
            global_downsample_rate=int(common["global_downsample_rate"]),
            video_action_freq_ratio=int(common["video_action_freq_ratio"]),
            num_video_frames=int(common["num_video_frames"]),
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            batch_size=1,
            video_height=self.video_size[0],
            video_width=self.video_size[1],
            load_pretrained_backbones=False,   # all weights come from the checkpoint
            training_mode="finetune",
        )

    def _load_model(self):
        ckpt_dir = self._resolve_checkpoint_dir()
        logger.info("Building Motus (config only, no backbone downloads)")
        model = self._Motus(self._build_model_config()).to(self.device)
        logger.info("Loading Motus checkpoint from %s", ckpt_dir)
        model.load_checkpoint(ckpt_dir, strict=False)
        model.eval()
        return model

    def _load_vlm_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)

    # -- language ------------------------------------------------------------
    def _prompt(self, instruction: str) -> str:
        return f"{self.scene_prefix}{instruction}"

    def _t5_cache_path(self, prompt: str) -> Path:
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
        return self.t5_cache_dir / f"{digest}.pt"

    def _ensure_t5_encoder(self, device: str):
        if self._t5_encoder is not None:
            return self._t5_encoder
        ckpt = os.path.join(self.wan_path, "models_t5_umt5-xxl-enc-bf16.pth")
        tok = os.path.join(self.wan_path, "google", "umt5-xxl")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"T5 encoder weights not found: {ckpt}. Download Wan-AI/Wan2.2-TI2V-5B, or "
                "pre-encode instructions so wan_path is only needed for the VAE."
            )
        logger.info("Loading WAN umT5-xxl text encoder on %s (~9.4GB bf16)", device)
        self._t5_encoder = self._T5EncoderModel(
            text_len=512,
            dtype=self.torch.bfloat16,
            device=device,
            checkpoint_path=ckpt,
            tokenizer_path=tok,
        )
        return self._t5_encoder

    def _release_t5_encoder(self):
        if self._t5_encoder is None:
            return
        try:
            del self._t5_encoder.model
        except AttributeError:
            pass
        self._t5_encoder = None
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
        logger.info("Released T5 encoder (frees ~9.4GB VRAM for the diffusion loop)")

    def set_language(self, instruction: Optional[str]):
        """Resolve the T5 embedding for ``instruction`` (disk-cached, then freed)."""
        if not instruction:
            raise ValueError(
                "Motus requires a language instruction; env._current_instruction was empty."
            )
        if instruction == self.instruction and self._t5_embedding is not None:
            return

        self.instruction = instruction
        prompt = self._prompt(instruction)
        cache_file = self._t5_cache_path(prompt)

        if cache_file.exists():
            emb = self.torch.load(cache_file, map_location="cpu")
            if isinstance(emb, list):
                emb = emb[0]
            self._t5_embedding = emb
            logger.info("T5 embedding cache hit: %s", cache_file)
            return

        if self.t5_mode == "cache_only":
            raise FileNotFoundError(
                f"t5_mode='cache_only' but no cached embedding at {cache_file}. "
                f"Pre-encode it with encode_instructions.py for: {instruction!r}"
            )

        encode_device = "cpu" if self.t5_mode == "cpu" else self.device
        encoder = self._ensure_t5_encoder(encode_device)
        out = encoder([prompt], encode_device)
        emb = out[0] if isinstance(out, (list, tuple)) else out
        if emb.dim() == 3:
            emb = emb.squeeze(0)
        self._t5_embedding = emb.detach().to("cpu")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self._t5_embedding, cache_file)
        logger.info("Cached T5 embedding %s -> %s", tuple(self._t5_embedding.shape), cache_file)

        if self.t5_mode != "keep":
            self._release_t5_encoder()

    # -- observation ---------------------------------------------------------
    def update_observation_window(self, img_arr, state):
        """img_arr = [cam_high, cam_left_wrist, cam_right_wrist] as HWC arrays."""
        torch = self.torch
        if len(img_arr) != 3:
            raise ValueError(f"Motus needs exactly 3 views (high, left, right); got {len(img_arr)}")

        frame = build_three_view(*[_to_uint8_rgb(v) for v in img_arr])
        if frame.shape[:2] != self.video_size:
            frame = self._resize_with_padding(frame, self.video_size)
        tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)

        if self.observation_window is None:
            self.observation_window = deque(maxlen=1)
        self.observation_window.append(tensor.unsqueeze(0).to(self.device))

        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != 14:
            raise ValueError(f"Expected 14-dim qpos, got {state_arr.shape[0]}")
        state_t = torch.from_numpy(self._normalize(state_arr[None, :])).to(self.device)
        self.current_state = state_t

    # -- normalization -------------------------------------------------------
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if self.action_normalization == "none":
            return arr.astype(np.float32)
        rng = np.where(self.action_max - self.action_min == 0, 1.0, self.action_max - self.action_min)
        return ((arr - self.action_min) / rng).astype(np.float32)

    def _denormalize(self, arr: np.ndarray) -> np.ndarray:
        if self.action_normalization == "none":
            return arr.astype(np.float32)
        rng = np.where(self.action_max - self.action_min == 0, 1.0, self.action_max - self.action_min)
        return (arr * rng + self.action_min).astype(np.float32)

    # -- inference -----------------------------------------------------------
    def _build_vlm_inputs(self, frame_chw) -> Dict[str, Any]:
        from PIL import Image

        arr = (frame_chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0)
        image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": self._prompt(self.instruction)},
                {"type": "image", "image": image},
            ],
        }]
        text = self._vlm_processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        encoded = self._vlm_processor(text=[text], images=[image], return_tensors="pt")
        inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "pixel_values": encoded["pixel_values"].to(self.device),
            "image_grid_thw": encoded.get("image_grid_thw", None),
        }
        if inputs["image_grid_thw"] is not None:
            inputs["image_grid_thw"] = inputs["image_grid_thw"].to(self.device)
        return inputs

    def get_action(self) -> np.ndarray:
        """Run one joint video/action diffusion pass.

        Returns absolute joint positions, shape [execute_steps * action_repeat, 14].
        """
        torch = self.torch
        if self.observation_window is None or len(self.observation_window) == 0:
            raise RuntimeError("No observation; call update_observation_window() first.")
        if self._t5_embedding is None:
            raise RuntimeError("No instruction; call set_language() first.")

        first_frame = self.observation_window[-1]
        vlm_inputs = self._build_vlm_inputs(first_frame[0])
        lang = [self._t5_embedding.to(self.device)]

        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=first_frame,
                state=self.current_state,
                num_inference_steps=self.num_inference_steps,
                language_embeddings=lang,
                vlm_inputs=[vlm_inputs],
            )

        actions = predicted_actions.squeeze(0).float().cpu().numpy()   # [chunk, 14]
        actions = self._denormalize(actions)
        actions = self._postprocess(actions)

        if self.save_video_debug and predicted_frames is not None:
            self._dump_debug_grid(first_frame[0], predicted_frames)
        self.step_count += 1

        actions = actions[: self.execute_steps]
        if self.action_repeat > 1:
            actions = np.repeat(actions, self.action_repeat, axis=0)
        return actions

    def _postprocess(self, actions: np.ndarray) -> np.ndarray:
        """Apply embodiment fixups: gripper rescale, per-joint clipping."""
        actions = np.asarray(actions, dtype=np.float32)
        if self.gripper_scale != 1.0:
            for idx in self.gripper_indices:
                actions[:, idx] *= self.gripper_scale
        if self.gripper_limits is not None:
            lo, hi = self.gripper_limits
            for idx in self.gripper_indices:
                actions[:, idx] = np.clip(actions[:, idx], lo, hi)
        if self.action_clip is not None:
            actions = np.clip(actions, self.action_clip[0], self.action_clip[1])
        return actions

    def _dump_debug_grid(self, condition_chw, predicted_frames):
        try:
            from PIL import Image

            frames = predicted_frames
            if frames.dim() == 5:
                if frames.shape[1] == 3:          # [B, C, T, H, W]
                    frames = frames.permute(0, 2, 1, 3, 4)
                frames = frames[0]                # [T, C, H, W]

            def to_np(t):
                return (t.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            tiles = [to_np(condition_chw)] + [to_np(frames[i]) for i in range(min(4, frames.shape[0]))]
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            out = self.debug_dir / f"ep{self.episode_count:04d}_step{self.step_count:04d}.png"
            Image.fromarray(np.concatenate(tiles, axis=1)).save(out)
        except Exception as exc:  # debug output must never break evaluation
            logger.warning("Failed to write debug grid: %s", exc)

    # -- episode lifecycle ---------------------------------------------------
    def reset_obsrvationwindows(self):
        """Note: spelling matches the challenge's other adapters (pi05, xr1)."""
        self.observation_window = None
        self.current_state = None
        self.episode_count += 1
        self.step_count = 0
        # instruction / T5 embedding survive: every episode of a run shares one task.
        logger.info("Motus reset (episode %d)", self.episode_count)
