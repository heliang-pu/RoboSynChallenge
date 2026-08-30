"""OpenPI-compatible preprocessing around realtime-vla's Pi0.5 kernels."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

from policy.pi05.realtime_vla.tokenizer_adapter import SentencePieceAutoTokenizer


def _resize_with_pad(image: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    current_width, current_height = pil_image.size
    if (current_width, current_height) == (width, height):
        return np.asarray(pil_image)
    ratio = max(current_width / width, current_height / height)
    resized_width = int(current_width / ratio)
    resized_height = int(current_height / ratio)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    canvas.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return np.asarray(canvas)


class RealtimeVlaPi05Policy:
    """Expose the click-bell model with the same observation/action contract as PI0."""

    def __init__(
        self,
        *,
        converted_checkpoint: str | Path,
        norm_stats_path: str | Path,
        tokenizer_path: str | Path,
        realtime_vla_dir: str | Path,
        prompt_for_allocation: str = "click the bell",
        state_dim: int = 14,
        action_dim: int = 14,
        num_views: int = 3,
        chunk_size: int = 50,
    ) -> None:
        realtime_vla_dir = Path(realtime_vla_dir).expanduser().resolve()
        sys.path.insert(0, str(realtime_vla_dir))
        import pi05_infer  # noqa: PLC0415

        pi05_infer.AutoTokenizer = SentencePieceAutoTokenizer
        with Path(converted_checkpoint).open("rb") as stream:
            checkpoint = pickle.load(stream)
        with Path(norm_stats_path).open() as stream:
            stats_file = json.load(stream)
        self._stats = stats_file.get("norm_stats", stats_file)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._delta_mask = np.asarray([True] * 6 + [False] + [True] * 6 + [False])[:action_dim]
        self._rng = np.random.default_rng(0)
        self._infer = pi05_infer.Pi05Inference(
            checkpoint=checkpoint,
            num_views=num_views,
            chunk_size=chunk_size,
            tokenizer_path=str(Path(tokenizer_path).expanduser().resolve()),
            discrete_state_input=True,
            max_prompt_text=prompt_for_allocation,
            state_dim_for_max_prompt=state_dim,
        )

    @staticmethod
    def _quantile_normalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        q01 = np.asarray(stats["q01"], dtype=np.float32)[: values.shape[-1]]
        q99 = np.asarray(stats["q99"], dtype=np.float32)[: values.shape[-1]]
        return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    @staticmethod
    def _quantile_unnormalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        dimension = len(stats["q01"])
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        output = values.copy()
        output[..., :dimension] = (values[..., :dimension] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return output

    @staticmethod
    def _prepare_image(image: np.ndarray) -> torch.Tensor:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        image = _resize_with_pad(image)
        normalized = image.astype(np.float32) / 255.0 * 2.0 - 1.0
        return torch.from_numpy(normalized)

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, np.ndarray]:
        state = np.asarray(observation["observation/state"], dtype=np.float32)[: self._state_dim]
        normalized_state = self._quantile_normalize(state, self._stats["state"])
        bins = np.linspace(-1, 1, 257)[:-1]
        state_tokens = np.digitize(normalized_state, bins=bins) - 1
        images = torch.stack(
            [
                self._prepare_image(observation["observation/image"]),
                self._prepare_image(observation["observation/left_wrist_image"]),
                self._prepare_image(observation["observation/right_wrist_image"]),
            ]
        ).to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
        if noise is None:
            noise = self._rng.standard_normal((self._chunk_size, 32), dtype=np.float32)
        noise_tensor = torch.as_tensor(noise, device="cuda", dtype=torch.bfloat16)
        normalized_actions = self._infer.forward(
            images,
            noise_tensor,
            str(observation["prompt"]),
            state_tokens,
        ).float().cpu().numpy()
        actions = self._quantile_unnormalize(normalized_actions, self._stats["actions"])[..., : self._action_dim]
        actions[..., self._delta_mask] += state[self._delta_mask]
        return {"actions": actions}

