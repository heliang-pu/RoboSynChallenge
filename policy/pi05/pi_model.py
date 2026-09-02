"""Pi0.5 model wrapper with optional realtime-vla inference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI0:
    def __init__(
        self,
        train_config_name: str,
        model_name: str,
        checkpoint_id: int,
        pi0_step: int,
        pytorch_device: str = "cuda",
        inference_backend: str = "jax",
        checkpoint_root: str | None = None,
        converted_checkpoint: str | None = None,
        realtime_vla_dir: str | None = None,
        tokenizer_path: str | None = None,
        prompt_for_allocation: str = "click the bell",
    ) -> None:
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.inference_backend = inference_backend
        self.pytorch_device = pytorch_device

        root = Path(checkpoint_root or "policy/pi05/checkpoints")
        checkpoint_path = root / train_config_name / model_name / str(checkpoint_id)
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        if inference_backend == "jax":
            config = _config.get_config(train_config_name)
            self.policy = _policy_config.create_trained_policy(
                config,
                checkpoint_path,
                pytorch_device=pytorch_device,
            )
        elif inference_backend == "realtime_vla":
            from policy.pi05.realtime_vla.accelerated_policy import RealtimeVlaPi05Policy

            if converted_checkpoint is None:
                raise ValueError("converted_checkpoint is required for inference_backend=realtime_vla")
            norm_stats_candidates = list((checkpoint_path / "assets").glob("*/*/norm_stats.json"))
            if len(norm_stats_candidates) != 1:
                raise FileNotFoundError(
                    f"Expected one norm_stats.json below {checkpoint_path / 'assets'}, found {len(norm_stats_candidates)}"
                )
            self.policy = RealtimeVlaPi05Policy(
                converted_checkpoint=converted_checkpoint,
                norm_stats_path=norm_stats_candidates[0],
                tokenizer_path=tokenizer_path or str(Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"),
                realtime_vla_dir=realtime_vla_dir or os.environ.get("REALTIME_VLA_DIR", "../realtime-vla"),
                prompt_for_allocation=prompt_for_allocation,
            )
        else:
            raise ValueError(f"Unsupported inference_backend: {inference_backend}")

        print(f"loading model success! backend={inference_backend}")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.instruction = None

    def set_img_size(self, img_size) -> None:
        self.img_size = img_size

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction
        print(f"\nsuccessfully set instruction:{instruction}")

    def update_observation_window(self, img_arr, state) -> None:
        img_front, img_right, img_left = img_arr
        self.observation_window = {
            "observation/image": np.asarray(img_front),
            "observation/left_wrist_image": np.asarray(img_left),
            "observation/right_wrist_image": np.asarray(img_right),
            "observation/state": np.asarray(state),
            "prompt": self.instruction,
        }

    def get_action(self):
        if self.observation_window is None:
            raise RuntimeError("update_observation_window must be called before get_action")
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self) -> None:
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
