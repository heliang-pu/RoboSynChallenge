#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os

class PI0:

    def __init__(
        self,
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        pytorch_device="cuda",
        max_guidance_weight=10.0,
        rtc_correction="vjp",
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.max_guidance_weight = float(max_guidance_weight)
        self.rtc_correction = rtc_correction
        # `deploy_policy.eval` reads this for its CUDA-synchronized timing.
        self.pytorch_device = pytorch_device

        specified_path = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        config = _config.get_config(self.train_config_name)
        self.action_horizon = int(config.model.action_horizon)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            pytorch_device=pytorch_device,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"\nsuccessfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        self.observation_window = {
            "observation/image": img_front,
            "observation/left_wrist_image": img_left,
            "observation/right_wrist_image": img_right,
            "observation/state": state,
            "prompt": self.instruction,
        }

    def get_action(self, guidance=None):
        """Sample a chunk; returns (env-space actions, model-space actions).

        The model-space copy is what Real-Time Chunking must feed back as its
        guidance target next time -- the env-space actions have already been
        un-normalized by the output transform.
        """
        assert self.observation_window is not None, "update observation_window first!"
        kwargs = {"return_raw_actions": True}
        if guidance is not None:
            kwargs.update(
                prev_chunk=guidance["prev_chunk"],
                prefix_weights=guidance["prefix_weights"],
                max_guidance_weight=self.max_guidance_weight,
                rtc_correction=self.rtc_correction,
            )
        outputs = self.policy.infer(self.observation_window, **kwargs)
        return outputs["actions"], outputs["raw_actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
