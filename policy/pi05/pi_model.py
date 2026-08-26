#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import etils.epath as epath
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
from openpi import transforms as _transforms
from openpi.training import checkpoints as _checkpoints
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

        # Real-Time Chunking needs its guidance target in the *model's* action
        # space, but the policy hands back environment-space actions: for this
        # checkpoint the output chain is Unnormalize -> AbsoluteActions, i.e. the
        # model predicts deltas against whatever state that inference saw.
        # Feeding a previous chunk back verbatim would therefore anchor the new
        # chunk to a stale base state.  Rebuild the training-time input path for
        # actions so the previous plan can be re-expressed against the *current*
        # state before it is used as a target.
        checkpoint_dir = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"
        data_config = config.data.create(config.assets_dirs, config.model)
        norm_stats = _checkpoints.load_norm_stats(
            epath.Path(checkpoint_dir) / "assets", data_config.asset_id
        )
        self._delta_actions = next(
            (t for t in data_config.data_transforms.inputs if isinstance(t, _transforms.DeltaActions)),
            None,
        )
        self._normalize = _transforms.Normalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        )
        self._pad_actions = _transforms.PadStatesAndActions(config.model.action_dim)

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

    def to_model_action_space(self, actions_env):
        """Re-express absolute environment actions in the model's action space.

        Mirrors the transform chain training applies to action targets --
        `DeltaActions` against the *current* state, `Normalize`, then padding to
        `action_dim` -- which is exactly the inverse of the policy's output
        chain.  The state comes from the live observation window, so a chunk
        planned several steps ago is rebased onto where the robot is now.
        """
        data = {
            "state": np.asarray(self.observation_window["observation/state"], dtype=np.float32),
            "actions": np.asarray(actions_env, dtype=np.float32),
        }
        if self._delta_actions is not None:
            data = self._delta_actions(data)
        data = self._normalize(data)
        data = self._pad_actions(data)
        return data["actions"]

    def get_action(self, guidance=None):
        """Sample one action chunk in environment (absolute) action space.

        `guidance` carries the previous plan as absolute environment actions
        aligned to the new chunk's timeline; it is rebased here.
        """
        assert self.observation_window is not None, "update observation_window first!"
        kwargs = {}
        if guidance is not None:
            kwargs.update(
                prev_chunk=self.to_model_action_space(guidance["prev_actions_env"]),
                prefix_weights=guidance["prefix_weights"],
                max_guidance_weight=self.max_guidance_weight,
                rtc_correction=self.rtc_correction,
            )
        return self.policy.infer(self.observation_window, **kwargs)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
