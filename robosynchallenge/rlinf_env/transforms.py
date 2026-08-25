# ----------------------------------------------------------------------------
# RoboSynChallenge 的 openpi 输入/输出变换 —— 从 policy/pi05/src/openpi/policies/
# libero_policy.py 逐字 vendor 过来(EmbodiChainInputs / EmbodiChainOutputs / _parse_image)。
#
# 为什么要复制而不是 import:这三样是 RoboSynChallenge 自己加进 openpi fork 里的,
# 只存在于 policy/pi05/src/openpi。RLinf 的 venv 装的是 PyPI 上的 rlinf-openpi
# (RLinf 自己的 openpi 构建,RL 模型代码依赖它),它的 libero_policy 只有
# LiberoInputs / LiberoOutputs。两个 openpi 塞进一个进程只会重演包遮蔽的噩梦,
# 所以把 SFT 用的变换原样搬到本仓库。语义逐字一致是硬要求 —— 任何偏差都会让
# RL 策略看到的输入和 SFT 训练时不同,而且不报错。
#
# 与上游 fork 的同步:若 policy/pi05 那份改了,这里必须跟着改。
# ----------------------------------------------------------------------------

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

__all__ = ["EmbodiChainInputs", "EmbodiChainOutputs", "ROBOT_ACTION_DIM"]

# CobotMagic 双臂:每臂 6 关节 + 1 夹爪。模型内部 action_dim 是 32(padding),
# 输出时只取前 14 维。与 SFT 侧 EmbodiChainOutputs 的 `[:, :14]` 一致。
ROBOT_ACTION_DIM = 14


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class EmbodiChainInputs(transforms.DataTransformFn):
    """三路相机(主视角 + 左右腕)+ 14 维状态 -> pi0.5 的输入字典。训练与推理共用。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class EmbodiChainOutputs(transforms.DataTransformFn):
    """模型输出(padding 到 32 维)-> 机器人的 14 维动作。仅推理用。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ROBOT_ACTION_DIM])}
