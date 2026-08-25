# ----------------------------------------------------------------------------
# 从 policy/pi05/src/openpi/policies/libero_policy.py 逐字 vendor 过来的
# EmbodiChainInputs / EmbodiChainOutputs / _parse_image。
#
# 为什么要复制而不是 import:这两个类是本仓库 openpi fork 的自定义扩展,只存在于
# policy/pi05/src/openpi;RLinf venv 装的是 rlinf-openpi(RLinf 自己的 openpi 发行版,
# RL 采样机制依赖它),里面没有这两个类。跨包 import 会把两个不同版本的 openpi 拽进
# 同一进程——本会话已经在包遮蔽上摔过一次,不再走那条路。
#
# 语义契约:这里的代码必须和 SFT 用的版本逐字一致,否则 RL 推理与 SFT 的输入处理
# 出现分叉,动作分布悄悄漂移。升级 policy/pi05 的 openpi fork 时,请 diff 一下
# 源文件对应段落并同步这里。来源版本:ppo-post-training 分支合并 84b6c0e 之后。
#
# 唯一的改动:import 路径(openpi.transforms / openpi.models.model 是 openpi 的
# 核心公共 API,rlinf-openpi 同样提供)。
# ----------------------------------------------------------------------------

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

__all__ = ["EmbodiChainInputs", "EmbodiChainOutputs"]


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class EmbodiChainInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])
        # Create inputs dict. Do not change the keys in the dict below.
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
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class EmbodiChainOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        return {"actions": np.asarray(data["actions"][:, :14])}
