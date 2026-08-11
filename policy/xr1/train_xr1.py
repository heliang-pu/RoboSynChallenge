#!/usr/bin/env python
# ----------------------------------------------------------------------------
# XR-1 后训练入口（本适配层自有，不改上游 tools/train.py）
#
# 只做一件上游没有的事：注册一个「冻结 VLM」的 runner。
#
# 为什么必须冻结（单卡 4090 的硬约束，实测数字）:
#   模型 5.50B = VLM 4.83B + DiT/projector 0.68B
#   全参微调  : bf16 权重 11G + 梯度 11G + Adam(fp32 m/v/master) 66G ≈ 88G
#               -> GPU 只剩 37G 放不下；CPU RAM 也只有 44G 可用，offload 同样放不下
#   冻结 VLM  : 权重仍 11G，但可训参数只有 0.68B -> Adam 仅 8.1G
#               -> 合计约 25G，装得下
#
# 语义上这也是 VLA 后训练的常规做法：VLM 当冻结的视觉-语言特征提取器，
# 动作专家(DiT)和投影层去适配新本体。上游默认全参是因为他们用多机多卡。
#
# 用法与上游一致，全部 hydra 覆盖照常传：
#   python train_xr1.py model.type=FrozenVLMRunner data=... model=posttrain ...
# ----------------------------------------------------------------------------

import os
import sys

# 让 `import tools.train` / `mibot` 都能找到（本文件在 policy/xr1 下，
# 上游包在 policy/xr1/Xiaomi-Robotics-1/xr1）
_HERE = os.path.dirname(os.path.abspath(__file__))
_XR1_PKG = os.path.join(_HERE, "Xiaomi-Robotics-1", "xr1")
for path in (_XR1_PKG, _HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from transformers.utils import logging

from mibot.models import MIMODEL
from mibot.models.runner.base_runner import BaseRunner

logger = logging.get_logger(__name__)


@MIMODEL.register_module()
class FrozenVLMRunner(BaseRunner):
    """冻结 VLM 主干，只训练 DiT + 各投影层。"""

    def configure_model(self) -> None:
        super().configure_model()

        frozen = 0
        trainable = 0
        for name, parameter in self.model.named_parameters():
            if name.startswith("vlm."):
                parameter.requires_grad_(False)
                frozen += parameter.numel()
            else:
                trainable += parameter.numel()

        # VLM 整体不回传参数梯度，再对它的 MLP 做梯度检查点纯属浪费
        self.model.ffn_gradient_checkpointing = False

        logger.info(
            f"FrozenVLMRunner: 冻结 {frozen / 1e9:.2f}B (VLM), "
            f"可训练 {trainable / 1e9:.2f}B (DiT + projector)"
        )
        print(
            f"[FrozenVLMRunner] 冻结 {frozen / 1e9:.2f}B / 可训练 {trainable / 1e9:.2f}B",
            flush=True,
        )
        if trainable == 0:
            raise RuntimeError("没有可训练参数，检查参数名前缀是否仍是 'vlm.'")

    def configure_optimizers(self):
        """在上游实现之外补一道断言：确认冻结的 VLM 参数确实没进优化器。

        BaseRunner.build_optimizer 是按 p.requires_grad 过滤的，理论上不会进；
        但这条链路一旦悄悄失效（比如上游改了过滤逻辑），表现是显存暴涨然后 OOM，
        很难一眼看出原因，所以显式验一次。
        """
        result = super().configure_optimizers()

        optimizer = result["optimizer"] if isinstance(result, dict) else result
        in_optimizer = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        frozen_leaked = [
            name
            for name, parameter in self.model.named_parameters()
            if name.startswith("vlm.") and id(parameter) in in_optimizer
        ]
        if frozen_leaked:
            raise RuntimeError(
                f"冻结的 VLM 参数混进了优化器（{len(frozen_leaked)} 个，例: {frozen_leaked[:3]}）"
            )

        optimizer_params = sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        )
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"[FrozenVLMRunner] 优化器纳管 {optimizer_params / 1e9:.3f}B 参数，"
            f"requires_grad 参数 {trainable_params / 1e9:.3f}B，VLM 无泄漏",
            flush=True,
        )
        return result


def main():
    # 复用上游的 hydra 入口；import 本模块已经把 FrozenVLMRunner 注册进 MIMODEL
    from tools.train import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
