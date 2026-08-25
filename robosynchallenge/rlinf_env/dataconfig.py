# ----------------------------------------------------------------------------
# RoboSynChallenge 在 RLinf 侧的 openpi DataConfig。
#
# 为什么不能直接复用 RLinf 自带的 ``pi05_aloha_robotwin``:
# 那份用的是 ``aloha_policy.AlohaInputs(adapt_to_pi=True)``,会做 ALOHA 特有的关节
# 翻转/重对齐。RoboSynChallenge 的 SFT checkpoint 是用
# ``openpi.policies.libero_policy.EmbodiChainInputs`` 训的,不做这个变换。两者混用
# 不会报错,只会让动作空间悄悄错位,策略行为和 SFT 时不一致——属于最难查的那类问题。
#
# 除此之外两者是一样的:同样三路相机(cam_high / cam_left_wrist / cam_right_wrist)、
# 同样 14 维双臂、同样的 delta mask(每臂 6 关节做 delta、夹爪保持绝对,
# 即 ``make_bool_mask(6, -1, 6, -1)``)。
#
# ``register()`` 把配置塞进 RLinf 的 ``_CONFIGS_DICT``,之后 yaml 里
# ``actor.model.openpi.config_name`` 就能引用它。
# ----------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.policies import libero_policy
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

__all__ = ["LeRobotRoboSynChallengeDataConfig", "register", "CONFIG_NAME"]

# yaml 里 actor.model.openpi.config_name 要写的名字
CONFIG_NAME = "pi05_robosynchallenge"


@dataclasses.dataclass(frozen=True)
class LeRobotRoboSynChallengeDataConfig(DataConfigFactory):
    """镜像 policy/pi05 里 SFT 用的 LeRobotEmbodiChainDataConfig。

    这里刻意不带 sim-RECAP 的 ``acp_indicator_key`` —— 那是优势条件化 SFT 的东西,
    PPO/GRPO 走的是真正的策略梯度,不需要把 advantage 拼进 prompt。
    """

    # SFT 时 pi05_base_robosynchallenge_full 用的是 True(绝对动作转 delta)。
    # 改这个值会让 RL 的动作空间和 checkpoint 对不上。
    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # 推理路径上环境直接给这些键(见 robosynchallenge/rlinf_env/vla_env.py 与
        # policy/pi05/deploy_policy.py 的 encode_obs),训练数据集里则是
        # observation.images.* —— 两边在这里对齐。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.cam_high",
                        "observation/left_wrist_image": "observation.images.cam_left_wrist",
                        "observation/right_wrist_image": "observation.images.cam_right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.EmbodiChainInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.EmbodiChainOutputs()],
        )

        if self.extra_delta_transform:
            # 每臂:6 个关节做 delta,夹爪保持绝对。双臂共 14 维。
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def register(
    repo_id: str = "RoboSynChallenge/cobotmagic_Sim_mixer_operating",
    assets_dir: str | None = None,
    name: str = CONFIG_NAME,
) -> str:
    """把配置注册进 RLinf 的 openpi 配置表。幂等。

    Args:
        repo_id: 决定 norm stats 在 ``<checkpoint>/assets/<repo_id>/`` 下的查找路径。
            换任务时要跟着换,否则会加载到别的任务的归一化统计量。
        assets_dir: 显式指定 assets 目录;为 None 时用 checkpoint 自带的。
        name: 注册名,yaml 的 ``openpi.config_name`` 要与之一致。

    Returns:
        实际注册的名字。
    """
    from openpi.models import pi0_config
    from openpi.training.config import AssetsConfig, TrainConfig

    from rlinf.models.embodiment.openpi import dataconfig as _rlinf_dataconfig

    if name in _rlinf_dataconfig._CONFIGS_DICT:
        return name

    data_kwargs = dict(
        repo_id=repo_id,
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    )
    if assets_dir is not None:
        data_kwargs["assets"] = AssetsConfig(assets_dir=assets_dir)

    _rlinf_dataconfig._CONFIGS_DICT[name] = TrainConfig(
        name=name,
        # 与 policy/pi05 的 pi05_base_robosynchallenge_full 保持一致:
        # pi05=True, action_horizon=50。discrete_state_input 是 Pi0Config 对 pi05 的默认值。
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotRoboSynChallengeDataConfig(**data_kwargs),
        num_train_steps=20_000,
    )
    return name
