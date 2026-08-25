# ----------------------------------------------------------------------------
# RoboSynChallenge <-> RLinf 接入层
#
# RLinf 自带的 EmbodiChain 适配器(rlinf/envs/embodichain/embodichain_env.py)只做到
# CartPole:它的 _wrap_obs 只把 robot 的 qpos/qvel/qf 拼成 {"states": tensor},没有图像
# 通路,也没有语言指令。VLA(pi0.5)需要多路相机 + prompt,所以这里继承它并补齐。
#
# 成败判定一律用官方 is_task_success(),不自己发明规则、不做 reward shaping。
#
# 两个挂钩点:
#   env 类      通过 scripts/patch_rlinf_env.py 注册成 env_type: robosynchallenge
#   openpi 配置 register_dataconfig() 塞进 RLinf 的 _CONFIGS_DICT,
#               供 yaml 的 actor.model.openpi.config_name 引用
# ----------------------------------------------------------------------------

from .dataconfig import CONFIG_NAME, LeRobotRoboSynChallengeDataConfig
from .dataconfig import register as register_dataconfig
from .vla_env import RoboSynChallengeVLAEnv, install_official_reward

__all__ = [
    "RoboSynChallengeVLAEnv",
    "install_official_reward",
    "LeRobotRoboSynChallengeDataConfig",
    "register_dataconfig",
    "CONFIG_NAME",
]
