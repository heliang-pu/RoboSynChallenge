# ----------------------------------------------------------------------------
# G0.5 (Galaxea GalaxeaVLA) Policy Adapter for RoboSynChallenge
#
# 遵循 RoboSynChallenge 统一评估接口:
#   - get_model(usr_args)    -> model
#   - eval(env, model, obs)  -> (obs, info, truncated)
#   - reset_model(model)     -> None
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.insert(0, parent_directory)

from g05_model import CAMERA_KEYS, G05


def _format_env_action(action, env):
    """把 G0.5 输出的 14 维绝对关节位置转成 env.step 接受的 (1, 14) torch tensor。"""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.shape[0] < env_action_dim:
        raise ValueError(
            f"Policy action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
        )
    action_array = action_array[:env_action_dim]

    # 夹到动作空间范围内。G0.5 是流匹配连续输出，反归一化后偶尔会越界一点点，
    # 越界值喂给仿真会把关节打飞，这里按 env 声明的上下界截断（无界时自动跳过）。
    space = env.unwrapped.single_action_space
    low, high = np.asarray(space.low, dtype=np.float32).reshape(-1), np.asarray(
        space.high, dtype=np.float32
    ).reshape(-1)
    if low.shape == action_array.shape and np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
        action_array = np.clip(action_array, low, high)

    action_tensor = torch.as_tensor(
        action_array, dtype=torch.float32, device=env.unwrapped.device
    )
    return action_tensor.unsqueeze(0)


def encode_obs(obs):
    """Gymnasium Dict 观测 -> G0.5 输入格式。

    EmbodiChain observation keys:
        "sensor/cam_high/color"        -> 头部相机
        "sensor/cam_left_wrist/color"  -> 左腕相机
        "sensor/cam_right_wrist/color" -> 右腕相机
        "robot/qpos"                   -> 14 维关节状态

    相机 key 与 G0.5 robotwin embodiment 的 shape_meta 完全同名，直接透传。
    返回:
        img_dict: {cam_high/cam_left_wrist/cam_right_wrist: (H, W, 3) uint8}
        state:    (14,) 关节状态
    """
    img_dict = {}
    for key in CAMERA_KEYS:
        # (1, H, W, 4) RGBA -> (H, W, 3) RGB
        img_dict[key] = obs["sensor"][key]["color"][0, ..., :3]

    # (num_envs, num_joints) -> (num_joints,)
    state = obs["robot"]["qpos"][0]

    return img_dict, state


def get_model(usr_args):
    """构造 G0.5 策略。

    usr_args（来自 deploy_policy.yml，可被 eval.sh 的 --overrides 覆盖）:
        ckpt_path            — model_state_dict.pt 路径（必填）
        dataset_stats_path   — dataset_stats.json，留空则从 ckpt 上级目录自动找
        action_horizon       — 单次推理产出的动作 chunk 长度，留空用模型自带 horizon
        replan_steps         — 每次推理实际执行多少步后重新推理
        num_inference_steps  — 流匹配去噪步数，留空用 checkpoint 配置
        control_frequency    — 喂给 action tokenizer 的控制频率
        mixed_precision      — no / fp16 / bf16
        pytorch_device       — cuda / cpu
        seed                 — 随机种子
    """
    ckpt_path = usr_args.get("ckpt_path")
    if ckpt_path is None:
        raise ValueError("ckpt_path must be provided in usr_args (deploy_policy.yml)")

    def _opt_int(key):
        value = usr_args.get(key)
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
            return None
        return int(value)

    model = G05(
        ckpt_path=ckpt_path,
        dataset_stats_path=usr_args.get("dataset_stats_path"),
        device=usr_args.get("pytorch_device", "cuda"),
        mixed_precision=usr_args.get("mixed_precision", "bf16"),
        action_horizon=_opt_int("action_horizon"),
        replan_steps=int(usr_args.get("replan_steps", 16)),
        num_inference_steps=_opt_int("num_inference_steps"),
        control_frequency=float(usr_args.get("control_frequency", 25.0)),
        sim_cfg_name=usr_args.get("sim_cfg_name", "sim_robotwin"),
        sim_task=usr_args.get("sim_task", "robotwin"),
        embodiment=usr_args.get("embodiment", "robotwin"),
        seed=_opt_int("seed"),
    )
    return model


def eval(env, model, obs):
    """跑一次推理并把动作 chunk 逐步执行到环境里。

    1. 首次调用时从 env 上取语言指令
    2. 编码观测、更新观测窗
    3. model.get_action() 拿 [T, 14] 动作 chunk
    4. 逐步 env.step，中途 truncated 就提前退出
    """
    # 首次调用设置语言指令
    if model.observation_window is None:
        instruction = getattr(env, "_current_instruction", None)
        model.set_language(instruction)

    # 编码观测并更新观测窗
    img_dict, state = encode_obs(obs)
    model.update_observation_window(img_dict, state)

    # G0.5 一次推理产出一个动作 chunk
    actions = model.get_action()

    # 逐步执行。G0.5 的 chunk 是开环执行的：整段执行完再重新推理，
    # 所以中途不需要刷新观测窗（下次进 eval 时会用新 obs 重新填）。
    final_obs = obs
    info = {}
    truncated = False
    for action in actions:
        action_tensor = _format_env_action(action, env)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        # gym_config 里 actionmanager 可能配成 delta action，默认必须是绝对 qpos
        _trunc = truncated.any() if hasattr(truncated, "any") else truncated
        if bool(_trunc):
            break

    return final_obs, info, truncated


def reset_model(model):
    """清空 G0.5 的观测窗和语言指令（episode 之间调用）。"""
    model.reset_obsrvationwindows()
