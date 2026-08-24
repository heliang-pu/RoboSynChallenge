# ----------------------------------------------------------------------------
# OpenDM DM0.5 Policy Adapter for RoboSynChallenge
#
# 遵循 RoboTwin 统一评估接口:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info
#   - reset_model(model) -> None
#
# 模型以 HTTP 服务方式运行(launch/run_dm05_server.sh),本适配器只做
# 观测编码 + HTTP 调用 + 动作回放,不引入 opendm 的训练/推理依赖。
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.insert(0, parent_directory)

from dm_model import DM05


def _format_env_action(action, env):
    """Convert DM05 output into the torch action format EmbodiChain accepts."""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.shape[0] < env_action_dim:
        raise ValueError(
            f"Policy action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
        )
    action_array = action_array[:env_action_dim]

    action_tensor = torch.as_tensor(
        action_array, dtype=torch.float32, device=env.unwrapped.device
    )
    return action_tensor.unsqueeze(0)


def encode_obs(obs):
    """Convert Gymnasium Dict observation to DM05 input format.

    EmbodiChain observation keys:
        "sensor/cam_high/color"        -> Head
        "sensor/cam_left_wrist/color"  -> Left wrist
        "sensor/cam_right_wrist/color" -> Right wrist
        "robot/qpos"                   -> joint state

    Returns:
        img_arr:  [head, left_wrist, right_wrist] (H, W, C) RGB numpy arrays,
                  与推理服务 --inference-config.image-prompts
                  "Head" "Left wrist" "Right wrist" 一一对应
        state:    joint state vector
    """
    img_head = obs["sensor"]["cam_high"]["color"][0, ..., :3]
    img_left = obs["sensor"]["cam_left_wrist"]["color"][0, ..., :3]
    img_right = obs["sensor"]["cam_right_wrist"]["color"][0, ..., :3]

    state = obs["robot"]["qpos"][0]
    if hasattr(state, "cpu"):
        state = state.cpu().numpy()
    img_arr = [np.asarray(img_head), np.asarray(img_left), np.asarray(img_right)]

    return img_arr, state


def get_model(usr_args):
    """Create the DM05 HTTP-client policy.

    usr_args 中可用字段(均来自 deploy_policy.yml,亦可被命令行覆盖):
        server_url    — DM05 推理服务地址 (default http://127.0.0.1:7891)
        dm_step       — 每次推理后执行的动作步数 (default 30)
        robot_type    — norm_stats 的机器人 profile(如 "Aloha" / "DOS W1"),
                        自训 SFT checkpoint 通常留空
        control_mode  — 文本条件字段,基座 DM05 需显式给出,SFT 通常留空
        speed         — 文本条件字段,同上
        state_indices — 可选的 qpos 维度选择/重排列表,用于对齐 norm_stats 维度
        infer_timeout — 单次推理 HTTP 超时秒数
        seed          — 可选的确定性采样种子
    """
    state_indices = usr_args.get("state_indices")

    model = DM05(
        server_url=usr_args.get("server_url", "http://127.0.0.1:7891"),
        dm_step=int(usr_args.get("dm_step", 30)),
        robot_type=usr_args.get("robot_type"),
        control_mode=usr_args.get("control_mode"),
        speed=usr_args.get("speed"),
        state_indices=state_indices,
        timeout=float(usr_args.get("infer_timeout", 60.0)),
        seed=usr_args.get("seed_sampling"),
    )
    return model


def eval(env, model, obs):
    """Run one inference cycle and execute actions in the environment.

    1. Sets the language instruction (on first call when observation_window is None)
    2. Encodes observation and posts it to the DM05 service
    3. Executes up to model.dm_step actions from the returned chunk
    """
    if model.observation_window is None:
        instruction = getattr(env, "_current_instruction", None)
        model.set_language(instruction)

    img_arr, state = encode_obs(obs)
    model.update_observation_window(img_arr, state)

    actions = model.get_action()[: model.dm_step]

    final_obs = obs
    info = {}
    truncated = False
    for action in actions:
        action_tensor = _format_env_action(action, env)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        # The `gym_config` setting configures the `actionmanager` to support delta action input;
        # the default action must be absolute `qpos`.
        _trunc = truncated.any() if hasattr(truncated, "any") else truncated
        if bool(_trunc):
            break
        img_arr, state = encode_obs(final_obs)
        model.update_observation_window(img_arr, state)

    return final_obs, info, truncated


def reset_model(model):
    """Reset DM05 client state (observation window and instruction)."""
    model.reset_obsrvationwindows()
