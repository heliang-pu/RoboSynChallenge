# ----------------------------------------------------------------------------
# XR-1 (Xiaomi-Robotics-1) Policy Adapter for RoboSynChallenge
#
# 遵循 RoboTwin 统一评估接口:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info, truncated
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import glob
import os
import sys

import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.insert(0, parent_directory)

from xr1_model import XR1

DEFAULT_XR1_REPO = os.path.join(parent_directory, "Xiaomi-Robotics-1", "xr1")


def _format_env_action(action, env):
    """Convert XR-1 output into the torch action format EmbodiChain accepts."""
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
    """Convert gym Gymnasium Dict observation to XR-1 input format.

    EmbodiChain observation keys:
        "sensor/cam_high/color"        -> ego view
        "sensor/cam_left_wrist/color"  -> left wrist
        "sensor/cam_right_wrist/color" -> right wrist
        "robot/qpos"                   -> 14-dim joint state

    Returns:
        img_arr: [cam_high, cam_right_wrist, cam_left_wrist] as (H, W, 3) uint8
        state:   (14,) absolute joint position
    """
    img_front_raw = obs["sensor"]["cam_high"]["color"]
    img_left_raw = obs["sensor"]["cam_left_wrist"]["color"]
    img_right_raw = obs["sensor"]["cam_right_wrist"]["color"]

    img_front = img_front_raw[0, ..., :3]
    img_left = img_left_raw[0, ..., :3]
    img_right = img_right_raw[0, ..., :3]

    state = obs["robot"]["qpos"][0]
    img_arr = [img_front, img_right, img_left]

    return img_arr, state


def _relocate_models_path(path, label):
    """跨机器重定位模型路径。

    同一份 deploy_policy.yml 要在 4090 和 pro6000 上都能用，但两台机器的模型根
    不同（/home/phl/workspace/models vs /home/fmc3-8/workspace/models）。
    配置里的路径不存在时，把 "models/" 之后的部分接到候选根上重试，
    而不是让用户改 yml（改了另一台就坏）。
    """
    path = os.path.expanduser(str(path))
    if os.path.exists(path):
        return path

    marker = os.sep + "models" + os.sep
    if marker not in path:
        return path
    suffix = path.split(marker, 1)[1]

    candidates = []
    override = os.environ.get("XR1_MODELS_ROOT")
    if override:
        candidates.append(override)
    candidates += sorted(glob.glob("/home/*/workspace/models"))
    candidates.append(os.path.join(os.path.expanduser("~"), "workspace", "models"))

    for root in candidates:
        candidate = os.path.join(root, suffix)
        if os.path.exists(candidate):
            print(f"[XR1] {label} 原路径不存在，已重定位到本机: {candidate}")
            return candidate
    return path


def _resolve_model_path(usr_args, repo_root):
    """确定权重位置，优先级: model_path > checkpoints/<train_config>/<model_name> > pretrained_ckpt"""
    explicit = usr_args.get("model_path")
    if explicit:
        return _relocate_models_path(explicit, "model_path")

    train_config_name = usr_args.get("train_config_name")
    model_name = usr_args.get("model_name")
    if train_config_name and model_name:
        candidate = os.path.join(
            repo_root, "policy", "xr1", "checkpoints", str(train_config_name), str(model_name)
        )
        if os.path.exists(candidate):
            return candidate
        print(f"[XR1] 微调检查点不存在 ({candidate})，回退到预训练权重")

    pretrained = usr_args.get("pretrained_ckpt")
    if not pretrained:
        raise ValueError(
            "无法确定 XR-1 权重路径：请在 deploy_policy.yml 里设置 model_path 或 pretrained_ckpt"
        )
    return _relocate_models_path(pretrained, "pretrained_ckpt")


def get_model(usr_args):
    """Create and return an XR-1 policy model instance.

    usr_args 关键字段:
        model_path          — 直接指定权重目录/文件（最高优先级）
        pretrained_ckpt     — 官方发布权重 model_states.pt 的路径（兜底）
        backbone_path       — 本地 Qwen3-VL-4B-Instruct 目录
        stats_path          — 动作/状态统计量 JSON（由 convert_lerobot_to_xr1.py 产出）
        xr1_step            — 每次推理执行多少步（<= 30）
        attn_implementation — auto / flash_attention_2 / sdpa
    """
    repo_root = os.path.abspath(os.path.join(parent_directory, "..", ".."))

    model_path = _resolve_model_path(usr_args, repo_root)
    backbone_path = usr_args.get("backbone_path") or ""
    if backbone_path:
        backbone_path = _relocate_models_path(backbone_path, "backbone_path")
    if not backbone_path:
        raise ValueError("必须在 deploy_policy.yml 里设置 backbone_path (本地 Qwen3-VL-4B-Instruct)")

    xr1_repo = os.path.expanduser(str(usr_args.get("xr1_repo") or DEFAULT_XR1_REPO))

    stats_path = usr_args.get("stats_path")
    if stats_path:
        stats_path = os.path.expanduser(str(stats_path))
        if not os.path.isabs(stats_path):
            stats_path = os.path.join(repo_root, stats_path)
    else:
        # 没显式给就按 train_config_name 去 training_data 里找同名的统计量。
        # 统计量和权重不配套会让动作幅度整体错，所以宁可自动找也别默默用 demo 的。
        train_config_name = usr_args.get("train_config_name")
        if train_config_name:
            candidate = os.path.join(
                repo_root, "policy", "xr1", "training_data", str(train_config_name), "xr1_stats.json"
            )
            if os.path.isfile(candidate):
                print(f"[XR1] 自动匹配到统计量: {candidate}")
                stats_path = candidate

    device = usr_args.get("pytorch_device") or "cuda"
    max_joint_delta = usr_args.get("max_joint_delta")

    model = XR1(
        model_path=model_path,
        backbone_path=backbone_path,
        xr1_repo=xr1_repo,
        stats_path=stats_path,
        xr1_step=int(usr_args.get("xr1_step", 10)),
        device=device,
        decode_mode=usr_args.get("decode_mode", "auto"),
        ik_shrink_retry=usr_args.get("ik_shrink_retry", (0.5, 0.25, 0.1)),
        attn_implementation=usr_args.get("attn_implementation", "auto"),
        max_joint_delta=float(max_joint_delta) if max_joint_delta is not None else None,
    )
    return model


def eval(env, model, obs):
    """Run one inference cycle and execute actions in the environment."""
    # eef_ik 解码要用 EmbodiChain 的 FK/IK，第一次调用时绑定 robot
    if model.decode_mode == "eef_ik":
        model.bind_env(env)

    # Set language instruction if first call
    if model.observation_window is None:
        instruction = getattr(env, "_current_instruction", None)
        model.set_language(instruction)

    # Encode and update observation window
    img_arr, state = encode_obs(obs)
    model.update_observation_window(img_arr, state)

    # XR-1 一次推理产出 30 步绝对 qpos，只执行前 xr1_step 步后重新观测
    actions = model.get_action()[: model.xr1_step]

    final_obs = obs
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
    """Reset XR-1 internal state (observation window and instruction)."""
    model.reset_obsrvationwindows()
