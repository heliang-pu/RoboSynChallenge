# ----------------------------------------------------------------------------
# Motus policy adapter for RoboSynChallenge.
#
# Implements the evaluator contract used by scripts/eval_policy.py:
#     get_model(usr_args: dict) -> model
#     eval(env, model, obs)     -> (obs, info, truncated)
#     reset_model(model)        -> None
#
#   bash policy/motus/eval.sh <task_name> <setting> <ckpt_dir> <model_name> <gpu>
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import torch

_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
if _POLICY_DIR not in sys.path:
    sys.path.insert(0, _POLICY_DIR)

from motus_model import MotusPolicy  # noqa: E402


def _resolve(usr_args, key, env_var, default=None):
    value = usr_args.get(key)
    if value in (None, "", "null"):
        value = os.environ.get(env_var, default)
    return value


def encode_obs(obs):
    """EmbodiChain observation -> (three camera frames, 14-dim qpos).

    Keys:
        sensor/cam_high/color        (1, 480, 640, 4) uint8
        sensor/cam_left_wrist/color  (1, 480, 640, 4) uint8
        sensor/cam_right_wrist/color (1, 480, 640, 4) uint8
        robot/qpos                   (1, 14) float

    Camera order matters: Motus stitches high on top, left bottom-left,
    right bottom-right.
    """
    img_high = obs["sensor"]["cam_high"]["color"][0, ..., :3]
    img_left = obs["sensor"]["cam_left_wrist"]["color"][0, ..., :3]
    img_right = obs["sensor"]["cam_right_wrist"]["color"][0, ..., :3]

    state = obs["robot"]["qpos"][0]
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    state = np.asarray(state, dtype=np.float32).reshape(-1)

    return [img_high, img_left, img_right], state


def _format_env_action(action, env):
    """Motus emits absolute qpos; the env consumes a (1, 14) float32 tensor."""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    if action_array.shape[0] < env_action_dim:
        raise ValueError(
            f"Motus action has dim {action_array.shape[0]}, but env expects {env_action_dim}."
        )
    action_array = action_array[:env_action_dim]
    action_tensor = torch.as_tensor(
        action_array, dtype=torch.float32, device=env.unwrapped.device
    )
    return action_tensor.unsqueeze(0)


def get_model(usr_args):
    """Build the Motus policy.

    usr_args (deploy_policy.yml) fields:
        ckpt_path            directory containing mp_rank_00_model_states.pt
        wan_path             Wan2.2-TI2V-5B dir (VAE + umT5 encoder + tokenizer)
        vlm_path             Qwen3-VL-2B-Instruct dir (processor/config only)
        model_config         yml under policy/motus/configs — MUST match the
                             config the checkpoint was trained with
        num_inference_steps  denoising steps (default from model_config)
        execute_steps        chunk actions executed per inference
        action_repeat        env steps each predicted action is held for
        action_normalization "none" (robotwin-format training) | "minmax"
        t5_mode              auto | cpu | keep | cache_only
    """
    ckpt_path = _resolve(usr_args, "ckpt_path", "MOTUS_CKPT")
    if not ckpt_path:
        ckpt_path = usr_args.get("model_name")  # eval.sh passes the ckpt dir here
    wan_path = _resolve(usr_args, "wan_path", "MOTUS_WAN_PATH")
    vlm_path = _resolve(usr_args, "vlm_path", "MOTUS_VLM_PATH")

    if not ckpt_path:
        raise ValueError("ckpt_path (or MOTUS_CKPT) must point at the Motus checkpoint directory")
    if not wan_path:
        raise ValueError("wan_path (or MOTUS_WAN_PATH) must point at Wan2.2-TI2V-5B")
    if not vlm_path:
        raise ValueError("vlm_path (or MOTUS_VLM_PATH) must point at Qwen3-VL-2B-Instruct")

    model_config = usr_args.get("model_config") or "configs/robosyn_infer.yml"
    if not os.path.isabs(model_config):
        model_config = os.path.join(_POLICY_DIR, model_config)

    device = usr_args.get("pytorch_device") or "cuda"
    if device == "cpu":
        # deploy_policy.yml's pytorch_device drives the simulator; Motus needs a GPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"

    stat_path = usr_args.get("stat_path")
    if stat_path and not os.path.isabs(stat_path):
        stat_path = os.path.join(_POLICY_DIR, stat_path)

    return MotusPolicy(
        checkpoint_path=ckpt_path,
        model_config_path=model_config,
        wan_path=wan_path,
        vlm_path=vlm_path,
        device=device,
        num_inference_steps=usr_args.get("num_inference_timesteps")
        or usr_args.get("num_inference_steps"),
        execute_steps=usr_args.get("execute_steps"),
        action_repeat=int(usr_args.get("action_repeat", 1)),
        action_normalization=usr_args.get("action_normalization", "none"),
        stat_path=stat_path,
        embodiment_type=usr_args.get("embodiment_type", "robosyn"),
        gripper_indices=usr_args.get("gripper_indices", (6, 13)),
        gripper_scale=float(usr_args.get("gripper_scale", 1.0)),
        gripper_limits=usr_args.get("gripper_limits"),
        action_clip=usr_args.get("action_clip"),
        t5_mode=usr_args.get("t5_mode", "auto"),
        t5_cache_dir=usr_args.get("t5_cache_dir"),
        save_video_debug=bool(usr_args.get("save_video_debug", False)),
    )


def eval(env, model, obs):
    """One inference cycle: observe, predict a chunk, execute it."""
    if model.observation_window is None:
        instruction = getattr(env, "_current_instruction", None)
        model.set_language(instruction)

    img_arr, state = encode_obs(obs)
    model.update_observation_window(img_arr, state)

    actions = model.get_action()

    final_obs, info, truncated = obs, {}, False
    for action in actions:
        action_tensor = _format_env_action(action, env)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        # The gym_config action manager may support delta actions; the default
        # (and what Motus predicts) is absolute qpos.
        _trunc = truncated.any() if hasattr(truncated, "any") else truncated
        if bool(_trunc):
            break

    # Refresh the window so the next call conditions on the latest frame.
    img_arr, state = encode_obs(final_obs)
    model.update_observation_window(img_arr, state)

    return final_obs, info, truncated


def reset_model(model):
    """Clear per-episode state (the cached T5 embedding is intentionally kept)."""
    model.reset_obsrvationwindows()
