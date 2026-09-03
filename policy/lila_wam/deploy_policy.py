# ----------------------------------------------------------------------------
# LiLa-WAM policy adapter for RoboSynChallenge.
#
# Implements the evaluator contract used by scripts/eval_policy.py:
#     get_model(usr_args: dict) -> model
#     eval(env, model, obs)     -> (obs, info, truncated, inference_times_s)
#     reset_model(model)        -> None
#
#   bash policy/lila_wam/eval.sh <task_name> <setting> <checkpoint> <model_name> <gpu>
# ----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policy.inference_timing import finish_inference, start_inference  # noqa: E402
from policy.lila_wam.lila_infer import LilaWamInference, encode_action  # noqa: E402

POLICY_DIR = Path(__file__).resolve().parent


def _resolve(usr_args: dict, key: str, env_var: str, default=None):
    value = usr_args.get(key)
    if value in (None, "", "null"):
        value = os.environ.get(env_var, default)
    return value


def _latest_checkpoint(directory: Path) -> Path:
    checkpoints = sorted(
        directory.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint_epoch_*.pt found in {directory}")
    return checkpoints[-1]


def get_model(usr_args: dict):
    checkpoint = _resolve(usr_args, "checkpoint_path", "LILA_WAM_CHECKPOINT")
    if not checkpoint:
        raise ValueError("checkpoint_path is required (deploy_policy.yml or LILA_WAM_CHECKPOINT)")
    checkpoint = Path(str(checkpoint)).expanduser().resolve()
    run_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    if checkpoint.is_dir():
        checkpoint = _latest_checkpoint(checkpoint)

    config_path = _resolve(usr_args, "config_path", "LILA_WAM_CONFIG") or run_dir / "config.yaml"
    norm_stats_path = (
        _resolve(usr_args, "norm_stats_path", "LILA_WAM_NORM_STATS") or run_dir / "norm_stats.json"
    )
    task_cond_dir = _resolve(usr_args, "task_cond_dir", "LILA_WAM_TASK_COND_DIR")

    task_name = usr_args.get("task_cond_name") or usr_args.get("task_name")
    device = usr_args.get("device") or usr_args.get("pytorch_device") or "cuda"

    model = LilaWamInference(
        config_path=config_path,
        checkpoint_path=checkpoint,
        norm_stats_path=norm_stats_path,
        task_cond_dir=task_cond_dir,
        task_name=task_name,
        device=str(device),
        state_obs_path=str(usr_args.get("state_obs_path", "robot/qpos")),
        action_execution_horizon=usr_args.get("action_execution_horizon"),
        num_inference_steps=usr_args.get("num_inference_steps"),
        exposure_match=usr_args.get("exposure_match"),
    )
    model.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    return model


def eval(env, model, obs):
    """Predict one action chunk and execute its execution horizon in the env."""
    inference_times_s: list[float] = []
    started_at = start_inference(model.inference_device)
    actions = model.get_action(obs)
    finish_inference(started_at, inference_times_s, model.inference_device)

    final_obs = obs
    info = None
    truncated = False
    for action in actions:
        action_tensor = encode_action(action, env, strict=model.strict_action_dim)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if env.get_wrapper_attr("is_task_success")():
            break
        if isinstance(truncated, torch.Tensor):
            is_truncated = bool(truncated.any().item())
        elif isinstance(truncated, np.ndarray):
            is_truncated = bool(truncated.any())
        else:
            is_truncated = bool(truncated)
        if is_truncated:
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    model.reset()
