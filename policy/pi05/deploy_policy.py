# ----------------------------------------------------------------------------
# π₀ Policy Adapter for RoboSynChallenge
#
# 遵循 RoboTwin 统一评估接口:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import os
import sys
import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.insert(0, parent_directory)

from concurrent.futures import ThreadPoolExecutor

from pi_model import PI0
from policy.inference_timing import finish_inference, start_inference
from rtc_runtime import ASYNC_MODES, ChunkScheduler, PhaseChunkSchedule



def _any_true(value):
    """Convert scalar/array/tensor done flags to a Python bool."""
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def _to_numpy(value):
    """Host numpy view of an observation field, whether it is a CPU/CUDA tensor or an array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _format_env_action(action, env):
    """Convert pi0 output into the torch action format EmbodiChain accepts."""
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
    """Convert gym Gymnasium Dict observation to π₀ input format.

    EmbodiChain observation keys:
        "sensor/cam_high/color"        -> base camera
        "sensor/cam_left_wrist/color"  -> left wrist
        "sensor/cam_right_wrist/color" -> right wrist
        "robot/qpos"                   -> joint state

    Returns:
        img_arr:  list of [img_front, img_right, img_left] as (H, W, C) numpy arrays
        state:    joint state vector
    """
    img_front_raw = obs["sensor"]["cam_high"]["color"]
    img_left_raw = obs["sensor"]["cam_left_wrist"]["color"]
    img_right_raw = obs["sensor"]["cam_right_wrist"]["color"]

    # With GPU physics (--device cuda) the observation tensors live on the sim
    # device; the openpi / realtime-vla preprocessing wants host numpy arrays.
    img_front = _to_numpy(img_front_raw[0, ..., :3])
    img_left = _to_numpy(img_left_raw[0, ..., :3])
    img_right = _to_numpy(img_right_raw[0, ..., :3])

    # Joint state — (num_envs, num_joints) -> squeeze env dim
    state = _to_numpy(obs["robot"]["qpos"][0])
    img_arr = [img_front, img_right, img_left]

    return img_arr, state


def get_model(usr_args):
    """Create and return a π₀ policy model instance.

    usr_args 中需要的字段:
        train_config_name  — openpi training config name
        model_name         — model name (e.g. "pi0_base")
        checkpoint_id      — checkpoint step number
        pi0_step           — number of action steps to execute per inference (default 10)
    """
    train_config_name = usr_args.get("train_config_name")
    model_name = usr_args.get("model_name")
    checkpoint_id = int(usr_args.get("checkpoint_id", 30000))
    pi0_step = int(usr_args.get("pi0_step", 10))
    pytorch_device = usr_args.get("pytorch_device", "cuda")

    async_mode = str(usr_args.get("async_mode", "off")).lower()
    if async_mode not in ASYNC_MODES:
        raise ValueError(f"async_mode must be one of {ASYNC_MODES}, got {async_mode!r}")
    inference_delay = int(usr_args.get("inference_delay", 0))
    if async_mode == "off":
        inference_delay = 0
    rtc_enabled = bool(usr_args.get("rtc", False))
    phase_config = usr_args.get("phase_action_chunks")
    phase_schedule = PhaseChunkSchedule(phase_config) if phase_config else None
    if phase_schedule is not None and async_mode != "off":
        raise ValueError(
            "phase_action_chunks currently requires async_mode='off'; changing H "
            "while an asynchronous inference is in flight is not deterministic"
        )
    initial_execution_horizon = (
        phase_schedule.select(0).execution_horizon
        if phase_schedule is not None
        else pi0_step
    )

    if train_config_name is None or model_name is None:
        raise ValueError(
            "train_config_name and model_name must be provided in usr_args"
        )

    inference_backend = str(usr_args.get("inference_backend", "jax"))
    if inference_backend != "jax" and (rtc_enabled or async_mode != "off"):
        raise ValueError(
            f"inference_backend={inference_backend!r} only supports the synchronous "
            "path: RTC guidance and async_mode need the jax backend"
        )

    model = PI0(
        train_config_name=train_config_name,
        model_name=model_name,
        checkpoint_id=checkpoint_id,
        pi0_step=pi0_step,
        pytorch_device=pytorch_device,
        max_guidance_weight=float(usr_args.get("max_guidance_weight", 10.0)),
        rtc_correction=str(usr_args.get("rtc_correction", "vjp")),
        inference_backend=inference_backend,
        checkpoint_root=usr_args.get("checkpoint_root"),
        converted_checkpoint=usr_args.get("converted_checkpoint"),
        realtime_vla_dir=usr_args.get("realtime_vla_dir"),
        tokenizer_path=usr_args.get("tokenizer_path"),
        prompt_for_allocation=usr_args.get("prompt_for_allocation", "click the bell"),
    )

    model.async_mode = async_mode
    model.scheduler = ChunkScheduler(
        action_horizon=model.action_horizon,
        execution_horizon=initial_execution_horizon,
        inference_delay=inference_delay,
        rtc_enabled=rtc_enabled,
        prefix_attention_schedule=str(usr_args.get("prefix_attention_schedule", "exp")),
    )
    model.executor = ThreadPoolExecutor(max_workers=1) if async_mode == "real" else None
    model.phase_chunk_schedule = phase_schedule
    model.active_phase = None

    setup = model.scheduler.describe()
    setup["async_mode"] = async_mode
    setup["rtc_correction"] = model.rtc_correction
    setup["phase_action_chunks"] = (
        phase_schedule.describe() if phase_schedule is not None else None
    )
    print(f"[pi05] chunk runtime: {setup}")
    if model.scheduler.clamped_from is not None:
        print(
            f"[pi05] WARNING: execution_horizon {model.scheduler.clamped_from} + "
            f"inference_delay {inference_delay} exceeds action_horizon "
            f"{model.action_horizon}; clamped to {model.scheduler.execution_horizon}."
        )
    return model


def _infer(model, obs, guidance):
    """Encode `obs` and sample one chunk of absolute environment actions."""
    img_arr, state = encode_obs(obs)
    model.update_observation_window(img_arr, state)
    return model.get_action(guidance)


def eval(env, model, obs):
    """Advance the environment by one execution horizon.

    Three runtimes share this path, selected by `model.async_mode`:

    * ``off``  -- synchronous. The env is frozen while the model runs, which is
      the original behaviour: sample a chunk, execute `pi0_step` of it, repeat.
    * ``sim``  -- the env keeps stepping on the *old* chunk for a fixed
      `inference_delay` steps before the new one takes over. Inference is still
      computed inline, so the delay is exact and the run is reproducible.
    * ``real`` -- inference runs on a background thread and the new chunk lands
      whenever it actually finishes. Realistic, but wall-clock dependent.

    With RTC enabled the sampler is additionally guided so each new chunk agrees
    with the plan it is replacing over the already-committed prefix.
    """
    if model.observation_window is None:
        instruction = getattr(env, "_current_instruction", None)
        model.set_language(instruction)

    sched = model.scheduler
    phase_schedule = getattr(model, "phase_chunk_schedule", None)
    if phase_schedule is not None:
        phase = phase_schedule.select(sched.step_index)
        sched.set_execution_horizon(phase.execution_horizon)
        execution_budget = min(
            sched.execution_horizon,
            phase.step_budget(sched.step_index),
        )
        if model.active_phase != phase.name:
            if model.active_phase is not None:
                sched.request_replan()
            print(
                f"[pi05] phase={phase.name!r} step={sched.step_index} "
                f"execution_horizon={sched.execution_horizon}"
            )
            model.active_phase = phase.name
    else:
        execution_budget = sched.execution_horizon
    inference_times_s = []
    final_obs, info, truncated = obs, None, False
    steps_taken = 0

    while steps_taken < execution_budget:
        # `sched.pending` only fills once a chunk exists, so in `real` mode the
        # in-flight future is what keeps us from launching the same replan twice.
        in_flight = getattr(model, "pending_future", None) is not None
        if sched.should_launch() and not in_flight:
            launch_step = sched.step_index
            guidance = sched.guidance(launch_step)
            first_chunk = sched.chunk is None

            if model.async_mode == "real" and not first_chunk:
                # Hand inference to the worker and keep stepping the old chunk;
                # the future is drained below once it resolves.
                model.pending_future = model.executor.submit(_infer, model, final_obs, guidance)
                model.pending_launch = (launch_step, guidance is not None, start_inference(model.pytorch_device))
            else:
                started_at = start_inference(model.pytorch_device)
                actions = _infer(model, final_obs, guidance)
                finish_inference(started_at, inference_times_s, model.pytorch_device)
                if guidance is not None:
                    sched.guided_launches += 1
                # The first chunk of an episode has nothing to overlap with, so
                # it lands immediately no matter what the delay model says.
                sched.stage(actions, launch_step, land_step=launch_step if first_chunk else None)
                if first_chunk:
                    sched.adopt()

        if getattr(model, "pending_future", None) is not None and model.pending_future.done():
            actions = model.pending_future.result()
            launch_step, was_guided, started_at = model.pending_launch
            finish_inference(started_at, inference_times_s, model.pytorch_device)
            if was_guided:
                sched.guided_launches += 1
            sched.stage(actions, launch_step, land_step=sched.step_index)
            model.pending_future = None

        if sched.should_adopt():
            sched.adopt()

        action_tensor = _format_env_action(sched.action(), env)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        # The `gym_config` setting configures the `actionmanager` to support delta
        # action input; the default action must be absolute `qpos`.
        sched.advance()
        steps_taken += 1

        # Check success after every environment step.  Item assembly can satisfy
        # its 3 mm contact criterion only briefly inside an action chunk; waiting
        # until the whole horizon has executed can miss that state.
        if _any_true(env.get_wrapper_attr("is_task_success")()):
            break
        if _any_true(terminated) or _any_true(truncated):
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    """Reset π₀ internal state (observation window, instruction, chunk timeline)."""
    pending = getattr(model, "pending_future", None)
    if pending is not None:
        pending.cancel()
        model.pending_future = None
    model.scheduler.reset()
    model.active_phase = None
    model.reset_obsrvationwindows()
