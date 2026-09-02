#!/usr/bin/env python
# ----------------------------------------------------------------------------
# RoboSynChallenge — unified policy eval script
# python scripts/eval_policy_parallel.py --config policy/{policy_name}/deploy_policy.yml
# ----------------------------------------------------------------------------

import os
import sys
import argparse
import importlib
import json
import platform
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import yaml
from tqdm.auto import tqdm

# Add policy directory to path for dynamic imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
POLICY_DIR = os.path.join(REPO_ROOT, "policy")


def prepend_existing_paths(paths):
    """Prepend existing paths to sys.path while preserving the given order."""
    for path in reversed(paths):
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)


WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
EMBODICHAIN_ROOT = os.environ.get(
    "EMBODICHAIN_ROOT", os.path.join(WORKSPACE_ROOT, "EmbodiChain")
)
prepend_existing_paths([REPO_ROOT, POLICY_DIR, EMBODICHAIN_ROOT])


def add_policy_dependency_paths(policy_name):
    """Expose policy-local pure-Python source without replacing simulator Python."""
    policy_root = os.path.join(POLICY_DIR, policy_name)
    paths = [
        policy_root,
        os.path.join(policy_root, "src"),
        os.path.join(policy_root, "packages", "openpi-client", "src"),
    ]
    prepend_existing_paths(paths)

import robosynchallenge
import embodichain.lab.gym.utils.gym_utils as gym_utils
from embodichain.utils import logger as emb_logger

CHALLENGE_MANAGER_MODULES = [
    "robosynchallenge.managers.actions",
    "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events",
    "robosynchallenge.managers.observations",
]

for module in CHALLENGE_MANAGER_MODULES:
    if module not in gym_utils.DEFAULT_MANAGER_MODULES:
        gym_utils.DEFAULT_MANAGER_MODULES.append(module)


def load_policy_adapter(policy_name):
    """Dynamically import a policy adapter package.

    Each policy adapter must live at policy/<policy_name>/ and export these
    functions from its top-level namespace (via __init__.py):

        get_model(usr_args: dict) -> model
        eval(env: gym.Env, model, obs) -> (obs, info, truncated, inference_times)
        reset_model(model) -> None
    """
    add_policy_dependency_paths(policy_name)
    try:
        pkg = importlib.import_module(f"policy.{policy_name}")
    except ImportError as e:
        print(f"Error: cannot import policy '{policy_name}': {e}")
        sys.exit(1)

    for fn_name in ("get_model", "eval", "reset_model"):
        if not hasattr(pkg, fn_name):
            print(f"Error: policy '{policy_name}' missing required function '{fn_name}'")
            sys.exit(1)

    return pkg


def _set_default(config, key, value):
    if config.get(key) is None:
        config[key] = value


def _as_int_or_none(value):
    if value is None:
        return None
    return int(value)


def select_cuda_device(config):
    """Align an unqualified policy CUDA device with the simulator GPU."""
    pytorch_device = str(config.get("pytorch_device", ""))
    gpu_id = int(config.get("gpu_id", 0))
    if pytorch_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    # JAX 权重默认落在 jax.devices()[0]。当仿真被指到非 0 号物理卡时(多卡机上
    # 0 号卡常被别的进程占满),权重必须跟着 gpu_id 走,否则 device_put 会 OOM。
    # 多卡机上还要把 JAX 只限制在这一张物理卡上:默认它会在全部 8 张卡上各预留 75% 显存,
    # 和同机别的作业互相挤爆。jax_cuda_visible_devices 必须在后端初始化(首次 jax.devices())之前设。
    try:
        import jax

        jax.config.update("jax_cuda_visible_devices", str(gpu_id))
        _devs = jax.devices()
        jax.config.update("jax_default_device", _devs[0])
    except Exception:  # JAX 不可用或后端未初始化时静默跳过
        pass


def resolve_episode_max_steps(config, gym_config):
    """Use the task env limit, with the deploy limit only as a fallback."""
    deploy_max_steps = _as_int_or_none(config.get("max_steps"))
    gym_max_steps = _as_int_or_none(gym_config.get("max_episode_steps"))
    max_env_steps = gym_max_steps or deploy_max_steps or 300
    return max_env_steps, deploy_max_steps, gym_max_steps


def configure_rollout_saving(config, gym_config):
    """Enable rollout dataset saving for sim-RECAP data collection.

    Mutates ``gym_config`` in place (before make_env deep-copies it):
      * redirects the dataset functor save_path away from the expert dataset
        directory so rollouts never pollute demonstration data;
      * sets ``save_failed_episodes=True`` so failed rollouts are kept (they
        are the negative examples advantage conditioning needs);
      * flips ``filter_dataset_saving`` to False so the dataset manager runs.

    Returns the resolved rollout dataset directory, or None when disabled
    (``rollout_save`` unset/false keeps evaluation behavior unchanged).
    """
    if not config.get("rollout_save"):
        return None

    dataset_cfgs = gym_config.get("env", {}).get("dataset", {})
    functor_names = [
        name
        for name, functor_cfg in dataset_cfgs.items()
        if isinstance(functor_cfg, dict) and "func" in functor_cfg
    ]
    if not functor_names:
        print("Error: rollout_save=true but gym_config has no dataset functor")
        sys.exit(1)

    save_dir = config.get("rollout_save_path")
    if not save_dir:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = (
            f"lerobot_dataset/rollouts/"
            f"{config.get('task_name')}_{config.get('setting')}_{stamp}"
        )

    for name in functor_names:
        dataset_cfgs[name]["save_failed_episodes"] = True
        params = dict(dataset_cfgs[name].get("params", {}))
        params["save_path"] = str(save_dir)
        dataset_cfgs[name]["params"] = params

    config["filter_dataset_saving"] = False

    # Resolve the same way LeRobotRecorder does (relative to the repo root).
    from robosynchallenge.data.constants import ROBOSYNCHALLENGE_ROOT

    resolved = Path(save_dir)
    if not resolved.is_absolute():
        resolved = Path(ROBOSYNCHALLENGE_ROOT) / save_dir
    print(f"Rollout saving enabled -> {resolved} (failed episodes kept)")
    return resolved


def get_recorder_dataset_dir(env):
    """Actual dataset directory created by the LeRobot recorder.

    The recorder nests an auto-named ``<robot>_<scene>_<task>_NNN`` directory
    under the configured save_path, so the sidecar must be written there, not
    at the save_path root.
    """
    dataset_manager = getattr(getattr(env, "unwrapped", env), "dataset_manager", None)
    if dataset_manager is None:
        return None
    for mode_cfgs in getattr(dataset_manager, "_mode_functor_cfgs", {}).values():
        for functor_cfg in mode_cfgs:
            functor = getattr(functor_cfg, "func", None)
            full_path = getattr(functor, "dataset_full_path", None)
            if full_path:
                return Path(full_path)
    return None


def get_saved_episode_count(env):
    """Best-effort count of episodes the LeRobot recorder has written."""
    dataset_manager = getattr(getattr(env, "unwrapped", env), "dataset_manager", None)
    if dataset_manager is None:
        return None

    episode_counts = []
    for mode_cfgs in getattr(dataset_manager, "_mode_functor_cfgs", {}).values():
        for functor_cfg in mode_cfgs:
            functor = getattr(functor_cfg, "func", None)
            if hasattr(functor, "curr_episode"):
                episode_counts.append(int(functor.curr_episode))
    return max(episode_counts) if episode_counts else None


def summarize_task_metrics(info, env_index=0):
    """Flatten ``info['metrics']`` (one env) into plain Python scalars/lists.

    ``compute_task_state`` returns per-env tensors; sequential eval only drives
    env 0, while parallel eval passes the slot index of the episode it labels.
    """
    if not isinstance(info, dict):
        return None
    metrics = info.get("metrics")
    if not metrics:
        return None

    summary = {}
    for key, value in metrics.items():
        try:
            if isinstance(value, torch.Tensor):
                value = value[env_index] if value.ndim > 0 else value
                summary[key] = value.tolist()
            elif isinstance(value, (bool, int, float)):
                summary[key] = value
            elif isinstance(value, np.ndarray):
                value = value[env_index] if value.ndim > 0 else value
                summary[key] = np.asarray(value).reshape(-1).tolist()
        except Exception:  # noqa: BLE001 - diagnostics must never break eval
            continue
    return summary or None


def format_task_metrics(summary):
    parts = []
    for key, value in summary.items():
        if isinstance(value, list):
            parts.append(f"{key}=[" + ", ".join(f"{v:.4f}" for v in value) + "]")
        elif isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return "  ".join(parts)


def write_rollout_success_sidecar(dataset_dir, config, episode_records, saved_count):
    """Persist per-episode success labels next to the rollout dataset.

    The sidecar is consumed by scripts/label_rollout_dataset.py, which writes
    ``episode_success`` ("success"/"failure") into the LeRobot episodes
    metadata in the format Evo-RL's value training expects.
    """
    sidecar = {
        "task_name": config.get("task_name"),
        "setting": config.get("setting"),
        "policy_name": config.get("policy_name"),
        "train_config_name": config.get("train_config_name"),
        "model_name": config.get("model_name"),
        "seed": config.get("seed"),
        "labels_field": "episode_success",
        "saved_episode_count": saved_count,
        "episodes": episode_records,
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = dataset_dir / "episode_success.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    if saved_count is not None and saved_count != len(episode_records):
        print(
            f"Warning: recorder saved {saved_count} episodes but "
            f"{len(episode_records)} were labeled; "
            "label_rollout_dataset.py will refuse this dataset."
        )
    print(f"Episode success labels written: {sidecar_path}")


class EpisodeVideoRecorder:
    """Record one RGB observation stream per episode through an ffmpeg pipe."""

    def __init__(self, save_dir, obs_keys="cam_high", fps=10, crf=23):
        self.save_dir = Path(save_dir)
        if isinstance(obs_keys, str):
            obs_keys = [key.strip() for key in obs_keys.split(",") if key.strip()]
        self.obs_keys = list(obs_keys)
        if not self.obs_keys:
            raise ValueError("At least one eval video observation key is required.")
        self.fps = int(fps)
        self.crf = int(crf)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._process = None
        self._episode_idx = None
        self._episode_seed = None
        self._tmp_path = None
        self._frame_size = None
        self._frame_count = 0

    def start_episode(self, episode_idx, seed):
        self.close_episode(success=False)
        self._episode_idx = int(episode_idx)
        self._episode_seed = int(seed)
        self._tmp_path = self.save_dir / f"episode_{episode_idx:03d}_seed_{seed}.tmp.mp4"
        self._process = None
        self._frame_size = None
        self._frame_count = 0

    def record(self, obs):
        if self._episode_idx is None:
            return

        frames = [self._extract_frame(obs, obs_key) for obs_key in self.obs_keys]
        frame = np.ascontiguousarray(np.concatenate(frames, axis=1))
        height, width = frame.shape[:2]
        if self._process is None:
            self._start_writer(width, height)
        elif self._frame_size != (width, height):
            raise ValueError(
                f"Video frame size changed from {self._frame_size} to {(width, height)}."
            )

        self._process.stdin.write(frame.tobytes())
        self._frame_count += 1

    def _extract_frame(self, obs, obs_key):
        frame = obs["sensor"][obs_key]["color"]
        if frame is None:
            raise KeyError(
                f"EmbodiChain camera color observation 'sensor/{obs_key}/color' "
                "not found for video recording."
            )

        frame = frame[0, ..., :3]
        # GPU physics keeps observations on the sim device; numpy cannot read a CUDA tensor directly.
        frame = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
        if frame.dtype != np.uint8:
            max_value = float(np.nanmax(frame)) if frame.size else 0.0
            if max_value <= 1.5:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def close_episode(self, success=None):
        if self._process is None:
            self._reset_episode_state()
            return

        self._process.stdin.close()
        return_code = self._process.wait()
        if return_code != 0:
            tmp_path = self._tmp_path
            self._reset_episode_state()
            raise RuntimeError(f"ffmpeg failed while saving eval video: {tmp_path}")

        if self._frame_count > 0:
            status = "success" if success else "fail"
            final_path = (
                self.save_dir
                / f"episode_{self._episode_idx:03d}_seed_{self._episode_seed}_{status}.mp4"
            )
            os.replace(self._tmp_path, final_path)
            print(f"  Video saved: {final_path}")
        elif self._tmp_path and self._tmp_path.exists():
            self._tmp_path.unlink()

        self._reset_episode_state()

    def _start_writer(self, width, height):
        self._frame_size = (width, height)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            str(self.crf),
            str(self._tmp_path),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def _reset_episode_state(self):
        self._process = None
        self._episode_idx = None
        self._episode_seed = None
        self._tmp_path = None
        self._frame_size = None
        self._frame_count = 0


class RecordingEnvProxy:
    """Proxy env.step/reset so policy adapters do not need video-specific code."""

    def __init__(self, env, recorder=None, reset_sync_steps=0):
        self._env = env
        self._recorder = recorder
        self._reset_sync_steps = int(reset_sync_steps)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs, info = self._sync_reset_obs(obs, info)
        if self._recorder:
            self._recorder.record(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._recorder and not (
            self._as_done(terminated) or self._as_done(truncated)
        ):
            self._recorder.record(obs)
        return obs, reward, terminated, truncated, info

    def _sync_reset_obs(self, obs, info):
        if self._reset_sync_steps <= 0:
            return obs, info

        env = getattr(self._env, "unwrapped", self._env)
        sim = getattr(env, "sim", None)
        if sim is None:
            return obs, info

        physics_dt = getattr(getattr(env, "sim_cfg", None), "physics_dt", None)
        sim.update(physics_dt, self._reset_sync_steps)

        if hasattr(env, "get_obs"):
            obs = env.get_obs()
        if hasattr(env, "get_info"):
            info = env.get_info()
        return obs, info

    @staticmethod
    def _as_done(value):
        if isinstance(value, torch.Tensor):
            return bool(value.any().item())
        if isinstance(value, np.ndarray):
            return bool(value.any())
        return bool(value)


def _as_bool_vector(value, num_envs, device):
    """Coerce env step outputs (tensor/ndarray/scalar) into a [num_envs] bool tensor."""
    if isinstance(value, torch.Tensor):
        vec = value.detach().to(device=device, dtype=torch.bool).reshape(-1)
    elif isinstance(value, np.ndarray):
        vec = torch.as_tensor(value.reshape(-1), dtype=torch.bool, device=device)
    else:
        vec = torch.full((num_envs,), bool(value), dtype=torch.bool, device=device)
    if vec.numel() == 1 and num_envs > 1:
        vec = vec.expand(num_envs).clone()
    return vec


class ParallelEvalProxy:
    """并行评估的裁判代理：逐步锁存 per-env 成功位，聚合喂给单环境写法的适配器。

    适配器契约不变：仍由适配器 env.step() 驱动、仍用
    ``env.get_wrapper_attr("is_task_success")()`` 决定是否提前 break——只是并行
    模式下该调用被本代理拦截，返回「所有在评 env 都已成功或截断」的聚合布尔，
    True 即等价于「本 wave 结束」。因此按官方模板写的适配器无需任何改动。

    官方口径逐条对应：
      * 成功 = 某次 step 后 is_task_success 为 True，且该 env 当步与此前均未截断；
      * 截断步不计成功（底层在截断步会对 done env 自动部分重置并清掉任务锁存，
        与官方「必须未 truncated」语义一致）；
      * 成功一经锁存不再回退，等价于官方单环境在成功当步立即结束 episode。
    """

    def __init__(self, env, num_envs):
        self._env = env
        self._num_envs = int(num_envs)
        self._device = env.unwrapped.device
        zeros = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._active = zeros.clone()
        self._latched_success = zeros.clone()
        self._truncated = zeros.clone()
        self._success_steps = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )
        self._done_metrics = [None] * self._num_envs
        self._wave_steps = 0
        self._terminated_warned = False

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_wrapper_attr(self, name):
        if name == "is_task_success":
            return self.all_done
        return self._env.get_wrapper_attr(name)

    @property
    def done_mask(self):
        return self._latched_success | self._truncated | ~self._active

    def all_done(self):
        return bool(self.done_mask.all().item())

    @property
    def wave_steps(self):
        return self._wave_steps

    def episode_result(self, slot, max_env_steps):
        success = bool(self._latched_success[slot].item())
        env_steps = int(self._success_steps[slot].item()) if success else max_env_steps
        return success, env_steps, self._done_metrics[slot]

    def begin_wave(self, active_count):
        self._active[:] = False
        self._active[:active_count] = True
        self._latched_success[:] = False
        self._truncated[:] = False
        self._success_steps[:] = 0
        self._done_metrics = [None] * self._num_envs
        self._wave_steps = 0

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action):
        if isinstance(action, torch.Tensor) and (
            action.ndim < 2 or action.shape[0] != self._num_envs
        ):
            raise ValueError(
                f"Parallel eval needs batched actions [num_envs={self._num_envs}, dof], "
                f"got shape {tuple(action.shape)} — this policy adapter is not "
                "batch-ready; run it with num_envs=1."
            )
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._wave_steps += 1

        term_now = _as_bool_vector(terminated, self._num_envs, self._device)
        if term_now.any() and not self._terminated_warned:
            # 挑战赛任务刻意把 compute_task_state 的 success/fail 位置零,正是为了
            # 避免评估中途触发底层 auto-reset。有任务打破该约定时并行口径不可信。
            print(
                "Warning: env reported terminated mid-episode; the simulator "
                "auto-resets those envs and parallel episode accounting may be "
                "wrong for this task."
            )
            self._terminated_warned = True

        trunc_now = _as_bool_vector(truncated, self._num_envs, self._device)
        succ_now = _as_bool_vector(
            self._env.get_wrapper_attr("is_task_success")(),
            self._num_envs,
            self._device,
        )
        newly_success = (
            succ_now
            & self._active
            & ~self._latched_success
            & ~self._truncated
            & ~trunc_now
        )
        if newly_success.any():
            self._success_steps[newly_success] = self._wave_steps
            self._latched_success |= newly_success
            for slot in newly_success.nonzero(as_tuple=False).flatten().tolist():
                self._done_metrics[slot] = summarize_task_metrics(info, env_index=slot)
        self._truncated |= trunc_now & self._active
        return obs, reward, terminated, truncated, info


def build_episode_plan(rng, max_episodes, num_shards, shard_index, fixed_episode_seed):
    """(episode_index, seed) list for this shard, drawn exactly like the sequential loop.

    rng 对每个 episode 序号都抽一次(包括不属于本分片的),保证任意
    (num_shards, num_envs) 组合下 episode k 的种子与单进程串行一致。
    """
    plan = []
    for episode in range(max_episodes):
        ep_seed = (
            int(fixed_episode_seed)
            if fixed_episode_seed is not None
            else int(rng.randint(0, 2**31 - 1))
        )
        if num_shards > 1 and episode % num_shards != shard_index:
            continue
        plan.append((episode, ep_seed))
    return plan


def settle_after_wave_reset(env, reset_sync_steps, obs, info):
    """Replicate RecordingEnvProxy._sync_reset_obs once per wave (not per slot)."""
    if reset_sync_steps <= 0:
        return obs, info
    raw = getattr(env, "unwrapped", env)
    sim = getattr(raw, "sim", None)
    if sim is None:
        return obs, info
    physics_dt = getattr(getattr(raw, "sim_cfg", None), "physics_dt", None)
    sim.update(physics_dt, int(reset_sync_steps))
    if hasattr(raw, "get_obs"):
        obs = raw.get_obs()
    if hasattr(raw, "get_info"):
        info = raw.get_info()
    return obs, info


def run_parallel_episodes(
    env,
    eval_env,
    policy_pkg,
    model,
    config,
    rng,
    max_episodes,
    max_env_steps,
    num_envs,
    num_shards,
    shard_index,
    fixed_episode_seed,
    reset_sync_steps,
):
    """Wave-parallel episode loop for num_envs > 1.

    种子协议与串行完全一致:rng 按 episode 序号逐个抽取、分片规则照旧,episode k
    永远拿到与单进程串行时相同的 seed_k;每个槽位用
    ``reset(seed=seed_k, options={"reset_ids": [slot]})`` 单独播种,场景与单环境
    同种子逐位一致(已实测)。已知偏差:interval 模式的随机化事件(如灯光)在
    批量下共享同一条 RNG 流,数值与串行不同但同分布。

    Returns the same accumulators the sequential loop builds so the summary and
    metrics-file code stays shared.
    """
    plan = build_episode_plan(
        rng, max_episodes, num_shards, shard_index, fixed_episode_seed
    )

    device = env.unwrapped.device
    episodes_run = 0
    success_count = 0
    episode_records = []
    action_steps = []
    all_inference_times = []
    episode_inference_totals = []

    total_waves = (len(plan) + num_envs - 1) // num_envs
    for wave_index, wave_start in enumerate(range(0, len(plan), num_envs)):
        wave = plan[wave_start : wave_start + num_envs]
        # 空槽也用本 wave 首个种子重置:保证整批 elapsed_steps 同步、统一在
        # max_env_steps 截断,避免残留状态在 wave 中途单独触发 auto-reset。
        slot_seeds = [seed for _, seed in wave]
        slot_seeds += [slot_seeds[0]] * (num_envs - len(wave))

        obs, info = None, None
        for slot, slot_seed in enumerate(slot_seeds):
            reset_ids = torch.tensor([slot], dtype=torch.int32, device=device)
            obs, info = eval_env.reset(
                seed=slot_seed, options={"reset_ids": reset_ids}
            )
        obs, info = settle_after_wave_reset(env, reset_sync_steps, obs, info)

        eval_env.begin_wave(active_count=len(wave))
        policy_pkg.reset_model(model)

        wave_inference_times = []
        episode_span = f"{wave[0][0] + 1:03d}..{wave[-1][0] + 1:03d}"
        progress_bar = tqdm(
            total=max_env_steps,
            desc=f"Wave {wave_index + 1:02d}/{total_waves:02d} (ep {episode_span})",
            unit="step",
            dynamic_ncols=True,
            leave=False,
        )
        try:
            stalled_steps = -1
            while not eval_env.all_done() and eval_env.wave_steps < max_env_steps:
                if eval_env.wave_steps == stalled_steps:
                    progress_bar.write(
                        "Warning: policy adapter made no env progress; aborting wave."
                    )
                    break
                stalled_steps = eval_env.wave_steps

                eval_result = policy_pkg.eval(eval_env, model, obs)
                if len(eval_result) == 4:
                    obs, info, _truncated, new_times = eval_result
                else:
                    obs, info, _truncated = eval_result
                    new_times = []
                wave_inference_times.extend(new_times)

                progress_bar.update(
                    max(0, min(eval_env.wave_steps, max_env_steps) - progress_bar.n)
                )
                done_count = int(
                    (eval_env.done_mask[: len(wave)]).sum().item()
                )
                progress_bar.set_postfix(
                    done=f"{done_count}/{len(wave)}",
                    inference=len(wave_inference_times),
                    infer_time=f"{new_times[-1]:.3f}s" if new_times else "cached",
                )
        finally:
            progress_bar.close()

        all_inference_times.extend(wave_inference_times)
        wave_inference_total = float(sum(wave_inference_times))
        wave_average = (
            float(np.mean(wave_inference_times)) if wave_inference_times else 0
        )

        for slot, (episode, ep_seed) in enumerate(wave):
            episodes_run += 1
            episode_success, env_steps, done_metrics = eval_env.episode_result(
                slot, max_env_steps
            )
            success_count += int(episode_success)
            effective_steps = env_steps if episode_success else max_env_steps
            action_steps.append(effective_steps)
            # 一次批量推理同时服务整个 wave,无法按 episode 拆分,
            # 这里记 wave 总耗时(该 episode 存续期间的推理墙钟)。
            episode_inference_totals.append(wave_inference_total)

            status = (
                "\033[92mSUCCESS\033[0m" if episode_success else "\033[91mFAIL\033[0m"
            )
            print(
                f"  Episode {episode + 1} (wave {wave_index + 1}, slot {slot}): "
                f"{status}; action_steps={effective_steps}/{max_env_steps}; "
                f"seed={ep_seed}"
            )
            print(
                f"  [{episodes_run:3d}/{len(plan)}] {status}  "
                f"(success rate: {success_count}/{episodes_run} = "
                f"{100 * success_count / episodes_run:.1f}%)"
            )
            episode_metrics = done_metrics or summarize_task_metrics(
                info, env_index=slot
            )
            if episode_metrics:
                print(f"  metrics: {format_task_metrics(episode_metrics)}")
        if wave_inference_times:
            print(
                f"  wave inference: {wave_average:.6f}s avg over "
                f"{len(wave_inference_times)} batched calls "
                f"(batch={num_envs})"
            )

    return (
        episodes_run,
        success_count,
        episode_records,
        action_steps,
        all_inference_times,
        episode_inference_totals,
    )


def create_eval_run_dir(config):
    """Create the shared directory for metrics and optional videos."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 多分片并发启动时同一秒内会撞目录名,后写的分片会覆盖先写的
    # evaluation_metrics.json(实测 item_assembly 4 分片只剩 3 份),带上分片号。
    num_shards = int(config.get("num_shards", 1) or 1)
    if num_shards > 1:
        timestamp += f"_s{int(config.get('shard_index', 0) or 0)}"
    save_root = Path(config.get("eval_result_dir") or "eval_result")
    checkpoint_path = config.get("checkpoint_path")
    model_name = config.get("model_name")
    if not model_name and checkpoint_path:
        model_name = Path(checkpoint_path).name
    save_dir = (
        save_root
        / str(config.get("task_name"))
        / str(config.get("policy_name"))
        / str(config.get("setting"))
        / str(config.get("train_config_name"))
        / str(model_name)
        / timestamp
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def create_video_recorder(config, run_dir=None):
    if not config.get("eval_video_log", False):
        return None

    run_dir = Path(run_dir) if run_dir else create_eval_run_dir(config)
    obs_keys = config.get("eval_video_obs_keys", config.get("eval_video_obs_key", "cam_high"))
    return EpisodeVideoRecorder(
        save_dir=run_dir / "videos",
        obs_keys=obs_keys,
        fps=10,
        crf=23,
    )


def make_env_from_configs(config, gym_config_dict, action_config_dict):
    """Build the gym env using EmbodiChain's config parser.

    This keeps nested robot/sensor/object dictionaries as EmbodiChain config
    objects instead of passing raw dicts into EmbodiedEnvCfg.
    """
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    gym_config = deepcopy(gym_config_dict)
    _set_default(gym_config, "num_envs", int(config.get("num_envs", 1)))
    _set_default(gym_config, "device", config.get("device", "cpu"))
    _set_default(gym_config, "headless", bool(config.get("headless", False)))
    # 渲染器默认跟随 EmbodiChain 的 auto 规则:RTX 卡 -> hybrid(行为不变),
    # A100/H100 等数据中心卡 -> fast-rt(实测 hybrid 在 A100 上只有 2 step/s)。
    _set_default(gym_config, "renderer", config.get("renderer", "hybrid"))  # 与官方脚本一致；A100/H100 等无 RT core 的卡需显式 --renderer auto
    _set_default(gym_config, "gpu_id", int(config.get("gpu_id", 0)))
    _set_default(gym_config, "arena_space", float(config.get("arena_space", 5.0)))
    # 不带索引的 "cuda" 在多卡机上会让部分张量落到 cuda:0:gpu_id≠0 时直接
    # device mismatch 报错,gpu_id=0 时也实测慢 5 倍以上(A100 3.5 vs 18 step/s)。
    if str(gym_config["device"]) == "cuda":
        gym_config["device"] = f"cuda:{int(gym_config['gpu_id'])}"

    max_env_steps, _, _ = resolve_episode_max_steps(config, gym_config)
    gym_config["max_episode_steps"] = max_env_steps

    env_cfg = gym_utils.config_to_cfg(
        gym_config,
        manager_modules=CHALLENGE_MANAGER_MODULES,
    )
    env_cfg.filter_dataset_saving = bool(config.get("filter_dataset_saving", True))
    env_cfg.sim_cfg = SimulationManagerCfg(
        headless=gym_config["headless"],
        sim_device=gym_config["device"],
        render_cfg=RenderCfg(renderer=gym_config["renderer"]),
        gpu_id=gym_config["gpu_id"],
        arena_space=gym_config["arena_space"],
    )
    physics_config = gym_config.get("physics", {})
    if "enable_ccd" in physics_config:
        env_cfg.sim_cfg.physics_config.enable_ccd = bool(
            physics_config["enable_ccd"]
        )

    action_kwargs = {}
    if action_config_dict:
        action_kwargs = (
            action_config_dict
            if "action_config" in action_config_dict
            else {"action_config": action_config_dict}
        )

    env = gym.make(
        id=gym_config["id"],
        max_episode_steps=max_env_steps,
        cfg=env_cfg,
        **action_kwargs,
    )
    return env, gym_config

def find_gym_config(config):
    """Load configs/<task_name>/<setting>/gym_config.json."""
    task_name = config.get("task_name")
    setting = config.get("setting")
    if not task_name or not setting:
        print("Error: task_name and setting are required to load gym_config")
        sys.exit(1)

    with open(f"configs/{task_name}/{setting}/gym_config.json", "r") as f:
        return json.load(f)

def find_action_config(config):
    """Load the task action_config.json, falling back to the setting folder."""
    task_name = config.get("task_name")
    setting = config.get("setting")
    if not task_name:
        print("Error: task_name is required to load action_config")
        sys.exit(1)

    action_cfg_path = f"configs/{task_name}/action_config.json"
    if not os.path.exists(action_cfg_path) and setting:
        action_cfg_path = f"configs/{task_name}/{setting}/action_config.json"

    with open(action_cfg_path, "r") as f:
        return json.load(f)

def _instruction_to_text(instruction):
    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, dict):
        for key in ("lang", "text", "prompt"):
            value = instruction.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _find_instruction_recursive(value):
    if isinstance(value, dict):
        text = _instruction_to_text(value.get("instruction"))
        if text:
            return text
        for child in value.values():
            text = _find_instruction_recursive(child)
            if text:
                return text
    elif isinstance(value, list):
        for child in value:
            text = _find_instruction_recursive(child)
            if text:
                return text
    return None


def extract_instruction_from_gym_config(gym_config):
    """Return the task language instruction declared in gym_config.json."""
    dataset_cfg = gym_config.get("env", {}).get("dataset", {})
    if isinstance(dataset_cfg, dict):
        for functor_cfg in dataset_cfg.values():
            if not isinstance(functor_cfg, dict):
                continue
            params = functor_cfg.get("params", {})
            text = _instruction_to_text(params.get("instruction"))
            if text:
                return text

    return _find_instruction_recursive(gym_config)


def _read_cpu_model():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def collect_platform_metadata():
    """Describe the hardware/software platform that gives timing its context."""
    accelerators = []
    if torch.cuda.is_available():
        accelerators = [
            torch.cuda.get_device_name(device_index)
            for device_index in range(torch.cuda.device_count())
        ]
    return {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "cpu": _read_cpu_model(),
        "accelerators": accelerators,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def format_platform(platform_metadata):
    accelerators = platform_metadata["accelerators"]
    accelerator_text = ", ".join(accelerators) if accelerators else "none detected"
    return (
        f"GPU={accelerator_text}; CPU={platform_metadata['cpu']}; "
        f"OS={platform_metadata['operating_system']}; "
        f"PyTorch={platform_metadata['torch_version']}; "
        f"CUDA={platform_metadata['torch_cuda_version']}"
    )


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="RoboSynChallenge Unified Policy Evaluator")

    # Required: config file
    parser.add_argument("--config", type=str, required=True,
                        help="Path to policy YAML config (e.g. policy/pi0/deploy_policy.yml)")

    # Override any config value from command line
    parser.add_argument("--overrides", nargs=argparse.REMAINDER,
                        help="Override config values, e.g. --train_config_name my_config")

    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    if args.overrides:
        for i in range(0, len(args.overrides), 2):
            key = args.overrides[i].lstrip("-")
            value = args.overrides[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            config[key] = value

    return config


def main():
    config = parse_args_and_config()
    select_cuda_device(config)

    policy_name = config.get("policy_name")
    if not policy_name:
        print("Error: 'policy_name' is required in config")
        sys.exit(1)

    task_name = config.get("task_name")
    max_episodes = config.get("max_episodes")
    seed = config.get("seed")
    fixed_episode_seed = config.get("eval_fixed_episode_seed")
    headless = config.get("headless")

    # Load policy adapter
    print(f"Loading policy: {policy_name}")
    policy_pkg = load_policy_adapter(policy_name)

    # Resolve gym and action configs
    gym_config_dict = find_gym_config(config)
    action_config_dict = find_action_config(config)
    max_env_steps, deploy_max_steps, gym_max_steps = resolve_episode_max_steps(
        config, gym_config_dict
    )

    # Parallel evaluation (wave-batched episodes) is opt-in via num_envs > 1.
    num_envs = int(config.get("num_envs", 1) or 1)
    if num_envs > 1 and config.get("rollout_save"):
        print(
            "Error: rollout_save is incompatible with num_envs > 1 — the LeRobot "
            "recorder's rollout buffer assumes lockstep single-env episodes."
        )
        sys.exit(1)
    if num_envs > 1 and config.get("eval_video_log"):
        print(
            "Warning: eval_video_log is not supported with num_envs > 1; "
            "disabling video recording for this run."
        )
        config["eval_video_log"] = False

    # Optionally record rollouts (with success labels) for sim-RECAP training.
    # Must run before make_env_from_configs, which deep-copies gym_config_dict.
    rollout_dir = configure_rollout_saving(config, gym_config_dict)

    # Build environment
    env_id = gym_config_dict.get("id")
    print(f"Creating environment: {env_id}")
    env, gym_config_dict = make_env_from_configs(config, gym_config_dict, action_config_dict)
    instruction = extract_instruction_from_gym_config(gym_config_dict)
    if instruction:
        env._current_instruction = instruction
        print(f"Using instruction from gym_config: {instruction}")
    run_dir = create_eval_run_dir(config)
    result_path = run_dir / "evaluation_metrics.json"
    video_recorder = create_video_recorder(config, run_dir=run_dir)
    reset_sync_steps = int(config.get("eval_reset_sync_steps", 0))
    if num_envs > 1:
        sim_device = str(gym_config_dict.get("device", "cpu"))
        if not sim_device.startswith("cuda"):
            print(
                f"Warning: num_envs={num_envs} with device={sim_device!r} — DexSim "
                "falls back to per-env Python loops on CPU; pass --device cuda "
                "for real parallel speedup."
            )
        eval_env = ParallelEvalProxy(env, num_envs)
        print(f"Parallel evaluation enabled: num_envs={num_envs} (wave-batched episodes)")
    else:
        eval_env = (
            RecordingEnvProxy(env, video_recorder, reset_sync_steps=reset_sync_steps)
            if video_recorder or reset_sync_steps > 0
            else env
        )
    if video_recorder:
        print(f"Recording eval videos to: {video_recorder.save_dir}")

    # Create model
    print("Creating model...")
    model = policy_pkg.get_model(config)
    print("Model created successfully.")

    # Evaluation loop
    platform_metadata = collect_platform_metadata()
    rng = np.random.RandomState(seed)
    # 分片:每个进程只真正执行属于自己的那些 episode,但 rng 照常逐个抽取,
    # 保证各分片拿到的种子和「单进程跑满 max_episodes」时完全一致,合并后就是同一批集。
    num_shards = int(config.get("num_shards", 1) or 1)
    shard_index = int(config.get("shard_index", 0) or 0)
    episodes_run = 0
    success_count = 0
    episode_records = []
    loop_completed = False
    action_steps = []
    all_inference_times = []
    episode_inference_totals = []
    print(f"\n{'='*25} Starting Evaluation {'='*25}\n")
    print(f"  Policy: {policy_name}  |  Task: {task_name}")
    print(f"  Episodes: {max_episodes}  |  Seed: {seed}")
    if num_shards > 1:
        print(f"  Shard: {shard_index}/{num_shards} (本进程只跑 episode % {num_shards} == {shard_index})")
    if num_envs > 1:
        print(f"  Parallel: num_envs={num_envs}, wave-batched episodes, per-slot seeded resets")
    print(
        f"  Max env steps: {max_env_steps} "
        f"(deploy_config.max_steps={deploy_max_steps}, "
        f"gym_config.max_episode_steps={gym_max_steps})"
    )
    print(f"  Timing platform: {format_platform(platform_metadata)}")
    print(f"  Metrics file: {result_path}")
    print(f"{'='*70}\n")

    try:
        if num_envs > 1:
            (
                episodes_run,
                success_count,
                episode_records,
                action_steps,
                all_inference_times,
                episode_inference_totals,
            ) = run_parallel_episodes(
                env,
                eval_env,
                policy_pkg,
                model,
                config,
                rng,
                max_episodes,
                max_env_steps,
                num_envs,
                num_shards,
                shard_index,
                fixed_episode_seed,
                reset_sync_steps,
            )
        # 并行模式走上面的 wave 循环;下面的串行 episode 循环保持原样不动。
        sequential_episodes = range(max_episodes) if num_envs == 1 else ()
        for episode in sequential_episodes:
            ep_seed = (
                int(fixed_episode_seed)
                if fixed_episode_seed is not None
                else int(rng.randint(0, 2**31 - 1))
            )
            if num_shards > 1 and episode % num_shards != shard_index:
                continue
            episodes_run += 1
            if video_recorder:
                video_recorder.start_episode(episode, ep_seed)

            episode_success = False
            inference_times = []
            env_steps = 0
            progress_bar = None
            try:
                obs, info = eval_env.reset(seed=ep_seed)
                policy_pkg.reset_model(model)
                progress_bar = tqdm(
                    total=max_env_steps,
                    desc=f"Episode {episode + 1:03d}/{max_episodes:03d}",
                    unit="step",
                    dynamic_ncols=True,
                    leave=False,
                )

                while env_steps < max_env_steps:
                    # Upstream's timing change made ``eval`` return a 4-tuple,
                    # but adapters that predate it (including upstream's own
                    # smolvla) still return ``(obs, info, truncated)``.  Accept
                    # both so an untimed adapter degrades instead of crashing.
                    eval_result = policy_pkg.eval(eval_env, model, obs)
                    if len(eval_result) == 4:
                        obs, info, truncated, new_times = eval_result
                    else:
                        obs, info, truncated = eval_result
                        new_times = []
                    inference_times.extend(new_times)
                    env_steps = int(info["elapsed_steps"].item())

                    progress_bar.update(
                        max(0, min(env_steps, max_env_steps) - progress_bar.n)
                    )
                    progress_bar.set_postfix(
                        inference=len(inference_times),
                        infer_time=f"{new_times[-1]:.3f}s" if new_times else "cached",
                    )

                    is_truncated = RecordingEnvProxy._as_done(truncated)
                    if not is_truncated and env.get_wrapper_attr("is_task_success")():
                        episode_success = True
                        progress_bar.write("Task success!")
                        break
                    if is_truncated:
                        progress_bar.write("Task timeout!")
                        break
            finally:
                if progress_bar:
                    progress_bar.close()
                if video_recorder:
                    video_recorder.close_episode(success=episode_success)

            success_count += int(episode_success)
            effective_steps = env_steps if episode_success else max_env_steps
            action_steps.append(effective_steps)
            all_inference_times.extend(inference_times)
            episode_inference_totals.append(sum(inference_times))
            episode_average = float(np.mean(inference_times)) if inference_times else 0
            status = "\033[92mSUCCESS\033[0m" if episode_success else "\033[91mFAIL\033[0m"
            print(
                f"  Episode {episode+1}: {status}; action_steps="
                f"{effective_steps}/{max_env_steps}; inference="
                f"{episode_average:.6f}s "
                f"over {len(inference_times)} calls"
            )
            print(f"  [{episode+1:3d}/{max_episodes}] {status}  "
                  f"(success rate: {success_count}/{episode+1} = {100*success_count/(episode+1):.1f}%)")

            episode_metrics = summarize_task_metrics(locals().get("info"))
            if episode_metrics:
                print(f"  metrics: {format_task_metrics(episode_metrics)}")

            if rollout_dir is not None:
                episode_records.append(
                    {
                        "episode_index": len(episode_records),
                        "seed": ep_seed,
                        "success": bool(episode_success),
                        "env_steps": int(env_steps),
                        "metrics": episode_metrics,
                    }
                )
        loop_completed = True
    finally:
        if video_recorder:
            video_recorder.close_episode(success=False)
        if rollout_dir is not None:
            # The recorder writes an episode on the *next* reset, so the final
            # episode is still buffered here. One extra reset flushes it; if
            # the loop crashed mid-episode the buffer holds an unlabeled
            # partial episode instead, which we discard (save_data=False) to
            # keep dataset episodes aligned with the success sidecar.
            try:
                env.reset(options={"save_data": loop_completed})
            except Exception as flush_err:  # noqa: BLE001
                print(f"Warning: rollout flush reset failed: {flush_err}")
            # 记录器会在 save_path 下再建 <robot>_<scene>_<task>_NNN 子目录,
            # 边车必须写进真正的数据集目录。
            dataset_dir = get_recorder_dataset_dir(env) or rollout_dir
            write_rollout_success_sidecar(
                dataset_dir, config, episode_records, get_saved_episode_count(env)
            )
        # 不在这里 env.close():本机 DexSim 栈 close 会直接终止进程(exit 0),
        # 放在 finally 里会把后面的 summary 打印和 evaluation_metrics.json 全吞掉
        # (eval_result/ 里长期没有 metrics 文件就是这个原因)。close 挪到 main
        # 末尾、指标落盘之后;异常路径不 close,进程退出时由 OS 回收。

    inference_call_count = len(all_inference_times)
    average_action_steps = float(np.mean(action_steps))
    summary = {
        "episode_count": len(action_steps),
        "success_count": success_count,
        "success_rate": success_count / len(action_steps),
        "average_action_steps": average_action_steps,
        "average_action_steps_ratio": average_action_steps / max_env_steps,
        "inference_call_count": inference_call_count,
        "average_inference_calls_per_episode": inference_call_count / len(action_steps),
        "average_inference_time_seconds": (
            float(np.mean(all_inference_times)) if all_inference_times else None
        ),
        "average_inference_time_per_episode_seconds": float(
            np.mean(episode_inference_totals)
        ),
    }
    result_payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "config": {
            "policy": policy_name,
            "task": task_name,
            "setting": config.get("setting"),
            "checkpoint_path": config.get("checkpoint_path"),
            "dp_num_inference_steps": config.get("dp_num_inference_steps"),
            "episode_count": episodes_run,
            "episode_count_full": max_episodes,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "num_envs": num_envs,
            "timeout_action_steps": max_env_steps,
            "seed": seed,
        },
        "inference_timing_scope": (
            "raw observation preprocessing and transfer through executable action; "
            "env.step excluded"
            + (
                f"; batched calls each serving num_envs={num_envs} episodes — "
                "not comparable with single-env latency"
                if num_envs > 1
                else ""
            )
        ),
        "platform": platform_metadata,
        "summary": summary,
    }
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(result_payload, result_file, indent=2, ensure_ascii=False)
        result_file.write("\n")

    average_inference_time = summary["average_inference_time_seconds"]
    print(f"\n{'='*50}")
    print(
        f"  Evaluation Results Summary: {success_count}/{episodes_run} "
        f"({100*success_count/max(episodes_run,1):.1f}%)"
    )
    print(
        f"  Average action steps: {summary['average_action_steps']:.2f}/"
        f"{max_env_steps} ({100*summary['average_action_steps_ratio']:.2f}%; "
        f"failed episodes count as {max_env_steps} steps)"
    )
    if average_inference_time is None:
        print("  Average inference time: n/a (no inference calls recorded)")
    else:
        print(
            f"  Average inference latency: {average_inference_time:.6f}s "
            f"({1000*average_inference_time:.3f}ms) over "
            f"{summary['inference_call_count']} model calls"
        )
        print(
            "  Average total inference time per episode: "
            f"{summary['average_inference_time_per_episode_seconds']:.6f}s "
            f"over {summary['average_inference_calls_per_episode']:.2f} model calls"
        )
    print(f"  Timing platform: {format_platform(platform_metadata)}")
    print(f"  Metrics saved to: {result_path}")
    print(f"{'='*50}")

    # close 可能直接终止进程(见 finally 处注释),必须最后做,且先把 stdout 刷掉。
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
