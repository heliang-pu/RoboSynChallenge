# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Script to run the environment."""

import argparse
import sys
import traceback
import torch
import numpy as np
import tqdm

import gymnasium as gym
import robosynchallenge
from robosynchallenge.replay import replay_trajectory

from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
import embodichain.lab.gym.utils.gym_utils as gym_utils
from embodichain.lab.scripts.run_env import generate_and_execute_action_list, preview
from embodichain.utils.logger import log_warning, log_info

gym_utils.DEFAULT_MANAGER_MODULES = gym_utils.DEFAULT_MANAGER_MODULES + [
    "robosynchallenge.managers.actions",
    "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events",
    "robosynchallenge.managers.observations",
]


def _patch_motion_generator_to_cpu():
    """任务 action bank 的 `ret.positions[0].numpy()` 假设规划结果在 cpu 上。

    cpu 设备(日常口径)无感;cuda 设备下不打这个补丁 bank 直接 TypeError。
    tasks/ 目录有「与官方逐字节一致」红线,不改 bank,在规划器出口统一搬运。
    与 scripts/expert_plan_worker.py 中的同名补丁保持一致。
    """
    import torch as _torch
    from embodichain.lab.sim.planners import MotionGenerator

    original_generate = MotionGenerator.generate

    def generate_on_cpu(self, *args, **kwargs):
        ret = original_generate(self, *args, **kwargs)
        positions = getattr(ret, "positions", None)
        if _torch.is_tensor(positions) and positions.is_cuda:
            ret.positions = positions.cpu()
        elif isinstance(positions, (list, tuple)):
            ret.positions = type(positions)(
                p.cpu() if _torch.is_tensor(p) and p.is_cuda else p for p in positions
            )
        return ret

    MotionGenerator.generate = generate_on_cpu


_patch_motion_generator_to_cpu()


def _report_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").tolist()
    if isinstance(value, dict):
        return {key: _report_value(item) for key, item in value.items()}
    return value


def _generate_function(
    env,
    num_traj,
    time_id: int = 0,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    reset_first: bool = True,
    report_task_success: bool = False,
    **kwargs,
) -> bool:
    valid = True
    if reset_first:
        _, _ = env.reset()

    max_invalid_streak = int(kwargs.pop("max_invalid_streak", 0) or 0)
    invalid_counter = kwargs.pop("invalid_counter", None)
    invalid_streak = 0
    while True:
        for trajectory_idx in range(num_traj):
            valid = generate_and_execute_action_list(
                env, trajectory_idx, debug_mode, **kwargs
            )

            if valid and report_task_success:
                success = env.unwrapped.is_task_success().detach().to(device="cpu")
                log_info(
                    f"Success condition after expert rollout: {success.tolist()}",
                    color="green" if bool(success.all()) else "yellow",
                )
                metrics = getattr(env.unwrapped, "_last_success_metrics", None)
                evaluator = getattr(env.unwrapped, "_evaluate_task_state", None)
                if metrics is None and evaluator is not None:
                    evaluated = evaluator()
                    if isinstance(evaluated, tuple) and len(evaluated) >= 3:
                        metrics = evaluated[2]
                if metrics:
                    log_info(f"Success metrics: {_report_value(metrics)}")

            if not valid:
                _, _ = env.reset(options={"save_data": False})
                break

        if valid:
            break

        invalid_streak += 1
        if invalid_counter is not None:
            invalid_counter[0] += 1
        if max_invalid_streak and invalid_streak >= max_invalid_streak:
            log_warning(
                f"{invalid_streak} consecutive invalid generations; region looks "
                "kinematically infeasible for the scripted expert."
            )
            return False

        log_warning("Reset valid flag to True.")
        valid = True

    return True


def _get_saved_episode_count(env):
    try:
        dataset_manager = env.get_wrapper_attr("dataset_manager")
    except AttributeError:
        dataset_manager = getattr(
            getattr(env, "unwrapped", env), "dataset_manager", None
        )

    if dataset_manager is None:
        return None

    episode_counts = []
    for mode_cfgs in getattr(dataset_manager, "_mode_functor_cfgs", {}).values():
        for functor_cfg in mode_cfgs:
            functor = getattr(functor_cfg, "func", None)
            if hasattr(functor, "curr_episode"):
                episode_counts.append(int(functor.curr_episode))

    if not episode_counts:
        return None

    return max(episode_counts)


def _generate_until_saved_episode_target(args, env, gym_config, num_traj: int) -> bool:
    target_episodes = int(gym_config.get("max_episodes", 1))
    saved_episodes = _get_saved_episode_count(env)

    if saved_episodes is None:
        return False

    log_info(
        f"Collecting until {target_episodes} successful episodes are saved.",
        color="green",
    )

    _, _ = env.reset()
    saved_episodes = _get_saved_episode_count(env)
    attempt = 0
    progress = tqdm.tqdm(
        total=target_episodes,
        initial=min(saved_episodes, target_episodes),
        desc="Saved successful episodes",
        unit="episode",
    )

    max_attempts = int(getattr(args, "max_generation_attempts", 0) or 0)
    invalid_total = [0]
    while saved_episodes < target_episodes:
        if max_attempts and saved_episodes == 0 and invalid_total[0] >= 300:
            log_warning(
                f"{invalid_total[0]} total invalid generations with zero successes; "
                "region infeasible for the scripted expert."
            )
            sys.exit(3)
        zero_cap = min(60, max_attempts) if max_attempts else 0
        if zero_cap and saved_episodes == 0 and attempt >= zero_cap:
            log_warning(f"No successful episode in the first {attempt} attempts; giving up this region.")
            sys.exit(3)
        if max_attempts and attempt >= max_attempts:
            log_warning(
                f"Giving up after {attempt} attempts with "
                f"{saved_episodes}/{target_episodes} episodes saved."
            )
            sys.exit(3)
        attempt += 1
        previous_saved_episodes = saved_episodes

        hopeful = _generate_function(
            env,
            num_traj,
            attempt - 1,
            save_path=getattr(args, "save_path", ""),
            save_video=getattr(args, "save_video", False),
            debug_mode=getattr(args, "debug_mode", False),
            reset_first=False,
            regenerate=getattr(args, "regenerate", False),
            report_task_success=getattr(args, "report_task_success", False),
            max_invalid_streak=(100 if max_attempts else 0),
            invalid_counter=invalid_total,
        )
        if not hopeful:
            log_warning("Aborting this region (planner cannot reach it).")
            sys.exit(3)

        saved_before_reset = _get_saved_episode_count(env)
        save_this_episode = saved_before_reset < target_episodes
        settle = int(getattr(args, "success_settle_steps", 0) or 0)
        if save_this_episode and settle > 0:
            # 自适应静置:官方判定要求连续 success_stable_steps 步稳定,而边缘插入在
            # 松爪后仍会滑落/减速一段时间。每 5 步查一次判定,成功即停,最多 settle 步。
            try:
                robot = env.unwrapped.robot
                hold = robot.get_qpos()[:, env.unwrapped.active_joint_ids]
                stepped = 0
                while stepped < settle:
                    for _ in range(min(5, settle - stepped)):
                        env.step(hold)
                        stepped += 1
                    if bool(env.unwrapped.is_task_success().detach().reshape(-1)[0].item()):
                        break
            except Exception as exc:
                log_warning(f"Settle steps failed ({exc}); judging without settle.")
        if save_this_episode and getattr(args, "save_only_success", False):
            official_success = bool(
                env.unwrapped.is_task_success().detach().reshape(-1)[0].item()
            )
            if not official_success:
                try:
                    _, _, m = env.unwrapped._evaluate_task_state()
                    detail = {k: _report_value(m[k]) for k in (
                        "placement_ok_single_frame", "place_stable_count", "cube_xy_dist",
                        "cube_vertical_angle", "cube_lin_vel_norm", "cube_to_left_eef_dist",
                        "cube_to_right_eef_dist", "bottom_z_diff") if k in m}
                    log_warning(f"Episode failed the official success check; discarding. post-settle: {detail}")
                except Exception:
                    log_warning("Episode failed the official success check; discarding.")
                save_this_episode = False
        _, _ = env.reset(
            options={"save_data": save_this_episode}
        )
        saved_episodes = _get_saved_episode_count(env)

        progress.update(max(0, min(saved_episodes, target_episodes) - progress.n))

        if saved_episodes == previous_saved_episodes:
            log_warning(
                f"Attempt {attempt} did not save a successful episode "
                f"({saved_episodes}/{target_episodes}). Retrying."
            )
        else:
            log_info(
                f"Saved successful episodes: {saved_episodes}/{target_episodes} "
                f"after {attempt} attempts.",
                color="green",
            )

    progress.close()
    return True


def run_env_main(args, env, gym_config):
    if int(getattr(args, "num_envs", 1) or 1) > 1:
        if getattr(args, "replay", False) or getattr(args, "preview", False):
            log_warning("--num_envs > 1 与 --replay/--preview 互斥。")
            sys.exit(2)
        from scripts.parallel_collection import run_parallel_collection

        return run_parallel_collection(args, env, gym_config)

    if getattr(args, "replay", False):
        replay_trajectory(
            env,
            args.replay_trajectory,
            mode=getattr(args, "replay_mode", "kinematic"),
        )
        return

    if getattr(args, "preview", False):
        log_info(
            "Preview mode enabled. Launching environment preview...", color="green"
        )
        preview(env)

    log_info("Start offline data generation.", color="green")
    num_traj = 1

    if _generate_until_saved_episode_target(args, env, gym_config, num_traj):
        return

    log_warning(
        "No dataset recorder was found. Falling back to max_episodes generation attempts."
    )
    for i in range(gym_config.get("max_episodes", 1)):
        _generate_function(
            env,
            num_traj,
            i,
            save_path=getattr(args, "save_path", ""),
            save_video=getattr(args, "save_video", False),
            debug_mode=getattr(args, "debug_mode", False),
            regenerate=getattr(args, "regenerate", False),
            report_task_success=getattr(args, "report_task_success", False),
        )

    if not getattr(args, "report_task_success", False):
        _, _ = env.reset()


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()

    add_env_launcher_args_to_parser(parser)

    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay a native EmbodiChain or legacy state/action trajectory.",
    )
    parser.add_argument(
        "--replay_trajectory",
        type=str,
        default=None,
        help="Path to the .pt trajectory file to replay.",
    )
    parser.add_argument(
        "--replay_mode",
        choices=["kinematic", "dynamic", "control"],
        default="kinematic",
        help="Replay mode (default: kinematic).",
    )
    parser.add_argument(
        "--report_task_success",
        action="store_true",
        help="Log the task success predicate after each expert rollout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for reproducible collection: every episode gets a "
        "derived seed that is injected into env.reset() and recorded (with the "
        "official success verdict, config hash and git commit) in an "
        "episode_success.json sidecar next to the recorded dataset.",
    )
    parser.add_argument(
        "--save_only_success",
        action="store_true",
        help="Only save episodes that pass the official task-success check, so "
        "max_episodes counts successful episodes.",
    )
    parser.add_argument(
        "--max_generation_attempts",
        type=int,
        default=0,
        help="Give up (exit 3) if this many generation attempts pass without "
        "reaching the episode target. 0 = unlimited (official behaviour). "
        "Guards against kinematically infeasible randomization regions.",
    )
    parser.add_argument(
        "--success_settle_steps",
        type=int,
        default=0,
        help="Extra hold-position env steps after the expert action list ends, "
        "before the success check. The official evaluator needs the placement "
        "to hold for success_stable_steps consecutive steps, but expert scripts "
        "end at the instant of release; a short settle (like the trailing steps "
        "of an evaluation episode) lets the stability counter accumulate and "
        "lets the object physically settle. 0 keeps official behaviour.",
    )

    args = parser.parse_args()

    if args.replay and not args.replay_trajectory:
        parser.error("--replay requires --replay_trajectory <path>.")
    if args.replay and args.preview:
        parser.error("--replay and --preview are mutually exclusive.")
    if args.replay:
        args.filter_dataset_saving = True

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    physics_config = gym_config.get("physics", {})
    if "enable_ccd" in physics_config:
        env_cfg.sim_cfg.physics_config.enable_ccd = bool(
            physics_config["enable_ccd"]
        )
        if env_cfg.sim_cfg.physics_config.enable_ccd:
            log_info(
                "Scene-level continuous collision detection enabled.", color="green"
            )
    if args.max_episodes is not None:
        gym_config["max_episodes"] = args.max_episodes

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)

    if getattr(args, "seed", None) is not None and int(getattr(args, "num_envs", 1) or 1) == 1:
        # 并行模式自己管种子(每槽位显式 seed + 边车),不套 SeededCollection。
        from scripts.seeded_collection import SeededCollection

        env = SeededCollection(
            env, int(args.seed), gym_config,
            gym_config_path=getattr(args, "gym_config", None),
        )
        log_info(f"Seeded collection enabled (master seed {args.seed}).", color="green")

    try:
        run_env_main(args, env, gym_config=gym_config)
    except Exception as e:
        log_warning(f"An error occurred during environment execution: {e}")
        traceback.print_exc()
        env.close()
