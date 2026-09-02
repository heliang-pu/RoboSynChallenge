# ----------------------------------------------------------------------------
# RoboSynChallenge — wave-parallel expert data collection (num_envs > 1)
#
# 架构(与 docs/parallel_collection.md 对应):
#   批量环境(本进程):per-slot 种子化部分重置摆场景 -> 整批执行 -> per-env
#   官方判定 -> dataset_manager.apply(mode="save", env_ids=成功槽位) 直接落盘。
#   规划(worker 子进程):scripts/expert_plan_worker.py 在单环境里用同一 seed
#   复现场景并走原生专家规划链路,返回 (T, dof) 轨迹与官方初始 qpos。
#
# 关键口径:
#   * 种子协议与串行种子化采集一致:每个 episode 一个显式 seed,场景由该 seed
#     唯一决定,sidecar 逐集记录,可用串行采集单独复现任一集;
#   * worker 的初始 qpos 会回写进对应槽位(set_qpos),既保证执行起点与规划
#     一致,也中和了部分重置下机器人随机化路径与单环境不等价的上游问题
#     (物体场景跨进程逐位一致已实测;机器人 qpos 在 env_ids 部分重置路径上
#     有潜伏的分支差异);
#   * 等长执行:wave 内轨迹尾帧 padding 到共同 T,current_rollout_step 标量
#     语义因此保持正确;短轨迹 episode 尾部多出的「保持姿态」帧照常入库
#     (与 --success_settle_steps 的静置帧同性质);
#   * 落盘不走 reset(save_data=True):部分重置会把共享写指针清零,第二个槽位
#     就会存出空集。改为在 reset 之前对成功槽位直接 apply(mode="save")。
# ----------------------------------------------------------------------------

import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import tqdm

from embodichain.utils.logger import log_info, log_warning


class ExpertPlanWorker:
    """Line-delimited JSON RPC to the single-env planning subprocess."""

    def __init__(self, args, work_dir):
        worker_path = Path(__file__).resolve().parent / "expert_plan_worker.py"
        cmd = [
            sys.executable,
            str(worker_path),
            "--gym_config",
            args.gym_config,
            "--work_dir",
            str(work_dir),
            "--device",
            str(getattr(args, "device", "cpu")),
            "--gpu_id",
            str(getattr(args, "gpu_id", 0)),
            "--headless",
        ]
        if getattr(args, "action_config", None):
            cmd += ["--action_config", args.action_config]
        if getattr(args, "renderer", None):
            cmd += ["--renderer", str(args.renderer)]
        if getattr(args, "filter_visual_rand", False):
            cmd += ["--filter_visual_rand"]
        if getattr(args, "arena_space", None) is not None:
            cmd += ["--arena_space", str(args.arena_space)]

        log_info("Starting expert plan worker (single-env planner)...", color="green")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)
        ready = self._wait_response()
        if not ready.get("ready"):
            raise RuntimeError(f"expert plan worker bad handshake: {ready}")
        log_info("Expert plan worker ready.", color="green")

    def _wait_response(self):
        # worker 的 stdout 混着 EmbodiChain 日志,只认带哨兵键的 JSON 行。
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("rsc_plan_worker"):
                return msg
        raise RuntimeError(
            "expert plan worker exited unexpectedly "
            f"(returncode={self.proc.poll()})"
        )

    def plan(self, seed):
        """Returns (actions (T, dof) float32, init_qpos (dof_full,)) or None."""
        self.proc.stdin.write(json.dumps({"cmd": "plan", "seed": int(seed)}) + "\n")
        self.proc.stdin.flush()
        msg = self._wait_response()
        if not msg.get("ok"):
            return None
        npz_path = msg["npz"]
        try:
            data = np.load(npz_path)
            return np.asarray(data["actions"]), np.asarray(data["init_qpos"])
        finally:
            if os.path.exists(npz_path):
                os.remove(npz_path)

    def close(self):
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def _get_saved_episode_count(env):
    """recorder 的全局已存集数(与 scripts/run_env_seeded.py 同款,复制避免循环导入)。"""
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


def _get_recorder_dataset_dir(env):
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


def _write_sidecar(dataset_dir, args, gym_config, num_envs, records):
    payload = {
        "collection_mode": "parallel_wave",
        "master_seed": int(args.seed),
        "num_envs": int(num_envs),
        "gym_config": getattr(args, "gym_config", None),
        "task_id": gym_config.get("id"),
        "saved_episode_count": len(records),
        "labels_field": "episode_success",
        "episodes": records,
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = dataset_dir / "episode_success.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(out)
    log_info(f"[parallel_collection] 边车已写: {out} ({len(records)} 集)", color="green")


def run_parallel_collection(args, env, gym_config):
    raw = env.unwrapped
    num_envs = int(raw.num_envs)
    device = raw.device

    if getattr(args, "seed", None) is None:
        log_warning("并行采集必须显式给 --seed(逐集种子与可复现性都由它派生)。")
        sys.exit(2)
    if raw.dataset_manager is None:
        log_warning(
            "并行采集需要 dataset recorder;去掉 --filter_dataset_saving 或检查 "
            "gym_config 的 dataset functor。"
        )
        sys.exit(2)

    target = int(gym_config.get("max_episodes", 1))
    max_env_steps = int(gym_config.get("max_episode_steps") or raw.max_episode_steps)
    settle_budget = int(getattr(args, "success_settle_steps", 0) or 0)
    max_attempts = int(getattr(args, "max_generation_attempts", 0) or 0)
    rng = np.random.RandomState(int(args.seed))

    work_dir = tempfile.mkdtemp(prefix="rsc_plan_worker_")
    worker = ExpertPlanWorker(args, work_dir)

    records = []
    attempts = 0
    invalid_total = 0
    saved = _get_saved_episode_count(env) or 0
    initial_saved = saved
    log_info(
        f"Parallel collection: num_envs={num_envs}, target={target} episodes, "
        f"master_seed={args.seed}, settle={settle_budget}",
        color="green",
    )
    progress = tqdm.tqdm(
        total=target,
        initial=min(saved, target),
        desc="Saved successful episodes",
        unit="episode",
    )

    wave_index = 0
    while saved < target:
        wave_index += 1

        # ---- 1) 逐槽位:种子化部分重置摆场景 + worker 规划(失败换种子) ----
        plans = [None] * num_envs  # slot -> (seed, actions, init_qpos)
        for slot in range(num_envs):
            slot_failures = 0
            while True:
                if slot_failures >= 12:
                    # 换 12 个种子还不行,基本不是「这个场景不可达」而是系统性
                    # 故障(worker 报错/环境配置问题),别再烧种子了。
                    log_warning(
                        f"slot {slot} failed 12 consecutive plans — systematic "
                        "planner failure, see worker traceback above."
                    )
                    sys.exit(3)
                attempts += 1
                if max_attempts and attempts > max_attempts:
                    log_warning(
                        f"Giving up after {attempts - 1} attempts with "
                        f"{saved}/{target} episodes saved."
                    )
                    sys.exit(3)
                if saved == initial_saved and invalid_total >= 300:
                    log_warning(
                        f"{invalid_total} total invalid generations with zero "
                        "successes; region infeasible for the scripted expert."
                    )
                    sys.exit(3)

                ep_seed = int(rng.randint(0, 2**31 - 1))
                env.reset(
                    seed=ep_seed,
                    options={
                        "reset_ids": torch.tensor(
                            [slot], dtype=torch.int32, device=device
                        ),
                        "save_data": False,
                    },
                )
                result = worker.plan(ep_seed)
                if result is not None:
                    plans[slot] = (ep_seed, result[0], result[1])
                    break
                slot_failures += 1
                invalid_total += 1
                log_warning(
                    f"[wave {wave_index}] slot {slot} seed {ep_seed}: plan invalid, "
                    f"redrawing (invalid_total={invalid_total})."
                )

        # ---- 2) 初始 qpos 回写:执行起点与规划起点一致(官方单环境口径) ----
        for slot, (_, _, init_qpos) in enumerate(plans):
            qpos = torch.as_tensor(
                init_qpos, dtype=torch.float32, device=device
            ).unsqueeze(0)
            raw.robot.set_qpos(qpos, env_ids=[slot], target=False)
            raw.robot.set_qpos(qpos, env_ids=[slot])

        # ---- 3) 组批执行:尾帧 padding 到共同 T ----
        T = max(plan[1].shape[0] for plan in plans)
        if T + settle_budget >= max_env_steps:
            log_warning(
                f"轨迹长度 T={T} + settle={settle_budget} >= "
                f"max_episode_steps={max_env_steps}:会触发底层截断 auto-reset,"
                "把 gym_config 的 max_episode_steps 调大或减小 settle。"
            )
            sys.exit(2)
        dof = plans[0][1].shape[1]
        batch = np.zeros((T, num_envs, dof), dtype=np.float32)
        for slot, (_, actions, _) in enumerate(plans):
            batch[: actions.shape[0], slot] = actions
            batch[actions.shape[0] :, slot] = actions[-1]
        batch_t = torch.as_tensor(batch, device=device)

        for t in tqdm.tqdm(
            range(T),
            desc=f"Wave {wave_index} ({num_envs} envs)",
            unit="step",
            leave=False,
        ):
            env.step(batch_t[t])

        # ---- 4) 自适应静置 + per-env 官方判定 ----
        stepped = 0
        while stepped < settle_budget:
            for _ in range(min(5, settle_budget - stepped)):
                env.step(batch_t[-1])
                stepped += 1
            if bool(raw.is_task_success().detach().all()):
                break
        success_vec = raw.is_task_success().detach().reshape(-1).to("cpu")

        # ---- 5) 成功槽位直接落盘(在任何 reset 之前) ----
        success_slots = [i for i in range(num_envs) if bool(success_vec[i])]
        save_slots = success_slots[: max(0, target - saved)]
        if save_slots:
            raw.dataset_manager.apply(
                mode="save",
                env_ids=torch.tensor(save_slots, dtype=torch.long, device=device),
            )
        for slot in save_slots:
            records.append(
                {
                    "episode_index": len(records),
                    "seed": int(plans[slot][0]),
                    "slot": int(slot),
                    "success": True,
                    "env_steps": int(T + stepped),
                }
            )

        new_saved = _get_saved_episode_count(env)
        if new_saved is not None and new_saved != saved + len(save_slots):
            log_warning(
                f"recorder saved {new_saved - saved} episodes but "
                f"{len(save_slots)} were expected this wave."
            )
        saved = new_saved if new_saved is not None else saved + len(save_slots)
        progress.update(max(0, min(saved, target) - progress.n))
        log_info(
            f"[wave {wave_index}] success {len(success_slots)}/{num_envs} "
            f"(saved {saved}/{target}, attempts {attempts})",
            color="green" if success_slots else "yellow",
        )

    progress.close()
    worker.close()

    # recorder.finalize() 会把 current_rollout_step>0 的所有 env buffer 再存一遍
    # (给「最后一集没经过 reset 冲洗」的流程兜底)。我们已显式 apply(save) 过,
    # 清零写指针,finalize 只做 image writer 冲洗与元数据收尾。
    raw.current_rollout_step = 0
    try:
        raw.dataset_manager.finalize()
    except Exception as exc:  # noqa: BLE001
        log_warning(f"dataset_manager.finalize failed: {exc}")

    dataset_dir = _get_recorder_dataset_dir(env)
    if dataset_dir is not None:
        _write_sidecar(dataset_dir, args, gym_config, num_envs, records)
    else:
        log_warning("recorder dataset dir not found; sidecar not written.")
