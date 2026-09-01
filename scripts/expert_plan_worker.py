# ----------------------------------------------------------------------------
# RoboSynChallenge — single-env expert planning worker for parallel collection
#
# 并行采集的规划侧进程:批量环境负责摆场景/执行/录制,本 worker 在自己的
# num_envs=1 环境里用同一个 seed 复现同一场景(物体位姿跨进程逐位一致,已实测),
# 然后走完全原生的单环境专家规划链路(affordance 注册 / action bank / Toppra),
# 把轨迹和该 seed 的官方初始 qpos 发回驱动层。
#
# 之所以不在批量环境里直接规划:action bank 的 affordance 管线
# (_prepare_warpping)和 FK/IK 校验都是 batch==num_envs 的强约束,逐 env 规划
# 在 num_envs>1 下从节点生成到规划器一路报错;而单环境 worker 逐字节复用
# 官方采集代码路径,行为天然与串行采集一致。
#
# 协议(stdin/stdout, line-delimited JSON,响应带 "rsc_plan_worker" 哨兵键;
# EmbodiChain 的日志会混在 stdout 里,驱动层按哨兵过滤):
#   {"cmd": "plan", "seed": 123}
#     -> {"rsc_plan_worker": true, "ok": true, "npz": "/path.npz", "T": 340}
#        npz 内含 actions (T, dof_active) 与 init_qpos (dof_full,)
#     -> {"rsc_plan_worker": true, "ok": false, "error": "..."}   规划失败(换种子重试)
#   {"cmd": "quit"} -> 进程退出
# ----------------------------------------------------------------------------

import argparse
import json
import os
import sys
import tempfile
import traceback

# 以脚本方式启动时 sys.path[0] 是 scripts/,editable 安装会把 robosynchallenge
# 解析到 main worktree——强制用本 worktree 的代码。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym

import robosynchallenge  # noqa: F401  (注册任务环境)
import embodichain.lab.gym.utils.gym_utils as gym_utils
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)

gym_utils.DEFAULT_MANAGER_MODULES = gym_utils.DEFAULT_MANAGER_MODULES + [
    "robosynchallenge.managers.actions",
    "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events",
    "robosynchallenge.managers.observations",
]


def _patch_motion_generator_to_cpu():
    """任务 action bank 里的 `ret.positions[0].numpy()` 假设结果在 cpu 上。

    日常串行采集跑 cpu 设备,这个假设从未被打破;worker 为了与 cuda 批量环境
    保持 RNG 同流必须跑 cuda,规划结果就落在 cuda 上。tasks/ 目录有「与官方
    逐字节一致」的红线不能改 bank,这里在规划器出口统一把结果搬回 cpu。
    """
    from embodichain.lab.sim.planners import MotionGenerator

    original_generate = MotionGenerator.generate

    def generate_on_cpu(self, *args, **kwargs):
        ret = original_generate(self, *args, **kwargs)
        positions = getattr(ret, "positions", None)
        if torch.is_tensor(positions) and positions.is_cuda:
            ret.positions = positions.cpu()
        elif isinstance(positions, (list, tuple)):
            moved = [
                p.cpu() if torch.is_tensor(p) and p.is_cuda else p for p in positions
            ]
            ret.positions = type(positions)(moved)
        return ret

    MotionGenerator.generate = generate_on_cpu


_patch_motion_generator_to_cpu()


def _respond(payload):
    payload["rsc_plan_worker"] = True
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _plan_once(env, seed, work_dir):
    # 与 SeededCollection.reset 相同的播种口径(np + torch,reset 内部还会再
    # torch.manual_seed 一次),保证场景与串行种子化采集逐位一致。
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    action_list = env.get_wrapper_attr("create_demo_action_list")(action_sentence=0)
    if action_list is None or len(action_list) == 0:
        return {"ok": False, "error": "invalid_plan"}

    actions = (
        action_list
        if torch.is_tensor(action_list)
        else torch.as_tensor(np.asarray(action_list))
    )
    # (T, num_envs=1, dof) -> (T, dof);规划结果永远写在第 0 列。
    actions_np = actions[:, 0, :].detach().cpu().numpy().astype(np.float32)
    init_qpos = (
        env.unwrapped.robot.get_qpos()[0].detach().cpu().numpy().astype(np.float32)
    )

    fd, npz_path = tempfile.mkstemp(prefix="plan_", suffix=".npz", dir=work_dir)
    os.close(fd)
    np.savez(npz_path, actions=actions_np, init_qpos=init_qpos)
    return {"ok": True, "npz": npz_path, "T": int(actions_np.shape[0])}


def main():
    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--work_dir", type=str, required=True)
    args = parser.parse_args()

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    env_cfg.num_envs = 1
    # worker 只做规划,不落数据,也不需要 rollout buffer。
    env_cfg.filter_dataset_saving = True

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    _respond({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = request.get("cmd")
        if cmd == "quit":
            break
        if cmd != "plan":
            _respond({"ok": False, "error": f"unknown cmd: {cmd!r}"})
            continue
        try:
            _respond(_plan_once(env, int(request["seed"]), args.work_dir))
        except Exception as exc:  # noqa: BLE001 - 单次规划失败不拖垮 worker
            traceback.print_exc()
            _respond({"ok": False, "error": repr(exc)[:500]})

    # 不调 env.close():本机 DexSim 栈 close 直接终止进程;这里本来就要退出。
    os._exit(0)


if __name__ == "__main__":
    main()
