# 并行专家数据采集（`num_envs > 1`）

`scripts/run_env.py` 支持单进程多环境的专家演示采集（wave 批次模式）。
执行/渲染/录制吃 EmbodiChain 的多 arena 批量能力，专家**规划**放在一个
单环境 worker 子进程里逐 episode 做——两边用同一个 seed，场景逐位一致。

```bash
# 直接跑(注意 --num_envs 会覆盖 run_task.sh 里写死的 1,argparse 取末次出现)
bash launch/run_task.sh click_bell random 3_0 --max_episodes 100 --headless \
    --num_envs 4 --device cuda --seed 7 --save_only_success --success_settle_steps 30

# 或绕过 run_task.sh:
python -m scripts.run_env \
    --gym_config configs/click_bell/random/gym_config.json \
    --action_config configs/click_bell/action_config.json \
    --num_envs 4 --device cuda --headless --seed 7 \
    --max_episodes 100 --success_settle_steps 30
```

要求：**`--device cuda`**（CPU 下多环境退化成逐 env Python 循环）、**必须给
`--seed`**（逐集种子由它派生）。`num_envs=1` 走原串行路径，行为不变。

## 为什么规划要出进程

action bank 的 affordance 管线（`_prepare_warpping` 只取 arena 0）、FK/IK 与
规划器全链都有 batch==num_envs 的强约束，在批量环境里逐 env 规划从节点生成到
Toppra 一路报错；而 10 个任务的 bank 全是「单环境思维」写的（`actions[:, 0, :]`）。
与其批量化几千行规划代码，不如让规划留在它唯一被验证过的形态里：

- **worker**（`scripts/expert_plan_worker.py`）：`num_envs=1`、关录制的独立进程，
  收到 seed 后 `reset(seed)` 复现场景 → 原生 `create_demo_action_list()` →
  返回 `(T, dof)` 轨迹 + 该 seed 的官方初始 qpos。stdio JSON，同 venv。
- **驱动**（`scripts/parallel_collection.py`）：批量环境逐槽位
  `reset(seed=s, reset_ids=[slot], save_data=False)` 摆场景 → 把 worker 的
  init qpos `set_qpos` 回写进槽位 → 轨迹尾帧 padding 到共同 T → 整批
  `env.step` → 自适应静置 → per-env 官方 `is_task_success()` →
  `dataset_manager.apply(mode="save", env_ids=成功槽位)` 直接落盘。

init qpos 回写有双重作用：执行起点与规划起点严格一致；同时中和「部分重置下
机器人随机化与单环境路径不等价」的上游问题（见 docs/parallel_eval.md 已知偏差
——物体场景逐位一致，robot qpos 在 slot>0 有 IK 分支差异）。**因此每条落盘
episode 的场景与初始状态都等于串行种子化采集在该 seed 下会产出的那条**。

## 与串行采集的语义对照

| | 串行（原路径） | 并行 wave |
|---|---|---|
| 种子 | SeededCollection 每次 reset 现抽 | 驱动层从 `--seed` 派生，逐槽位显式播种 |
| 失败处理 | 换 seed 重试，`max_generation_attempts` 守卫 | 同语义（逐槽位重抽，全局 attempts/invalid 守卫，exit 3 口径不变） |
| 成功过滤 | reset(save_data) + recorder 内部 `_task_success` 过滤 | 静置后按官方 `is_task_success()` 逐 env 过滤，只对成功槽位 `apply(save)` |
| episode 长度 | 每集真实长度 | wave 内 padding 到共同 T（尾部「保持姿态」帧，与静置帧同性质）；`current_rollout_step` 标量语义因此保持正确 |
| 边车 | `episode_success.json`（seed/成败/步数） | 同名同字段，另加 `slot` 与 `collection_mode: parallel_wave` |

**为什么不能用 reset 落盘**：`current_rollout_step` 是全 env 共享的标量，部分
重置会把它清零——第一个槽位的 reset 落完盘，第二个槽位就会存出 0 帧的空集。
所以并行模式在任何 reset 之前对成功槽位直接 `apply(mode="save")`。

## 已知限制

- 尾部 padding 帧会进数据（等长方案）。要去掉需要把 EmbodiChain 的
  `current_rollout_step` 升级成 per-env 计数器（上游 `_traj_steps` 已有同款
  实现可抄，约 60-80 行 + recorder 两处切片），属后续优化。
- 录制吞吐大头在存盘：换 `AsyncLeRobotRecorder` + `image_writer_threads` 可把
  保存停顿从「随 N 线性增长」压到近零（上游文档 4env 实测 2.8×）。RSC 侧需要
  一个 ~20 行的子类做相对 save_path 解析（`robosynchallenge/managers/datasets.py`
  目前只 patch 了同步版），属后续优化。
- worker 是串行规划：每集多一次单环境 reset+规划（秒级）。规划占比高的任务
  加速比会被摊薄。
- 依赖 EmbodiChain 本地分支 `parallel-fixes` 的补丁（partial-reset 下的
  events/spatial 修复 + 规划器 B=1 放行），未合上游前不要用官方原版 EmbodiChain 跑。
