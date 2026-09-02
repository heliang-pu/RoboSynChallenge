# CLAUDE.md

**当前 worktree = `feat/parallel-collect`，实验分支**：单进程多环境**并行专家数据采集**
（`--num_envs N` wave 批次 + 单环境规划 worker）。从 `feat/parallel-eval` 切出（复用其
种子协议与并行评估），做完连同 eval 分支一起合回 `main`。
设计/用法/口径见 [docs/parallel_collection.md](docs/parallel_collection.md)
与 [docs/parallel_eval.md](docs/parallel_eval.md)；本文件只记环境与坑。

仓库总体约定见 `main` worktree（`../RoboSynChallenge`）的 CLAUDE.md。
本 worktree **没有自己的 .venv**（软链到 main 的），也依赖 EmbodiChain 本地分支
`parallel-fixes-upstream`（= 官方 tag v0.2.4 + 并行修复）（partial-reset 修复 + 规划器 B=1 放行，未合上游）。

## 本分支改了什么

- `scripts/expert_plan_worker.py`（新）：单环境规划 worker。同 seed 复现场景 →
  原生 `create_demo_action_list` → 返回 `(T, dof)` 轨迹 + 官方初始 qpos。
  含 `MotionGenerator` 结果搬回 cpu 的 monkeypatch（tasks/ 红线不许改 bank）。
- `scripts/parallel_collection.py`（新）：wave 驱动层。per-slot 种子化部分重置 →
  init qpos 回写 → 尾帧 padding 整批执行 → 自适应静置 → per-env 官方判定 →
  `dataset_manager.apply(mode="save", env_ids=成功槽位)` 直接落盘 → 边车。
- `scripts/run_env_seeded.py`：`num_envs>1` 分支接线；同款 MotionGenerator cpu 补丁
  （串行 cuda 采集因此也可用）；并行模式不套 SeededCollection。
- `launch/run_task_seeded.sh` **零改动**：`--num_envs N` 作为附加参数即可覆盖写死的
  `--num_envs 1`（argparse 取末次出现）。

## 坑（都实测踩过）

- **落盘不能走 reset(save_data=True)**：`current_rollout_step` 是全 env 共享标量，
  部分重置清零后第二个槽位会存出空集 → 必须在 reset 前直接 `apply(mode="save")`。
- **`recorder.finalize()` 会把 step>0 的所有 env buffer 再存一遍**（给"最后一集没
  被 reset 冲洗"的流程兜底）→ 已显式落盘的流程要先把 `current_rollout_step` 清零
  再 finalize，否则重复集（validate 门禁 `expected 4 found 6` 就是它）。
- **种子体系绑定设备**：随机化走 env 设备的 RNG 生成器，cuda 与 cpu 同种子不同场景。
  并行采集（cuda）与日常串行采集（cpu）的 seed 不可互换复现；并行体系内部
  （worker 与批量环境同 cuda）逐位一致。
- **task bank 假设规划结果在 cpu**（`ret.positions[0].numpy()`），cuda 设备下必须
  经 MotionGenerator monkeypatch 搬运；不要去改 `robosynchallenge/tasks/`（红线）。
- worker 以脚本方式启动时 `sys.path[0]` 是 `scripts/`，editable 安装会把
  `robosynchallenge` 解析到 main worktree —— worker 头部已强制插入本 worktree 根。
- 部分重置下 robot 随机化与单环境路径不等价（详见 docs/parallel_eval.md 已知偏差），
  采集侧靠「worker 回写官方 init qpos」中和，**不要删这步**。
- 首 wave 的槽位重置必须从 slot 0 按序来（材质备份惰性初始化，乱序会 segfault）。

## 评估经验（与 main 的 CLAUDE.md 同步，2026-09-02）

**已验证的经验：`table_rearrangement`、`handle_basket`、`click_bell` 三个任务用 step=10（每次推理执行 10 步，
即 `pi0_step: 10` / `act_step: 10`）成功率更高**，评测这三个任务不要改成更长的开环步数。依据如下：
- 来源：2026-08-30～31 在 2×A100 云机（16 卡）上跑的 pi0.5 `_ft67500` 全存档 × 执行长度扫描，
  44 个 checkpoint × H∈{10,50} × 20 集（同种子、官方判定），报告 `report/单任务微调权重评估_20260831.pdf`
  （`report/` 在 gitignore 里，只在本机）。
- 训练：2026-08-29 起跑，配置 `pi05_<task>_ft67500`（`policy/pi05/src/openpi/training/config.py`，
  一键脚本 `policy/pi05/train_scripts/train_ft67500.sh`）。基座 = all10 共训 `pi05_all10_h64_expert`
  的 ckpt 67500（`all10_expert_base_h64_bs64_steps100k`，海光 DCU 8 卡训，数据集 `RoboSynChallenge/all10_expert_h64`）；
  数据集 = 官方单任务 `RoboSynChallenge/cobotmagic_Sim_<task>`（每任务约 1000 集，10 任务共 86G）；
  `action_horizon=64`（必须与基座一致）、bs=50、2 个 epoch、cosine lr 峰值 1e-5 / warmup 500、
  `ema_decay=None`、norm_stats 复用基座 `all10_expert_h64` 的（不重算）、每任务存 4 个 checkpoint。
- 结果（成功率，20 集）：

  | 任务 | 基线 all10 ck67500 (H=50) | 最佳 H=10 | 最佳 H=50 |
  |---|---|---|---|
  | table_rearrangement | 55% | **95%** @ck7999 | 55% |
  | handle_basket | 60% | **95%** @ck3336 | 45% |
  | click_bell | 60% | **80%** @ck2220 | 50% |

  同一批扫描里 `manipulate_pipette`（75% vs 25%）和 `items_handover`（50% vs 30%）也是 H=10 更好；
  `drawer_open_place` 相反（H=10 为 0%，H=50 为 30%），`water_pouring`/`mixer_operating` H=50 略优。
  另外最终存档往往不是最好的（click_bell 最优在 ck2220，不是末尾 ck2959），挑权重要按存档逐个评。

**每次评估一律走 `scripts/eval_policy_parallel.py`**（各 `policy/*/eval.sh` 只是它的包装，`launch/` 下的评估脚本最终也
落到它），成功率、动作步数、推理时间都以它写出的 `evaluation_metrics.json` 为准，不要另写评估循环、
不要拿训练侧或采集侧脚本的数字充当评估结果。它逐集记：成败（未截断且 `is_task_success()` 为真）、
动作步数（成功取实际步数，失败按 H 计）、每集总推理时间（不含 `env.step()`）。

**官方榜单打分公式（每次评估都要按它算一遍总分，写进 report.md）**：

```
Overall Score = 75% × Success Rate + 20% × Action Efficiency + 5% × Inference Efficiency
Episode Action Efficiency    = (1 − Used Action Steps / H) × 100          # H = 该任务 max_episode_steps
Episode Inference Efficiency = max(0, 1 − Measured Inference Time / T) × 100
```

Success Rate 是官方判定下成功的 episode 百分比；Action/Inference Efficiency 按 episode 算再取平均。
T = 官方 ACT baseline 在 RTX 5090 上推出该任务对应动作步数所需的时间，官方只在 5090 上测，
本地没有 5090，所以 Inference Efficiency 只能给估计值，报告里要注明测量卡型。
`evaluation_metrics.json` 里已有对应原料：`success_rate`、`average_action_steps_ratio`
（= Used Steps / H 的均值，Action Efficiency ≈ (1 − 它) × 100）、
`average_inference_time_per_episode_seconds`（Measured Inference Time）。**官方 `eval.sh` / `eval_policy.py`
都不算效率项和总分，只吐这些原料**，总分要自己按公式算。
分母 T 用官方发布的 `evaluation_results/released_checkpoint_results.json`（与 `official/main` 一致）里 ACT 的
`estimated_average_total_inference_time_seconds`（RTX 5090，单次 11.6 ms × 每集调用次数）：
click_bell 0.067 s / drawer_open_place 0.174 s / mixer_operating 0.075 s / table_rearrangement 0.058 s /
water_pouring 0.066 s；**其余 5 个任务官方没发 T**，只能按「ACT 单次 11.6 ms × 该任务 ACT 每集调用次数」估。
成功率占 75%，但 Action Efficiency 占 20%——同样成功率下更早完成（步数少）的策略分更高，
所以 step=10 这类更早触发 `is_task_success` 的设置对总分是双重收益；失败集的 Used Steps 通常跑满 H，
Action Efficiency 归零，也会拖总分。
