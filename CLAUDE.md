# CLAUDE.md

**当前 worktree = `feat/parallel-eval`，实验分支**：给 `scripts/eval_policy_parallel.py` 加
单进程多环境并行评估（`--num_envs N`，wave 批次模式）。从 `main` 切出，做完合回。
设计/用法/口径的完整说明在 [docs/parallel_eval.md](docs/parallel_eval.md)，本文件只记环境与坑。

仓库总体约定（环境分层、任务→配置→环境、策略适配契约、分支分工）见 `main` worktree
（`../RoboSynChallenge`）的 CLAUDE.md，这里不重复。

## 本分支改了什么

- `scripts/eval_policy_parallel.py`：
  - `ParallelEvalProxy`：并行裁判代理。逐 `env.step` 锁存 per-env 成功位（截断步不计成功），
    把 `get_wrapper_attr("is_task_success")` 拦截成「全部完成才 True」的聚合标量——
    **策略适配器因此零改动**；收到第一维 ≠ num_envs 的动作会直接报错兜底。
  - `run_parallel_episodes`：wave 循环。种子与串行同序抽取、每槽位
    `reset(seed=seed_k, options={"reset_ids": [slot]})` 单独播种（初始场景与单环境同种子
    逐位一致，已实测）；分片 `num_shards` 规则不变，可叠加。
  - `env.close()` 从 `finally` 挪到 main 末尾指标落盘之后（见下「坑」）。
  - 串行循环体逐字节未动（`sequential_episodes = ... if num_envs == 1 else ()`）。
- 首个提交是「基线同步」：把 main 工作区未提交的 `num_shards` 分片版 eval_policy.py
  原样带进来，合回 main 时与其工作区版本无冲突。

## 环境

- 本 worktree **没有自己的 .venv**：`.venv` 是指向 `../RoboSynChallenge/.venv` 的软链接。
  跑评估直接 `.venv/bin/python scripts/eval_policy_parallel.py ...`（从本目录根执行，configs 相对路径）。
- 并行评估**必须 `--device cuda`**：DexSim 在 CPU 设备下对多环境退化成逐 env Python 循环。

## 坑（本分支实测记录）

- **`env.close()` 会直接终止进程（exit 0）**，本机 CPU/CUDA 设备都复现。原版把 close 放在
  `finally`，导致 summary 与 `evaluation_metrics.json` 从未写出（主 worktree `eval_result/`
  里一个 metrics 文件都没有）。本分支已把 close 挪到指标落盘后；这个修复对串行路径同样生效，
  合回 main 时值得单独说明。
- 截断步语义：底层 `env.step()` 在截断步会对 done env 自动部分重置并清任务锁存，所以
  「截断当步的成功不算成功」两边天然一致，不要在代理里试图“抢救”截断步的成功位。
- `mode: interval` 的随机化事件（灯光等）在批量下共享全局 RNG 流：初始场景可对表，
  过程随机量与串行不逐位一致（同分布）。
- 冒烟测试没有训好的 checkpoint 时，用随机初始化 ACT checkpoint 验证管线
  （lerobot 在根 venv 里，`ACTPolicy(config, dataset_stats=...).save_pretrained(...)`，
  state/action 14 维、三路 480×640 相机）。

## 策略支持状态

`act` 进程内路径 ✅（LeRobot 批量 select_action 天然可用）；worker 类
（`act` worker / `smolvla` / `pi05_lerobot`）与 `pi05`（openpi/JAX+RTC）❌ 未批量化，
详见 docs/parallel_eval.md 的表格。

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
