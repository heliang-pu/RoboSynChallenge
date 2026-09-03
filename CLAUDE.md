# CLAUDE.md

本 worktree = 分支 `fix/random-spawn-reachability`，从 `main`（`8fc3081`）切出。
只做一件事：**修正 10 个评测任务 `random` / `random`（random 已删除） 配置里物体的随机生成范围**，把机械臂够不到、
专家 IK 必定失败的区域去掉。仓库总说明见 `main` 的 CLAUDE.md，这里只写本分支特有的东西。

## 本分支改了什么

- `configs/<task>/random/gym_config.json` 与 `configs/<task>/random/gym_config.json`：8 个物体事件的 `position_range`
  （click_bell button、drawer duck、pipette、items_handover pen/holder、water_pouring cup、sample_loading rack、handle_basket milk）。
  逐项依据见 `docs/random_spawn_reachability.md`；旋转范围、`clear` / `coverage_*` / `aug_*` 配置没动。
- `scripts/analyze_rigid_spawn_range.py`：跟上 EmbodiChain 启动器的参数变化（`--renderer`、`max_episodes`，`make_env`
  改为与 `run_env.py` 同一条 `config_to_cfg` 路径），否则一启动就 `AttributeError: renderer` / `KeyError: enable_rt`；
  env 缺 `action_config` 属性时补挂。
- `robosynchallenge/tasks/` **一个字节都没改**：判定仍是官方版。

## 范围是怎么定的

用官方 `random` 配置逐物体扫网格（`analyze_rigid_spawn_range.py --event <事件> --grid-size gx gy --trials-per-point 3`，
判据 = 专家动作图能否生成），取不含「三次全失败」点的最大轴对齐矩形。扫描只完成了一部分就按要求停了（不占卡），
没扫到的物体用官方成功专家数据的位置分布（2026-08-27 覆盖审计）和实测的臂可达半径旁证；
handle_basket 的专家代码在上游就是坏的（见下），只能按抓取几何推断。**改后配置没有再跑仿真验证**，
验证命令写在 docs 的 §5，跑之前先确认可以占卡。

## 注意

- **上游 bug**：`HandleBasketEnv.__init__` 不保存 `kwargs["action_config"]`，`create_demo_action_list` 还调用不存在的
  `_sync_carry_basket_runtime_attrs()`；`origin/main` 同样。评测不受影响，采集/分析在第一步就抛 AttributeError。
- 这条分支不含 `.venv` / `tests/`，仿真要用 `main` worktree 的 `.venv`：
  `cd` 到本目录后 `../RoboSynChallenge/.venv/bin/python scripts/analyze_rigid_spawn_range.py ...`。
  资源路径由 `robosynchallenge/data/asset_resolver.py` 按**包根目录**（editable 安装指向 main）解析，gym_config 放哪都行。
- 改了 `random` 就等于改了评测分布：本分支下的成功率与官方口径（origin/main 的 random）**不可直接对表**，报告里要写明。
- 可达域本质是以臂基座为圆心的扇形，矩形只能取子集；要保留完整可行域得加「距臂基座距离」约束的随机化 functor，本分支没做。

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

**每次评估一律走 `scripts/eval_policy.py`**（各 `policy/*/eval.sh` 只是它的包装，`launch/` 下的评估脚本最终也
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
