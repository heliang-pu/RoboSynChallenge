# 评估协议、打分与报告

## 官方榜单打分公式（每次评估都要按它算一遍总分，写进 report.md）

```
Overall Score = 75% × Success Rate + 20% × Action Efficiency + 5% × Inference Efficiency
Episode Action Efficiency    = (1 − Used Action Steps / H) × 100     # H = 该任务 max_episode_steps
Episode Inference Efficiency = max(0, 1 − Measured Inference Time / T) × 100
```

Success Rate 是官方判定下成功的 episode 百分比；Action / Inference Efficiency 按 episode
算再取平均。`evaluation_metrics.json` 已有对应原料：`success_rate`、
`average_action_steps_ratio`（= Used Steps / H 的均值，Action Efficiency ≈ (1 − 它) × 100）、
`average_inference_time_per_episode_seconds`（Measured Inference Time）。
**官方 `eval.sh` / `eval_policy.py` 都不算效率项和总分，只吐这些原料**，总分要自己按公式算。

分母 T 用官方发布的 `evaluation_results/released_checkpoint_results.json`（与 `official/main`
一致）里 ACT 的 `estimated_average_total_inference_time_seconds`（RTX 5090，单次 11.6 ms ×
每集调用次数）：

| 任务 | T (s) |
|---|---|
| click_bell | 0.067 |
| drawer_open_place | 0.174 |
| mixer_operating | 0.075 |
| table_rearrangement | 0.058 |
| water_pouring | 0.066 |

**其余 5 个任务官方没发 T**，只能按「ACT 单次 11.6 ms × 该任务 ACT 每集调用次数」估。
官方只在 5090 上测，本地没有 5090，所以 Inference Efficiency 只能给估计值，
**报告里必须注明测量卡型**。

成功率占 75%，但 Action Efficiency 占 20%——同样成功率下更早完成（步数少）的策略分更高，
所以 step=10 这类更早触发 `is_task_success` 的设置对总分是双重收益；失败集的 Used Steps
通常跑满 H，Action Efficiency 归零，也会拖总分。

## 指标口径

`eval_policy_parallel.py` 逐集记录：成败（未截断且 `is_task_success()` 为真）、
动作步数（成功取实际步数，**失败按 H 计**）、每集总推理时间（不含 `env.step()`）。
每 episode 步数上限取 `max(deploy_config.max_steps, gym_config.max_episode_steps)`，
不是 yml 里那个值。计时统一用 `policy/inference_timing.py`（带 cuda synchronize），
不要各写一份。

## 执行长度 H 是被低估的关键参数

同一份权重仅改执行长度（每次推理执行几步），成功率最大相差 50 个百分点。
**不能全局统一取一个 H，必须逐任务定。**

依据：2026-08-30～31 在 2×A100（16 卡）上跑的 pi0.5 `_ft67500` 全存档 × 执行长度扫描，
44 个 checkpoint × H∈{10,50} × 20 集（同种子、官方判定）。

| 任务 | 基线 all10 ck67500 (H=50) | 最佳 H=10 | 最佳 H=50 |
|---|---|---|---|
| table_rearrangement | 55% | **95%** @ck7999 | 55% |
| handle_basket | 60% | **95%** @ck3336 | 45% |
| click_bell | 60% | **80%** @ck2220 | 50% |

同批扫描里 `manipulate_pipette`（75% vs 25%）、`items_handover`（50% vs 30%）也是 H=10 更好；
`drawer_open_place` 相反（H=10 为 0%，H=50 为 30%——开抽屉要一段连贯拉拽，每 10 步打断
重规划会让策略反复犹豫）；`water_pouring` / `mixer_operating` H=50 略优。

两条推论：

- 需要精细纠错的任务偏好短 H（闭环频率高，能及时修正抓取偏差）；需要持续稳定动作的任务必须长 H。
- **最终存档往往不是最好的**（click_bell 最优在 ck2220 而非末尾 ck2959），挑权重要按存档逐个评。

配置项：`pi0_step` / `act_step`（pi0.5 另有 `phase_action_chunks`，pi05_lerobot 有
`per_step` vs `chunk`）。

## 并行评估（`feat/parallel-eval` 分支）

`--num_envs N`：单进程内 wave 批次并行 N 个 episode，与 `--num_shards` 多进程分片正交可叠加。

- **必须 `--device cuda`**（CPU 下 DexSim 退化成逐 env Python 循环）。
- 种子口径不变：rng 按 episode 序号同序抽取，每槽位
  `reset(seed=seed_k, options={"reset_ids": [slot]})` 单独播种。物体场景与单环境同种子
  逐位一致（≤6e-8）；**但 robot 初始 qpos 在 slot>0 不逐位一致**（IK 在部分 env_ids
  路径下解不同），成功率统计等价、逐 seed 复现单集轨迹仍需 `num_envs=1`。
- 甜点要实测：4090 上 N=4 约 1.7×，N=6 起渲染饱和反而变慢（每步渲 3 相机 × N 路 view）。
  **A100 卡内并行不划算**（无 RT core，只能用 `fast-rt`，N>1 负扩展），A100 上靠
  **跨进程分片 + 多卡**，每卡 1 个进程。
- 渲染器：RTX 用 `hybrid`，数据中心卡（A100/H100）必须 `fast-rt`，`auto` 会自动选。
  A100 上误用 hybrid 会慢约 25 倍（2 step/s vs 11 step/s）。
- `rollout_save` 与并行互斥；`eval_video_log` 并行下自动关闭。

分片语义：种子 rng 在所有分片里照常逐个抽取、只跳过不属于自己的 episode，所以 N 个分片
合并后与「单进程跑满 max_episodes」是同一批种子——**改这段循环会静默破坏可比性**。

## 产物落盘与报告

评估视频一律写到 `/root/workspace/eval_results/`（评估机上的绝对路径，仓库外，不进 git），
按 `<task>/<model>/<framework>/<step>/` 分目录：顶层固定是 10 个任务名，往下是模型
（如 `pi05`）/ 推理框架（如 `jax`、`realtime_vla`）/ checkpoint 步数。视频、
`evaluation_metrics.json`、`report.md` 直接放在步数目录里。

**每次评估都要写评估报告**（`report.md`，与视频同目录），至少含：

- 按官方公式算出的 Overall Score
- **所用权重的完整标识**：仓库/revision 或训练配置 + checkpoint 步数（例如
  `pi05_ft67500/click_bell/2220`，或官方发布 ckpt 的 HF revision + train_config 的 steps）
- checkpoint / 配置、`max_episodes` 与种子口径、引擎版本
- 官方判定下的成功率、逐 episode 成败、失败模式归类
- 测量卡型（Inference Efficiency 是估计值时尤其要注明）

结论要长期留档的再拷一份进 `docs/`。仓库内的 `eval_result/` 只是本机临时产物，别当归档。

## 复现官方数字时的注意

主办方自述 `reset(seed)` 只固定 torch seed，未固定 Python `random`、NumPy 与策略推理噪声，
**同一 seed 的成功率有波动**，需要多轮取均值。n≈100 时抽样误差约 ±7 个百分点，
差距在这个范围内不必去找链路 bug。真要对齐，先确认引擎版本一致（见 SKILL.md 红线 3）。
