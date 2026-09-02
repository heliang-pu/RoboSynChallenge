# pi0.5 各任务的执行长度（`pi0_step`）决定表

> 2026-09-02 定稿。`pi0_step` = 每次推理后实际执行的动作步数（模型每次预测 50 步），是部署适配器
> `policy/pi05/deploy_policy.yml` / `--pi0_step` 的参数，主办方协议允许队伍自定；**不改任何官方文件**。
> 依据是下面两次扫描（表 A、表 B），最终取值由 heliang-pu 拍板。

## 1. 最终取值（含确定程度）

确定程度：**确定** = 两批权重、两次扫描方向一致且差距远大于噪声；**中等** = 只有一批权重有信号、或差距
接近 100 集的抽样噪声（±9 个百分点）；**不确定** = 成功率无差别或两批权重结论相反，取值是拍板/按动作效率
定的，换权重或复跑都可能翻转。

| 任务 | `pi0_step` | 确定程度 | 依据 / 不确定在哪 |
|---|---:|---|---|
| click_bell | **10** | 确定 | 表 A 75% vs 68%；表 B 80% vs 50%，两批一致 |
| handle_basket | **10** | 确定 | 表 A 96% vs 92%；表 B 95% vs 45%，一致 |
| table_rearrangement | **10** | 确定 | 表 A 78% vs 66%；表 B 95% vs 55%，一致 |
| drawer_open_place | **50** | 确定 | 表 A 26% vs 0%；表 B 30% vs 0%，一致；10 步两批都是 0 |
| items_handover | **50** | 中等（随权重批次翻转） | 28000 这批表 A 71% vs 48%（100 集，差距远超噪声）→ 对这批可信；但 ft67500 批次表 B 相反（50% vs 30%）。**换权重必须重验** |
| item_assembly | **50** | 中等 | 只有表 A 有信号：28% vs 17%（100 集，差 11 个点，约 1.9σ）；表 B 两档都 0 |
| manipulate_pipette | **30** | 不确定（10 与 30 之间） | 表 A 三档 66/72/68 只差 6 个点，在噪声内；表 B 里 50 崩到 25%——**唯一确定的是避开 50** |
| water_pouring | **50** | 不确定（10 与 50 之间） | 表 A 76/79/80、表 B 90 vs 95，两批差距都在噪声内，10 或 50 都可以 |
| mixer_operating | **10** | 不确定（拍板） | 表 A 三档全 100% 无区分；按动作效率取短执行。表 B（ft67500 批）倾向 50（85% vs 75%，20 集） |
| sample_loading | **10** | 不确定（无意义） | 两批三档都 ≈0，取值不影响成绩 |

**要靠复跑才能定的**：manipulate_pipette（10 vs 30）、water_pouring（10 vs 50）、mixer_operating（10 vs 50）
——在提交用的 checkpoint 上各跑 100 集对比，按 `0.75×SR + 0.2×AE` 取；items_handover 换权重批次时同样处理。

机器可读版本：[`policy/pi05/task_pi0_step.json`](../policy/pi05/task_pi0_step.json)。

按表 A 逐任务取优后的合计成功率 **629/1000 = 62.9%**，对比统一 10 步 55.6%、统一 50 步 60.0%。

## 2. 依据

### 表 A — `models/single_task/*`（28000；click_bell 为 BW1000/29999），random，每点 100 集，h ∈ {10, 30, 50}

| 任务 | h = 10 | h = 30 | h = 50 |
|---|---:|---:|---:|
| item_assembly | 17% | 22% | **28%** |
| sample_loading | 0% | 3% | 1% |
| drawer_open_place | 0% | 2% | **26%** |
| water_pouring | 76% | 79% | **80%** |
| click_bell | **75%** | 62% | 68% |
| handle_basket | **96%** | 93% | 92% |
| items_handover | 48% | 69% | **71%** |
| manipulate_pipette | 66% | **72%** | 68% |
| table_rearrangement | **78%** | 66% | 66% |
| mixer_operating | 100% | 100% | 100% |
| 合计 | 55.6% | 56.8% | 60.0% |

### 表 B — `pi05_<task>_ft67500` 全存档 × H ∈ {10, 50}，每点 20 集，取各 H 下最好的存档（另一批权重，
基座 all10 ck67500，`action_horizon=64`）

| 任务 | 基线 67500 | 最佳 H=10 | 最佳 H=50 | 取优 |
|---|---:|---:|---:|---:|
| table_rearrangement | 55% | **95%** @ck7999 | 55% | +40 |
| handle_basket | 60% | **95%** @ck3336 | 45% | +35 |
| items_handover | 20% | **50%** @ck3270 | 30% | +30 |
| manipulate_pipette | 50% | **75%** @ck8580 | 25% | +25 |
| click_bell | 60% | **80%** @ck2220 | 50% | +20 |
| mixer_operating | 75% | 75% | **85%** @ck12639 | +10 |
| water_pouring | 90% | 90% | **95%** @ck8080 | +5 |
| drawer_open_place | 55% | 0% | **30%** @ck16999 | −25 |
| item_assembly | 0% | 0% | 0% | 0 |
| sample_loading | 0% | 0% | 0% | 0 |

逐任务取最优 60.5% vs 基线 46.5%。**注意**：每点只有 20 集且是「44 个存档里取最大值」，有选择偏差，
最优值偏乐观；表 B 只用来判断方向，不作为提交数字。

### 两表冲突处怎么定的

- `items_handover`：表 A（100 集）50 好 23 个点，表 B（20 集）10 好 20 个点。**最优 H 依赖权重批次**；
  提交用的是 28000 这批，按表 A 取 50。换成 ft67500 系列权重时要改回 10。
- `manipulate_pipette`：表 A 三档差 6 个点以内，表 B 里 50 崩盘——取 30，兼顾两批。
- `mixer_operating`、`sample_loading`：成功率无差别，按动作效率（短执行更早触发判定）取 10。

## 3. 怎么用

- 单次评估：`--overrides ... --pi0_step <值>`（官方脚本和 `eval_policy_parallel.py` 都支持，值从 yml 覆盖）。
- 批量脚本读 `policy/pi05/task_pi0_step.json` 取值；`policy/pi05/deploy_policy.yml` 里的默认 10 保持不动。
- 换权重批次（尤其 ft67500 系列）时按表 B 的方向重验 `items_handover` / `manipulate_pipette`。
- 榜单 20% 是动作效率（见 [leaderboard_scoring_and_action_efficiency.md](leaderboard_scoring_and_action_efficiency.md)），
  两档成功率接近时优先短执行；差距大的任务（drawer / handover / item_assembly）成功率主导，用 50。

## 4. 待办

2026-09-02 晚的 10 任务正式评估（28000 权重，全部 `pi0_step=10`）跑完后，对 drawer_open_place、
items_handover、item_assembly、water_pouring 按 50、manipulate_pipette 按 30 补跑 100 集，正式成绩取本表配置。
