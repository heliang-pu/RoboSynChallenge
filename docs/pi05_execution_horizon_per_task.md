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

## 4. 官方条件复测（2026-09-02/03 夜，28000 权重，realtime-vla 加速后端）

条件：`random`、100 集、`--seed 0`、官方判定、官方 gym_config、CPU 物理、**Hybrid 渲染**（主办方默认），
`scripts/eval_policy_parallel.py` 分片并行（种子序列与官方单进程一致）。机器：RTX 4090 48 GB（fmc3-1）与 RTX PRO 6000；
两台机器逐集种子相同，但不是同一张卡，跨机对照要留 ±9 个百分点的抽样噪声余量。
视频与 `report.md` 归档在 a100-2 `/root/workspace/eval_results/<task>/pi05/realtime_vla/<step>/`（对照 H 在 `alt_h<H>/`）。

| 任务 | H | 成功率 | 平均步数 | AE | Score | 机器 |
|---|---:|---:|---:|---:|---:|---|
| click_bell | **10** | 76% | 131 | 63.6 | 69.7 | 4090 |
| drawer_open_place | **50** | 20% | 799 | 11.2 | 17.2 | 4090 |
| handle_basket | **10** | 90% | 426 | 14.8 | 70.5 | 4090 |
| item_assembly | **50** | 45% | 317 | 12.2 | 36.2 | 4090 |
| items_handover | **50** | 35% | 335 | 4.3 | 27.1 | 4090 |
| manipulate_pipette | 10 | 71% | 496 | 50.4 | 63.3 | PRO 6000 |
| manipulate_pipette | **30** | 68% | 497 | 50.3 | 61.1 | PRO 6000 |
| mixer_operating | **10** | 78% | 371 | 25.8 | 63.7 | 4090 |
| mixer_operating | 50 | 80% | 312 | 37.6 | 67.5 | PRO 6000 |
| sample_loading | **10** | 2% | 499 | 0.1 | 1.5 | PRO 6000 |
| table_rearrangement | **10** | 70% | 225 | 37.8 | 60.1 | 4090 |
| water_pouring | 10 | 76% | 345 | 31.0 | 63.2 | PRO 6000 |
| water_pouring | **50** | 52% | 344 | 31.3 | 45.3 | 4090 |

加粗 = 第 1 节定稿值。**定稿配置 10 任务宏平均：SR 53.6%，AE 25.1，IE 0，Overall 45.2**；
若三个「不确定」项改按本次复测取优（pipette 10、mixer 50、water 10）：SR 56.5%，AE 26.3，Overall 47.6。

同机对照补跑（第三轮，同样 100 集 / seed 0 / Hybrid）：

| 任务 | H | 成功率 | 平均步数 | AE | Score | 机器 |
|---|---:|---:|---:|---:|---:|---|
| water_pouring | 50 | 52% | 344 | 31.2 | 45.2 | PRO 6000（与 4090 的 52% 完全一致） |
| mixer_operating | 10 | 79% | 376 | 24.9 | 64.2 | PRO 6000（4090 为 78%） |
| items_handover | 10 | 18% | 348 | 0.4 | 13.6 | 4090（H=50 为 35%） |

跨机复现性很好（同配置两机差 ≤1 个点），所以上表跨机对照可以直接比。

### 渲染器差异：同一权重、同一种子，FastRT（a100-2）vs Hybrid（RTX）

a100-2 上用 FastRT 渲染（`--renderer auto`，A100 无 RT core 时的选择，**非官方条件**）跑的同一批配置，100 集 / seed 0：

| 任务 | H | FastRT 成功率 | Hybrid 成功率 | 差 |
|---|---:|---:|---:|---:|
| click_bell | 10 | 77% | 76% | −1 |
| handle_basket | 10 | 87%（67 集） | 90% | +3 |
| table_rearrangement | 10 | 74% | 70% | −4 |
| mixer_operating | 10 / 50 | 82% / 87% | 78% / 80% | −4 / −7 |
| manipulate_pipette | 10 / 30 | 76%（80 集） / 74% | 71% / 68% | −5 / −6 |
| water_pouring | 10 / 50 | 73% / 74% | 76% / 52% | +3 / **−22** |
| item_assembly | 50 | 56% | 45% | −11 |
| items_handover | 50 | 74% | 35% | **−39** |
| drawer_open_place | 50 | 28%（86 集） | 20% | −8 |
| sample_loading | 10 | 4% | 2% | −2 |

FastRT 下定稿配置 Overall 53.0（SR 62.9%），Hybrid 下 45.2（SR 53.6%）。items_handover、water_pouring(H=50)、item_assembly
对渲染器最敏感——主办方用 Hybrid，**任何用 FastRT 得到的成功率都不能当提交预期**，只能用来比较 H / checkpoint 的相对好坏，
而且 water_pouring 的 H 结论在两种渲染器下相反（FastRT 50≈10，Hybrid 10 远好于 50），相对结论也要在 Hybrid 下复核。

第四轮（H=30 对照，100 集 / seed 0 / Hybrid）：

| 任务 | H | 成功率 | 平均步数 | AE | Score | 机器 |
|---|---:|---:|---:|---:|---:|---|
| items_handover | 30 | **61%** | 329 | 6.1 | 47.0 | 4090（H=10 18%，H=50 35%） |
| item_assembly | 30 | **59%** | 302 | 16.2 | 47.5 | PRO 6000（H=50 45%） |
| drawer_open_place | 30 | 0% | 900 | 0 | 0 | PRO 6000（H=50 20%；10 与 30 都是 0） |

### 复测后的建议配置（待拍板）

| 任务 | 定稿值 | 复测建议 | 单项分变化 | 依据 |
|---|---:|---:|---:|---|
| water_pouring | 50 | **10** | +17.9 | 两机 H=50 都 52%，H=10 76% |
| items_handover | 50 | **30** | +19.9 | 61% vs 35%，一次 100 集，建议换种子再验一次 |
| item_assembly | 50 | **30** | +11.3 | 59% vs 45%，一次 100 集，同上 |
| mixer_operating | 10 | **50** | +3.8 | 80% vs 78–79%，AE 高 12 |
| manipulate_pipette | 30 | 10 或 30 | +2.2 | 71% vs 68%，噪声内 |
| 其余 5 个 | 不变 | 不变 | — | click_bell/basket/table 10；drawer 50（10/30 都是 0）；sample_loading 10 |

全部采纳后 10 任务宏平均：**SR 60.5%，Overall 50.8**（定稿配置 45.2）。

对三个待定项的结论（复测后）：
- manipulate_pipette：10 与 30 差 3 个点、AE 相同，仍在噪声内，**10/30 任取**；30 保留为定稿值。
- mixer_operating：两机都是 50 略高（80% vs 78–79%），且 AE 高 12（成功集更早触发判定），**建议改 50**（+3.3～3.8 分）。
- water_pouring：H=50 在两台机器上都是 52%，H=10 是 76%，差 24 个点是真实的（Hybrid 下），**建议改 10**（+17.9 分）。
  a100-2 上 FastRT 的 H=50 是 85%（前 34 集）——同一权重换渲染器成功率差 30 个点，water_pouring 对渲染器很敏感。
- items_handover：28000 这批 H=10 只有 18%，H=50 35%，**维持 50**。
- items_handover 35%、drawer 20% 明显低于表 A（71% / 26%）——表 A 是 FastRT 渲染，Hybrid 下画面不同，
  两个任务对渲染差异敏感，提交前要按主办方渲染器（Hybrid）的数字预期。
