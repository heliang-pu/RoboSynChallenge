# 榜单打分公式与「动作效率」怎么拿分

> 2026-09-02 记录。公式来自 [robosyn-bench.net](https://robosyn-bench.net/#/leaderboard) 的 FAQ，
> 分母 T / 上限 H 的数值直接取自榜单前端代码 `static/js/pages-app.js`
> （`inferenceEfficiencyScore` / `actionEfficiencyScore` / `ACT_INFERENCE_BASELINE_SECONDS` /
> `LEADERBOARD_MAX_ACTION_STEPS_BY_TASK`），不是我们推测的。

## 1. 公式（榜单前端实际实现）

```
Overall Score = 75% × Success Rate + 20% × Action Efficiency + 5% × Inference Efficiency

Episode Action Efficiency    = (1 − Used Action Steps / H) × 100      # 失败集 Used = H → 0
Episode Inference Efficiency = max(0, min(100, (1 − Measured Inference Time / T) × 100))
```

三项都按 episode 算再取平均；Success Rate = 官方判定成功的集数百分比（100 集时就是集数）。

| 任务 | H（步数上限） | T（ACT 基线每集推理时间，RTX 5090） |
|---|---:|---:|
| click_bell | 361 | 0.0669 s |
| drawer_open_place | 900 | 0.1737 s |
| mixer_operating | 500 | 0.0746 s |
| table_rearrangement | 361 | 0.0576 s |
| water_pouring | 500 | 0.0659 s |
| handle_basket | 500 | **无**（前端 `return 0`） |
| items_handover | 350 | 无 |
| item_assembly | 361 | 无 |
| manipulate_pipette | 1000 | 无 |
| sample_loading | 500 | 无 |

前端逻辑：`actInferenceBaselineSeconds(taskId)` 先查发布结果里 ACT 的 `inference_time_ms`，没有就查上表，
再没有就返回 `null`，随后 `inferenceEfficiencyScore` 直接返回 0——**后 5 个任务所有人推理效率都是 0**。

## 2. 推理效率这 5% 对 VLA 恒为 0

T 是 ACT（约 8000 万参数，单次 11.6 ms，每集 5–15 次调用）跑完一集的推理总时间，量级 0.06–0.17 s。
pi0.5（33 亿参数）每集要调用约 19 次（click_bell，`pi0_step=10`）：

| | 每次调用 | 每集推理总时间 | 1 − 时间/T | 计入 |
|---|---:|---:|---:|---:|
| 原版 OpenPI JAX（A100，1 卡 + 14 核） | ~100 ms | ~1.9 s | −27 | 0 |
| 加速版 realtime-vla（同条件） | ~73 ms | ~1.4 s | −20 | 0 |
| 纯 kernel 极限 43 ms（4090） | 43 ms | 0.82 s | −11 | 0 |

要拿到 > 0 需要每次调用 ≈ 3.5 ms——比 ACT 整个模型还快 3 倍，任何 VLA 都做不到；主办方自己的 DP
（每集 0.98 s）、SmolVLA 同样是 0，连 ACT 代入自己的 T 也是 0。**结论：realtime-vla 加速在仿真榜单上不
产生分数**（仿真是同步的，推理快慢也不改变动作序列，成功率/步数不变，同种子对照已验证）。它的价值在真机
决赛的控制频率（19 Hz vs 7 Hz）和评估吞吐。实际总分 ≈ **0.75 × 成功率 + 0.2 × 动作效率**。

## 3. 动作效率的拆解：成功率 × 成功集的效率

失败集贡献 0，所以 `AE = SR × mean_success(1 − steps/H) × 100`。2026-09-02 在 a100-2 上用单任务 pi0.5
checkpoint（`models/single_task/*/28000`，click_bell 用 `BW1000/29999`）、random、100 集、seed 0、
`pi0_step=10` 跑的结果（并行评估器、FastRT 渲染、GPU 物理，**非主办方标准条件，量级参考**；加速版数据）：

| 任务 | 成功率 | 成功集平均步数 / H | 成功集 AE | 总 AE | 成功率 +10 pt → | 成功集步数 −20% → |
|---|---:|---:|---:|---:|---:|---:|
| click_bell | 60% | 66 / 361 | 82 | **49** | 57 | 51 |
| water_pouring | 71% | 271 / 500 | 46 | **33** | 37 | 40 |
| mixer_operating | 79% | 307 / 500 | 39 | **31** | 34 | 40 |
| handle_basket | 89% | 399 / 500 | 20 | **18** | 20 | 32 |

官方 ACT 五任务平均 AE = 33.6（榜单 48.4 分里 6.7 分来自它）；主办方挂的 pi0.5 sim 基线：成功率 38.5%、
平均 797.6 步（H 加权后 AE 很低）。

两类任务、两种抓手：

- **A 类（click_bell）**：成功集本来就快（66 步，AE 82），瓶颈完全是成功率——只能提成功率。
- **B 类（basket / mixer / water）**：成功率不低但成功集慢；步数每减 20%，总 AE 涨 7–14 分，比提 10 个点
  成功率更值。慢的构成：① 判定的硬时间成本（handle_basket 要「篮子移到左侧且瓶在篮内」**连续 75 步**；
  water_pouring 是「抓起 → 倾倒 → 扶正」三段 latch）；② 策略学的是专家示范的节奏（`action_config` 里的
  `duration` 与停顿决定了成功要多少步）；③ 反复抓取/回退的重试。

## 4. 怎么提高动作效率（按成本排序）

1. **按任务设 `pi0_step`**（零成本，部署适配器参数）：短执行长度（10）让 `is_task_success` 更早被检测、闭环
   纠错更勤。本仓库扫描（`_ft67500` 全存档 × H∈{10,50} × 20 集）：click_bell / handle_basket /
   table_rearrangement / manipulate_pipette / items_handover 用 10 更好，drawer_open_place 相反（10 为 0%，
   50 为 30%），water / mixer 50 略优。**逐任务定，不要一刀切。**
2. **挑 checkpoint 时同时看步数**（只花算力）：最后一个存档往往不是最好的（click_bell 最优 ck2220），
   每个存档 20 集筛，指标用 `0.75×SR + 0.2×AE`。
3. **压掉成功后的磨蹭与重试**（数据清洗 + 微调）：只留一次成功、无回退的干净示范；判定条件达成后
   示范应立刻停住。
4. **让策略比示范更快**（要重训，收益最大，也是文献的主战场）。可选路线：
   - 数据层：对示范做时间重采样/去掉静止帧再训练——
     [ESPADA: Execution Speedup via Semantics Aware Demonstration Data](https://arxiv.org/pdf/2512.07371)、
     DemoSpeedup（语义感知地降采样示范，机器人学会更快动作）。本仓库对应做法：缩短采集用 `action_config`
     里各段 `duration`、去停顿后**重新采集**再微调（评估用的官方 `action_config` 不能改，采集用的可以）。
   - 部署层：对预测的 50 步动作块做时间重规划/跳步执行，但要保证可跟踪——
     [RACE (ICLR 2026): Time-Optimal Execution of Action Chunk Policies Beyond Demonstration Speed](https://iclr.cc/virtual/2026/poster/10010308)、
     [B-spline Policy: 用 B 样条表示动作块，时间缩放后仍能平滑重采样](https://arxiv.org/html/2607.09648v1)。
     最简单的试验：适配器里按 stride=2 执行（每个环境步前进 2 个预测动作），基本零成本；风险是关节跟踪滞后、
     抓取时机错位，必须以 `0.75×SR + 0.2×AE` 复评而不是只看步数。
   - 训练层：速度可控/速度增广——
     [TempoVLA: 用变速轨迹增广训练速度可控的 VLA](https://arxiv.org/html/2606.06491)、
     [SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies](https://arxiv.org/pdf/2506.11948)、
     SpeedAug / RLT（速度增广 + 在线微调稳定性）。
   - 接触丰富任务的加速示范精修：
     [Refinement of Accelerated Demonstrations via Iterative Reference Learning Control](https://arxiv.org/html/2604.16850)。
   - 这些工作的共同警告：**单纯提高执行频率会改变状态转移、显著增加状态误差并导致失败**，所以任何提速都要用
     成功率一起验收。

推荐顺序：1 → 2 → 3 → 4（先 stride 试验，再考虑重采集/重训）。今晚 10 任务正式评估出来后，按第 3 节把每个
任务归到 A/B 类，再决定投入。

## 5. 计算脚本

`scripts/score_leaderboard.py`（逻辑与前端一致；用法 `python scripts/score_leaderboard.py <runs_root> --json out.json`）：从各分片 `evaluation_metrics.json` 汇总
成功率、平均步数、每集推理时间，按上表 H/T 计算三项与总分；无 T 的任务推理效率记 0。
