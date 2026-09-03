# 官方 SmolVLA checkpoint 在 A100 上的并行复评（2026-09-01）

用 `feat/parallel-eval` 的分片并行在 a100-2（8×A100-80G，只用 GPU0–3）复评主办方发布的
三个 SmolVLA checkpoint，目的是验证并行评估流程、顺带拿到一组数字。**结论：流程可用，
但数字与官方口径不可比**（原因见下）。

## 配置

- checkpoint：`RoboSynChallenge/SmolVLA_sim_{drawer_open_place,item_assembly,sample_loading}`
  （revision 见 checkpoint 目录 `OFFICIAL_REVISION`）
- 协议：`--setting random --max_episodes 100 --seed 0`，`smolvla_steps=10`（yml 与 origin 一致）
- 并行：每任务 `num_shards=4`（种子集与单进程 100 集完全一致），三任务共 12 进程，
  GPU0–3 每卡 3 进程，`EMBODICHAIN_RENDER_GPU_ID=i`
- 与官方不同的两点：渲染器 **fast-rt**（A100 无 RT core，hybrid 只有 2 step/s）；
  worker 环境 **lerobot 0.6.2**（`smolvla_venv`，图像走 uint8→float 补丁）
- 未开录像（`--eval_video_log False`）
- 墙钟：20:09 → 21:50，**1h41m**（三任务 300 集）

## 结果

| 任务 | 权重（官方发布 ckpt，均 50k steps） | step=10（A100 fast-rt） | 平均步数 | step=50（同链路） | 官方 SmolVLA（RTX，2026-08-21） |
|---|---|---|---|---|---|
| drawer_open_place | `SmolVLA_sim_drawer_open_place` @c0088d84 | **15%**（15/100） | 822.7/900 | 19%（19/100） | 62% |
| item_assembly | `SmolVLA_sim_item_assembly` @db320201 | **39%**（39/100） | 323.6/361 | 0%（0/100） | 24% |
| sample_loading | `SmolVLA_sim_sample_loading` @2a4f41ab | **2%**（2/100） | 498.4/500 | 0%（0/100） | 25% |
| 合计 | | 18.7%（56/300） | | 6.3%（19/300） | |

权重说明：三份都是主办方 HF 仓库 `RoboSynChallenge/SmolVLA_sim_<task>` 的发布存档（revision 记在各
checkpoint 目录的 `OFFICIAL_REVISION`），`train_config.json` 均为 **steps=50000**，chunk_size/n_action_steps=50、
flow num_steps=10。不是我们自训的权重，没有其它 step 数的存档可选。`step=10/50` 指评估时每次请求
弹出的动作数（`--smolvla_steps`），不是训练步数；step=50 结果目录 `a100-2:/data/workspace/scratch/smolvla_eval_step50`。

**注意：本表全部用的是旧引擎（79caf6e6 + parallel-fixes）**，与官方 pin 的 v0.2.4 不可比——引擎版本
才是差距根因，见 `smolvla_official_gap_investigation.md`（v0.2.4 上 drawer 已复测到 36.5%/39.4%）。
item_assembly 还叠加了官方 PR #44 的吸附机制语义变化，旧引擎数字作废。

官方数字来源：robosyn-bench.net 的 `released-checkpoint-evals.js`（random × 100 集，
协议版本 `bd6bf77`，评测硬件 RTX 5090 × 4 分片）。

## 为什么不可比

方向不一致（drawer/sample 大幅偏低，item 反而偏高），说明不是简单的"整体退化"：

1. **渲染器域差**：训练数据与官方评测都是 RTX 上的 hybrid，这里是 fast-rt；
2. **lerobot 版本**：官方评测所用版本未知，0.6.2 的预处理管线与旧版可能不同；
3. 协议本身（判定、配置、步数、种子机制）与 origin 一致，排除。

要拿到可比数字：在 4090/pro6000 上用 hybrid + `smolvla-worker-lr044`（lerobot 0.4.4）
的 worker 环境各跑 20 集对照；若 drawer 回到 60% 附近即坐实 A100 链路问题。

## 顺带确认的工程结论

- A100 上跨进程分片是正解：`--device cuda:<i>` 单进程 18 step/s，同卡多进程近线性；
  重场景（sample_loading）单进程即占半张卡，3 进程/卡把 4 张卡打满，合计 ~28 env-steps/s。
- 官方自己的评测也用 4 分片并发（`evaluation_results/README.md`），分片口径无争议。
- `eval_policy.py` 的两处修复由此而来：渲染器默认 `auto`、`cuda` 设备自动补 gpu_id 索引。
