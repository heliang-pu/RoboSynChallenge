# 官方 SmolVLA 复现差距调查：根因是 EmbodiChain 引擎版本（2026-09-02）

## 现象

主办方发布的 `SmolVLA_sim_drawer_open_place` checkpoint，官方 100 集 62%，主办方用我们提供的
8 个 base seed（833500…834200，各 13 集）复测 48/104 = 46.2%；我们同种子只有 17–19%。

## 实验矩阵（同 checkpoint、同种子、`--setting random`、`smolvla_steps=10`）

| 环境 | 引擎 | 渲染器 | worker | 结果 |
|---|---|---|---|---|
| a100-2（8-31 报告） | 旧 EmbodiChain（79caf6e6 + parallel-fixes） | fast-rt | lerobot 0.6.2 / tf 5.16.1 | 18/104 = 17.3% |
| 本机 4090 | 旧 EmbodiChain | **hybrid** | 同上 | 20/104 = 19.2% |
| 本机 4090 | **origin/main 引擎（69deef71）** | hybrid | 同上 | **42/104 = 40.4%** |
| 主办方 | 最新仓库 | RTX（hybrid） | ？ | 48/104 = 46.2% |

逐种子（4090 上游引擎 / 主办方）：833500 6/6、833600 5/10、833700 4/5、833800 5/5、
833900 3/4、834000 6/6、834100 8/5、834200 5/7。剩余差距在主办方自述的种子不可复现范围内
（`reset(seed)` 只设 torch seed）。

## 逐项排除（都是实测，不是推断）

- **渲染器/GPU**：4090 hybrid 与 A100 fast-rt 同为 ~18%，排除。
- **transformers 版本**：5.16.1 与 5.5.4（lerobot 原钉版）对同一固定观测的策略前向输出逐位相同
  （max diff = 0），排除。
- **权重加载**：checkpoint 500 个张量键全部匹配、数值逐位一致，排除。
- **观测 qf**：训练数据全零，评估时环境给的也是零，排除。qvel 两边都是真实值。
- **gripper 表示**：评估 obs 的 gripper 是归一化 [0,1]（raw 0..0.05），官方数据首帧 0.324 只是
  专家第一步张开后的读数（我们命令张开一步后 0.25–0.38），一致，排除。
- **初始分布**：机器人首帧 qpos 均值/标准差、鸭子位姿分布（x≈0.69±0.07，y≈0.25±0.04）两边一致，排除。
- **判定/配置**：drawer 判据与 origin 一致（origin 只把成功位改成锁存，检测等价）；
  `configs/drawer_open_place/random` 与 origin 逐字节一致；worker/yml/eval.sh 与 origin 一致。
- **引擎版本**：本地 EmbodiChain 落后 origin/main 85 个提交（2026-07-31 → 09-01，含物理默认
  参数、被动关节默认值、材质随机化生命周期等）。换引擎后 19% → 40%。**这是根因。**

## 顺带发现：`smolvla_steps=10` 实际是 50 步开环

worker 只在 `reset` 命令时清 lerobot 的动作队列；`select_action` 只在队列空时重新推理，队列长度
= `n_action_steps` = 50。所以每次请求弹 10 个动作、**每 5 次请求才真推理一次**，中间 4 次送的
新观测被忽略。日志证据：`infer_time` 每第 1/6/11… 次请求 0.2–0.4s，其余 ~0.08s。主办方用同一份
代码，行为一致，因此不是差距来源，但协议语义与字面不符。真闭环（每次请求清队列，worker 开关 `SMOLVLA_REINFER_EACH_REQUEST=1`）在**旧引擎** A100 上的
对照：31/104 = 29.8%（同引擎开环 19.2%）——闭环本身 +10 个点，与引擎效应（+21 个点）独立。
与官方对表时应保持默认（开环），因为主办方同代码同行为。

## 处置

- **官方引擎版本是 tag `v0.2.4`**（RSC `docs/getting_started/installation.md`：`git checkout tags/v0.2.4`，
  9ebee300，2026-08-06）。EmbodiChain HEAD 的 #550 重构删掉了 `embodichain_tasks.tableware.base_agent_env`，
  RSC 全部任务 import 它，**HEAD 不可用**。本地引擎分支 `parallel-fixes-upstream` = v0.2.4 + partial-reset
  修复 + `EMBODICHAIN_RENDER_GPU_ID` + planner B=1 放行；visual.py 的纹理池补丁被上游更通用的实现覆盖，弃用。
  旧基线 79caf6e6 → v0.2.4 共 17 个提交（含 #456 Remove default physics arguments、#459 材质生命周期）。
  v0.2.4 上并行评估/并行采集冒烟均通过；8 种子 drawer 在 v0.2.4 上的确认结果见文末。
- RSC 两条并行分支已合入 origin/main（tasks/ 重新与官方逐字节一致；`run_env.py` 收尾段取 origin
  的 finally + `flush_cleanup_queue` 写法）。
- **所有评测必须在 v0.2.4 引擎上跑**（`parallel-fixes-upstream` 分支）；旧引擎（0.2.3 基线）的历史成功率
  （含本仓库 README 里的表）与官方口径不可比。
- 上表中 40.4% 那行用的是 HEAD(69deef71)+旧 embodichain_tasks 副本；v0.2.4 的复测结果待补。
