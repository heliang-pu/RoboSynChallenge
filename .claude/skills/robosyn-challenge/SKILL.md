---
name: robosyn-challenge
description: RoboSynChallenge 2026 双臂操作挑战赛的参赛上下文与本仓工作约定——赛制/时间表/提交物、官方评测协议与打分公式、成功判定与引擎版本红线、环境分层、策略适配契约、分支与产物规范。用于跑评测、算总分、写评估报告、接新策略、判断某个改动会不会破坏官方口径。不用于 sim-RECAP 一轮流程（见 sim-recap skill）。
---

# RoboSynChallenge 2026

基于 EmbodiChain 的双臂（CobotMagic，14-DoF）操作挑战赛：10 个仿真任务、专家数据采集、
10+ 策略的训练/部署接入、统一评估协议。主办方 EDEM-AI，成果在 NeurIPS 2026 展示。

面向使用者的完整说明在仓库 `README.md` / `SETUP.md` / `docs/`。本 skill 只装
「读多个文件才能看出来」以及「踩过才知道」的东西。

## 五条红线（违反会让成绩与官方对不上）

1. **成功判定只认官方版**：每步调用 `env.get_wrapper_attr("is_task_success")()`，
   且该 episode 必须未 truncated。`compute_task_state()` 返回的 success 位在多数任务里
   被刻意置 0，**不能拿来算成功率**；`XxxTestEnv` 变体恒为 True，只供目视/采集。
   `robosynchallenge/tasks/` 必须与 `origin/main` 逐字节一致——改这里等于改判据。
2. **官方入口脚本一律原样**：`scripts/eval_policy.py`、`scripts/run_env.py`、
   `launch/run_task.sh` 与 `origin/main` 逐字节一致，正式/对外评测只用它们。
   要加功能改 `_parallel` / `_seeded` 版（见 references/architecture.md）。
3. **引擎版本必须是官方 pin 的 EmbodiChain `v0.2.4`**
   （`docs/getting_started/installation.md` 里的 `git checkout tags/v0.2.4`）。
   旧引擎的历史成功率与官方口径不可比——实测同一 checkpoint 同一组种子，
   旧引擎 17–19%、v0.2.4 上 36–39%，差距全部来自引擎。EmbodiChain HEAD 也不可用：
   上游 #550 重构删掉了 `embodichain_tasks.tableware.base_agent_env`，本仓所有任务都 import 它。
4. **评估一律走 `scripts/eval_policy_parallel.py`**（各 `policy/*/eval.sh` 只是它的包装）。
   成功率、动作步数、推理时间都以它写出的 `evaluation_metrics.json` 为准；
   不要另写评估循环，不要拿训练侧或采集侧脚本的数字充当评估结果。
5. **公开仓库卫生**：内网 SSH 端点、私有 pip 源地址一律不入库，只写「需分发权限」。
   机器地址走环境变量（如 `RECAP_SYNC_REMOTE`、`DM05_REMOTE_HOST`），未 export 时脚本直接报错退出。

## 10 个评测任务

`bash launch/_print_available_tasks.sh` 是权威清单，分三档：

- 低阶：`click_bell`、`handle_basket`、`water_pouring`、`table_rearrangement`
- 中阶：`items_handover`、`drawer_open_place`、`mixer_operating`
- 高阶：`item_assembly`、`manipulate_pipette`、`sample_loading`

`tasks/_other_tasks/` 里的（`pour_water`、`open_pan` 等）是历史/派生环境，**不进评测**，
别拿它们的判定当参考。

setting：`clear`（无随机化，日常验证）、`random`（**官方评测口径**）、
`aug_*` / `coverage_*`（自制采集配置）。

## 赛制与时间表

| 时间 | 阶段 |
|---|---|
| 7月13日—10月11日 | 训练阶段：本地训练与仿真自测 |
| **10月11日—10月18日** | **仿真初赛**：提交策略，主办方统一评测（不是自行上报成绩） |
| 10月18日 | 晋级公布 |
| 10月18日—11月15日 | 决赛更新：晋级团队继续优化 |
| 11月15日起 | 真机决赛：主办方统一部署真实机器人 |
| 12月上旬 | 公布结果，NeurIPS 2026 展示 |

**提交物**：代码仓库 URL、HuggingFace checkpoint URL、模型与实验说明、训练数据来源、
运行方式与依赖说明；策略须按官方部署接口封装成 `deploy_policy.py` + `deploy_policy.yml` + `eval.sh`。

排行榜先按成功率排序，成功率相同时动作步数更少、推理时间更低者优先。

## 按任务分流

- **跑评测 / 算总分 / 写评估报告** → 读 [references/evaluation.md](references/evaluation.md)
  （打分公式、T 值、执行长度 H 的逐任务经验、并行评估、产物落盘规范）。
- **接新策略 / 改适配器 / 改配置** → 读 [references/architecture.md](references/architecture.md)
  （任务→配置→环境、策略适配契约、overrides、manager 注入）。
- **装环境 / 跑训练 / 跑测试 / 采数据** → 读 [references/environments.md](references/environments.md)
  （环境分层、常用命令、测试现状、采集流水线）。
- **切分支 / 放产物 / 提交** → 读 [references/repo-conventions.md](references/repo-conventions.md)
  （分支分工与 worktree、产物落盘、gitignore、无 lint 门禁）。
