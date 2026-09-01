# CLAUDE.md

**当前 worktree = `feat/parallel-eval`，实验分支**：给 `scripts/eval_policy.py` 加
单进程多环境并行评估（`--num_envs N`，wave 批次模式）。从 `main` 切出，做完合回。
设计/用法/口径的完整说明在 [docs/parallel_eval.md](docs/parallel_eval.md)，本文件只记环境与坑。

仓库总体约定（环境分层、任务→配置→环境、策略适配契约、分支分工）见 `main` worktree
（`../RoboSynChallenge`）的 CLAUDE.md，这里不重复。

## 本分支改了什么

- `scripts/eval_policy.py`：
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
  跑评估直接 `.venv/bin/python scripts/eval_policy.py ...`（从本目录根执行，configs 相对路径）。
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
