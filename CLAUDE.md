# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RoboSynChallenge：基于 EmbodiChain 的双臂（CobotMagic，14-DoF）操作挑战赛仓库——10 个仿真任务、
专家数据采集流水线、多个策略的训练/部署接入，以及统一的评估协议。
面向使用者的完整说明在 [README.md](README.md)、[SETUP.md](SETUP.md)、`docs/`；本文件只记录
「读多个文件才能看出来」的东西。

**当前 worktree = `feat/realtime-vla-pi05`，实验分支。** 唯一主题是 pi0.5 推理加速：
用 [`dexmal/realtime-vla`](https://github.com/dexmal/realtime-vla) 的 Triton kernel 替换推理路径，
**OpenPI checkpoint 保持只读**。分支分工见文末。

> 这个 worktree 里**没有 `tests/`、没有仓库根 `.venv`**，policy 目录也比 `main` 少
> （没有 `dm05` / `lila_wam` / `pi05_lerobot`）。别照搬 `main` 的 CLAUDE.md。

## 本分支：realtime-vla 加速

代码全在 `policy/pi05/realtime_vla/`，用法见该目录的 `README.md`：

| 文件 | 作用 |
|---|---|
| `convert_checkpoint.py` | JAX checkpoint → 加速器权重（`.pkl`），**不改动原 checkpoint** |
| `tokenizer_adapter.py` | 把 OpenPI 原生的 `paligemma_tokenizer.model` 桥接成 realtime-vla 期望的 HF tokenizer 形状 |
| `accelerated_policy.py` | 加速推理路径 |
| `benchmark.py` / `benchmark_e2e.py` / `benchmark_jax.py` | 分层基准（kernel / 端到端 / JAX 基线） |
| `validate_outputs.py` | 加速前后输出一致性校验 |

**加速器是外部 clone，不是 submodule**：需要把 `dexmal/realtime-vla` clone 到本仓库同级目录并
`checkout b86a942`（README 里钉死了这个 commit）。换 commit 前先跑 `validate_outputs.py`。

**跑转换/基准用的是 `main` worktree 那边的 pi05 环境**，不是本目录的——README 里的命令写的是
`../RoboSynChallenge/policy/pi05/.venv/bin/python`，配合 `PYTHONPATH=.` 让它 import 到本 worktree 的代码。
这是有意的：本分支只改推理路径，不复制一份 pi05 训练环境。

**已实测结果**（`RESULTS.md`，2026-08-20，RTX 4090 / PyTorch 2.7.1+cu126 / Triton 3.3.1，
checkpoint `pi05_click_bell_baseline/19999`，三路 224×224 + 14 维 state，输出 50 步 chunk）：
端到端 **80.89 ms → 43.26 ms（1.87×，延迟降 46.5%）**。改动前后要重跑基准就用同一组条件，
否则数字不可比。**仿真回归 2026-09-01 才第一次跑通**（同种子逐集结局与 JAX 一致，见 `RESULTS.md`
「Simulator regression」），之前一直在 `Creating model...` 处 abort，根因和修法都记在那一节里；
最容易再踩的一条：**realtime-vla 用默认 `global` 模式录 CUDA graph 会把 DexSim 渲染线程弄崩**，
`accelerated_policy.py` 里已强制 `thread_local`，别去掉。

在仿真里排崩点一律 `PYTHONUNBUFFERED=1`（SIGABRT 不冲刷 stdout，日志会缺后半截，8-20 那次
就是这样误判成 `gym.make()` 崩的）；要拿到 `evaluation_metrics.json` 必须
`EMBODICHAIN_SIM_EXIT_PROCESS=0`，否则 `env.close()` 的 `os._exit(0)` 跑在写指标之前。
多卡机（A100 ×8）上**不要用 `CUDA_VISIBLE_DEVICES`**（Vulkan 不认，会和 CUDA 选到不同物理卡），
用 `--gpu_id N`；`select_cuda_device` 会把 JAX 也限制到这张卡，否则它在每张卡上预留 75% 显存。

## 已并入的 `feat/parallel-eval`

`scripts/eval_policy_parallel.py` 带两种并行：`--num_shards K --shard_index i`（多进程分片，种子序列与串行
一致，可跨 GPU）和 `--num_envs N`（单进程多环境 wave 批次，需 `--device cuda`，适配器必须能吃
批量观测）。**pi0.5 适配器（含本分支的 realtime_vla 后端）没有批量化，只能用分片**；
`env.close()` 已挪到指标落盘之后。完整口径见 `docs/parallel_eval.md`。

## 环境分层（最容易踩的坑）

**仓库刻意不提供统一环境**，仿真侧和训练侧的 torch/JAX 版本直接冲突，不要试图合并：

| 环境 | 位置 | 用途 |
|---|---|---|
| 仿真/采集/评估 | 仓库根 `.venv`（uv，Python 3.11 钉死） | `scripts/run_env_seeded.py`、`scripts/eval_policy_parallel.py`、`launch/*.sh` |
| 策略训练 | `policy/{act,dp,pi0,pi05}/`（各自 pyproject + uv.lock，已提交） | `finetune.sh` 内部走 `uv run --frozen`，首次运行自动建环境 |
| 重型策略 | `policy/{g05,motus,xr1}/` 的 `setup_env.sh` / README | 含 flash-attn、DeepSpeed 等自编译组件，不进 uv |

**本 worktree 没有自己的 `.venv`**，也没有建过 `policy/pi05/.venv`——上面说的加速命令用的是
`main` worktree 那份。

评估时两边碰头的手法是：**把策略源码目录拼进 `PYTHONPATH` 而不是安装它**
（见 `policy/pi05/eval.sh` 与 `scripts/eval_policy_parallel.py:add_policy_dependency_paths`），
从而在仿真环境里调用训练侧代码，绕开版本冲突。这是有意设计，不要「顺手修好」成正常安装。

路径基准（仓库里不写机器绝对路径）：`REPO_ROOT` 由 `BASH_SOURCE`/`__file__` 上溯，
`WS_ROOT=dirname $REPO_ROOT`，`EMBODICHAIN_ROOT=$WS_ROOT/EmbodiChain`、`MODELS_ROOT=$WS_ROOT/models`，
均可用环境变量覆盖。YAML 里的路径一律**相对仓库根**，所以对应命令必须从仓库根执行。
EmbodiChain 必须与本仓库**同级 clone**。

## 常用命令

```bash
bash launch/_print_available_tasks.sh          # 任务清单

# 评估（统一入口，各 policy 的 eval.sh 只是包装）
python scripts/eval_policy_parallel.py --config policy/<name>/deploy_policy.yml \
    --overrides --task_name click_bell --setting random --max_episodes 100
cd policy/pi05 && bash eval.sh <task> <setting> <train_config> <model_name> <gpu_id>

# 加速链路（详见 policy/pi05/realtime_vla/README.md）
python -m policy.pi05.realtime_vla.convert_checkpoint --jax-path <ckpt> --output <pkl> --prompt "..."
python -m policy.pi05.realtime_vla.benchmark_e2e ...
python -m policy.pi05.realtime_vla.validate_outputs ...
```

各 `finetune.sh` 头部注释就是该策略的完整用法/超参手册，改训练前先读它。

## 架构要点

**任务 → 配置 → 环境**。`robosynchallenge/tasks/<task>/` 定义环境与专家策略，
`configs/<task>/<setting>/{gym_config.json,action_config.json}` 定义场景/相机/随机化与专家轨迹参数。
`action_config.json` 通常在任务根目录，找不到才回落到 setting 子目录（`find_action_config`）。
setting：`clear`（无随机化，日常验证）、`random`（官方评测口径）、`random_3p`（random + 仅录像用第三视角，
**不进模型观测**）。评测任务只有 `_print_available_tasks.sh` 列出的 10 个。

`robosynchallenge/managers/{actions,datasets,events,observations}.py` 通过把模块名 append 进
`gym_utils.DEFAULT_MANAGER_MODULES` 注入 EmbodiChain 的 functor 注册表——这就是 gym_config 里
自定义 functor 名字能被解析的原因。新增 manager 必须同步这个列表（`scripts/eval_policy_parallel.py` 顶部有一份）。

**策略适配契约**。`scripts/eval_policy_parallel.py` 用 `importlib.import_module(f"policy.{name}")` 动态加载，
要求包顶层导出 `get_model(usr_args) / eval(env, model, obs) / reset_model(model)`
（通常靠 `policy/<name>/__init__.py` 里 `from .deploy_policy import *`）。
`eval` 返回 `(obs, info, truncated, inference_times)`，旧的 3 元组也兼容。
计时统一用 `policy/inference_timing.py`（**带 cuda synchronize**）——本分支做延迟测量尤其别绕开它，
不同步的计时会把 kernel 排队时间算漏。

适配器**自己驱动环境**：`eval()` 内部把整个 action chunk 逐个 `env.step()`，每步检查
`is_task_success()` 与 `truncated.any()` 提前 break，返回最后一次的四元组。
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
语言指令不走 yml：`eval_policy.py` 从 gym_config 递归提取 instruction 写到 `env._current_instruction`，
适配器首次调用时读它并 `model.set_language(...)`——加速路径要保持这个时序，prompt 是转换时就烘进权重的。

**配置与 overrides**。`deploy_policy.yml` 是唯一配置源，命令行 `--overrides` 后面是
`--key value` 成对的 REMAINDER，值会被 `eval()` 尝试解析。`policy_name` / `task_name` / `setting` 必填。
每 episode 步数上限取 `max(deploy_config.max_steps, gym_config.max_episode_steps)`，不是 yml 里那个值。

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

**成功判定只认官方版**。评测口径 = 每步调用 `env.get_wrapper_attr("is_task_success")()`，
且必须未 truncated。`compute_task_state()` 返回的 success 位在多数任务里被刻意置 0，**不能拿来算成功率**；
`XxxTestEnv` 变体的 `is_task_success` 恒为 True，只供目视/采集。
`robosynchallenge/tasks/` 与 `origin/main` 逐字节一致，改这里等于改判据——不要动。

## 仓库约定

- **分支分三类**：`main` 是**测评分支**（各实验分支合流后跑评测的地方）；
  `official/main` 跟踪 `origin/main`（主办方 EDEM-AI），只负责把上游拉进来；
  `sim-recap`、`feat/rtc-async-pi05`、`feat/realtime-vla-pi05`（本分支）、`ppo-post-training`
  是实验分支，各管一摊。README 里 `<!-- branch-readme:begin/end -->` 之间那段是
  **每个分支各自维护**的分支说明，不要当冲突合掉。
- **各分支已开 worktree**，`git worktree list` 是权威清单。要改别的线就 `cd` 过去，
  别在本目录 `git checkout`（会报 already checked out）。各 worktree 有各自的 CLAUDE.md 与环境状态。
- 远端：`origin` = 主办方 EDEM-AI（只读上游），`mine` = 个人 fork（推这里）。
- 大产物一律 gitignore：`lerobot_dataset/`、`training_data/`、`eval_result/`、`/report/`、
  `outputs/`、`checkpoints`。基准结论要留档就写进 `policy/pi05/realtime_vla/RESULTS.md` 或 `docs/`。
- 公开仓库卫生：内网 SSH 端点、私有 pip 源地址一律不入库，只写「需分发权限」。
- `policy/{g05,motus,xr1}` 的上游是 git submodule，clone 后需 `git submodule update --init`。
- **没有 lint/format 门禁**：`pyproject.toml` 的 `[tool.black]` 是个空节，没有 ruff/pre-commit 配置。
  别去找「跑一下 lint」的命令，按周围代码风格写即可。
- `scripts/README.md` 和 `launch/README.md` 是逐脚本的用途/参数手册，先查表再读源码。
