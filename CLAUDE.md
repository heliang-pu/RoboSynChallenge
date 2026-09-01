# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RoboSynChallenge：基于 EmbodiChain 的双臂（CobotMagic，14-DoF）操作挑战赛仓库——10 个仿真任务、
专家数据采集流水线、多个策略的训练/部署接入，以及统一的评估协议。
面向使用者的完整说明在 [README.md](README.md)、[SETUP.md](SETUP.md)、`docs/`；本文件只记录
「读多个文件才能看出来」的东西。

**当前 worktree = `sim-recap`，实验分支。** 主题是把 RECAP 搬进仿真：用仿真 rollout 自举价值函数，
把 advantage 写回数据集，供 pi0.5 做 ACP 微调。分支分工见文末。

> 这个 worktree 里**没有 `tests/`，也没有仓库根 `.venv`**——两者都只在 `main` 上。
> 要跑测试或用仿真环境，去 `../RoboSynChallenge`。别照搬 `main` 的 CLAUDE.md，
> 那边有 `policy/pi05_lerobot`、`policy/lila_wam` 等这条分支上不存在的目录。

## 本分支：sim-RECAP 闭环

九步流水线在 `launch/recap/`，一键版 `launch/run_sim_recap_round.sh`，手册
`docs/tutorials/sim_recap.md`，另有项目级 skill `.claude/skills/sim-recap/`：

```
01_rollout      策略 rollout（评估器自动打成败标签）
02_set_label    写/修边车标签
03_build_pool   合并专家池与 rollout 池
04_value_train  价值函数训练（third_party/evo_rl 收编的 pistar06）
05_value_qc     质检选 checkpoint
06_publish      发布 LeRobot v2.1 数据集（reward / no-reward 两版）
07_acp_finetune ACP 微调 pi0.5
08_eval         评估
09_bake_prompt  烘 prompt
```

`_common.sh` 是所有步骤的公共前导（`WORK_ROOT` / `REPO` / `PY_SIM` 等），改路径约定要动它而不是逐个脚本。

**质检子集的构成是写死的**：`05_value_qc.sh` 取 10 个 rollout 成功集 + 30 个 rollout 失败集
+ 20 个专家集，按 rollout 边车与合并池布局自动推导。比较成败分离度时别以为是随机抽样。

**侧录数据从哪来**：`eval_policy.py` 的 `rollout_save` / `rollout_save_path` 就是给这条链用的——
它会改写 gym_config 的 dataset functor（**保留失败集**，正常采集是丢掉的），并在数据集目录写
`episode_success.json` sidecar，供 `scripts/label_rollout_dataset.py` 消费。

**round1 的教训（结论留档，别重复踩）**：第一轮跑出来的 advantage 实质只是价值函数的残差漂移，
不是真实优势信号；positive 集里约 67% 来自失败 rollout。分析报告写在 gitignore 的 `report/` 下，
clone 不到——要引用就转写进 `docs/`。

**环境变量**：训练机地址不入库。`launch/recap/` 下的 `sync_*.sh` / `upload_*_to_hygon.sh` 需要
`RECAP_REMOTE=user@host`（另有 `RECAP_PORT` / `RECAP_KEY` / `RECAP_NAS`）；训练参数也走环境变量
（`RECAP_BATCH_SIZE`、`RECAP_NUM_TRAIN_STEPS`、`RECAP_FSDP_DEVICES`、`RECAP_CHECKPOINT_BASE_DIR`、
`RECAP_INDICATOR_KEY`、`RECAP_ACP_ENABLE` / `RECAP_ACP_DROPOUT` 等）。未 export 时脚本直接报错退出。

## 环境分层（最容易踩的坑）

**仓库刻意不提供统一环境**，仿真侧和训练侧的 torch/JAX 版本直接冲突，不要试图合并：

| 环境 | 位置 | 用途 |
|---|---|---|
| 仿真/采集/评估 | 仓库根 `.venv`（uv，Python 3.11 钉死） | `scripts/run_env.py`、`scripts/eval_policy.py`、`launch/*.sh` |
| 策略训练 | `policy/{act,dp,pi0,pi05}/`（各自 pyproject + uv.lock，已提交） | `finetune.sh` 内部走 `uv run --frozen`，首次运行自动建环境 |
| 重型策略 | `policy/{dm05,g05,motus,xr1}/` 的 `setup_env.sh` / README | 含 flash-attn、DeepSpeed 等自编译组件，不进 uv |

**本 worktree 没有自己的 `.venv`**，`_common.sh` 里的 `PY_SIM` 指向哪个解释器要先确认。

评估时两边碰头的手法是：**把策略源码目录拼进 `PYTHONPATH` 而不是安装它**
（见 `policy/pi05/eval.sh` 与 `scripts/eval_policy.py:add_policy_dependency_paths`），
从而在仿真环境里调用训练侧代码，绕开版本冲突。这是有意设计，不要「顺手修好」成正常安装。

路径基准（仓库里不写机器绝对路径）：`REPO_ROOT` 由 `BASH_SOURCE`/`__file__` 上溯，
`WS_ROOT=dirname $REPO_ROOT`，`EMBODICHAIN_ROOT=$WS_ROOT/EmbodiChain`、`MODELS_ROOT=$WS_ROOT/models`，
均可用环境变量覆盖。YAML 里的路径一律**相对仓库根**，所以对应命令必须从仓库根执行。
EmbodiChain 必须与本仓库**同级 clone**。

## 常用命令

```bash
bash launch/_print_available_tasks.sh          # 任务清单

# 采集：<task> <setting(clear|random)> <format(3_0|2_1)>
bash launch/run_task.sh click_bell clear 2_1 --max_episodes 100 --headless
python scripts/validate_lerobot_dataset.py ... # 首个失败门即非零退出，可当硬门禁

# 评估（统一入口，各 policy 的 eval.sh 只是包装）
python scripts/eval_policy.py --config policy/<name>/deploy_policy.yml \
    --overrides --task_name click_bell --setting random --max_episodes 100

# RECAP 一轮
bash launch/run_sim_recap_round.sh ...
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
自定义 functor 名字能被解析的原因。新增 manager 必须同步这个列表（`scripts/eval_policy.py` 顶部有一份）。

**策略适配契约**。`scripts/eval_policy.py` 用 `importlib.import_module(f"policy.{name}")` 动态加载，
要求包顶层导出 `get_model(usr_args) / eval(env, model, obs) / reset_model(model)`
（通常靠 `policy/<name>/__init__.py` 里 `from .deploy_policy import *`）。
`eval` 返回 `(obs, info, truncated, inference_times)`，旧的 3 元组也兼容。
计时统一用 `policy/inference_timing.py`（带 cuda synchronize），不要各写一份。
接新策略：抄 `policy/Your_Policy/`。

适配器**自己驱动环境**，不是「返回动作让上层执行」：`eval()` 内部把整个 action chunk 逐个
`env.step()`，每步检查 `is_task_success()` 与 `truncated.any()` 提前 break，返回最后一次的四元组。
语言指令不走 yml：`eval_policy.py` 从 gym_config 递归提取 instruction 写到 `env._current_instruction`。

**配置与 overrides**。`deploy_policy.yml` 是唯一配置源，命令行 `--overrides` 后面是
`--key value` 成对的 REMAINDER，值会被 `eval()` 尝试解析。`policy_name` / `task_name` / `setting` 必填。
每 episode 步数上限取 `max(deploy_config.max_steps, gym_config.max_episode_steps)`，不是 yml 里那个值。

**成功判定只认官方版**。评测口径 = 每步调用 `env.get_wrapper_attr("is_task_success")()`，
且必须未 truncated。`compute_task_state()` 返回的 success 位在多数任务里被刻意置 0，**不能拿来算成功率**；
`XxxTestEnv` 变体的 `is_task_success` 恒为 True，只供目视/采集。
`robosynchallenge/tasks/` 与 `origin/main` 逐字节一致，改这里等于改判据——不要动。
完整判据见 [docs/success_criteria.md](docs/success_criteria.md)：用旧几何版判定打过的标签
与官方口径不可直接对表——这条对 RECAP 尤其要命，成败标签是整条链的输入。

## 仓库约定

- **分支分三类**：`main` 是**测评分支**（各实验分支合流后跑评测的地方）；
  `official/main` 跟踪 `origin/main`（主办方 EDEM-AI），只负责把上游拉进来；
  `sim-recap`（本分支）、`feat/rtc-async-pi05`、`feat/realtime-vla-pi05`、`ppo-post-training`
  是实验分支，各管一摊。README 里 `<!-- branch-readme:begin/end -->` 之间那段是
  **每个分支各自维护**的分支说明，不要当冲突合掉。
- **各分支已开 worktree**，`git worktree list` 是权威清单。要改别的线就 `cd` 过去，
  别在本目录 `git checkout`（会报 already checked out）。各 worktree 有各自的 CLAUDE.md 与环境状态。
- 远端：`origin` = 主办方 EDEM-AI（只读上游），`mine` = 个人 fork（推这里）。
- 大产物一律 gitignore：`lerobot_dataset/`、`training_data/`、`eval_result/`、`/report/`、
  `outputs/`、`checkpoints`。报告类结论要留档就写进 `docs/`，别指望 `report/` 能被 clone 到。
- 公开仓库卫生：内网 SSH 端点、私有 pip 源地址一律不入库，只写「需分发权限」。
- `policy/{g05,motus,xr1}` 的上游是 git submodule，clone 后需 `git submodule update --init`。
- **没有 lint/format 门禁**：`pyproject.toml` 的 `[tool.black]` 是个空节，没有 ruff/pre-commit 配置。
  别去找「跑一下 lint」的命令，按周围代码风格写即可。
- `scripts/README.md` 和 `launch/README.md` 是逐脚本的用途/参数手册，先查表再读源码。
