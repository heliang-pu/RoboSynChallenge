# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RoboSynChallenge：基于 EmbodiChain 的双臂（CobotMagic，14-DoF）操作挑战赛仓库——10 个仿真任务、
专家数据采集流水线、多个策略的训练/部署接入，以及统一的评估协议。
面向使用者的完整说明在 [README.md](README.md)、[SETUP.md](SETUP.md)、`docs/`；本文件只记录
「读多个文件才能看出来」的东西。

**当前 worktree = `ppo-post-training`，实验分支。** 主题是把 pi0.5 从 JAX 搬到 PyTorch，
接进 [RLinf](https://github.com/RLinf/RLinf) 的 PPO / GRPO 训练回路。分支分工见文末。

> 这个 worktree 里**没有 `tests/`、没有仓库根 `.venv`、没有 `.claude/skills/`**，
> policy 目录也比 `main` 少（没有 `lila_wam` / `pi05_lerobot`）。别照搬 `main` 的 CLAUDE.md。
> 它还是个**嵌套 worktree**（位于 `main` 仓库的 `.claude/worktrees/ppo-posttrain`），
> 所以 `WS_ROOT` 上溯出来的路径和别的 worktree 不一样，写脚本时别假设同级目录布局。

## 本分支：RLinf PPO 后训练

**先读 `docs/tutorials/rlinf_ppo.md`**——它不是概览，是这条线的全部踩坑记录：为什么要转 PyTorch、
checkpoint 转换与验证、给 RLinf 打补丁、奖励与终止怎么接、观测通路、并行吞吐实测，以及下面这几条。

| 位置 | 作用 |
|---|---|
| `robosynchallenge/rlinf_env/` | 接入层：`vla_env.py`（env 包装）、`dataconfig.py`（openpi DataConfig 注册）、`transforms.py` |
| `envs/rlinf/` | 训练环境冻结成 uv lock：`install_from_lock.sh` + `requirements.lock.txt` |
| `patches/` | EmbodiChain / RLinf 的本地修复以 patch 形式随仓库分发，不改上游 clone |
| `launch/rlinf_setup_env.sh` / `rlinf_train.sh` / `rlinf_bench_envs.sh` | 装环境 / 训练 / 环境吞吐基准 |
| `launch/run_sample_loading_until_22.sh` | sample_loading 的 rollout runner，带 deadline 与磁盘/显存看门狗 |

### 三个会静默出错的地方

**1. `norm_stats_path` 必须显式指到 openpi 的布局。**
yaml 里 `actor.model.openpi_data.norm_stats_path` 要指到
`<ckpt>/assets/RoboSynChallenge/cobotmagic_Sim_<task>`。RLinf 默认按 `<model_path>/<asset_id>/` 找，
openpi 的布局**多一层 `assets/`**，不指定直接 `FileNotFoundError`。
这个 yaml 块**只能放这一个键**，其余键会被原样塞进 openpi 的 DataConfig。
同时 `robosynchallenge/rlinf_env/dataconfig.py` 里 `register()` 的 `repo_id` 也要换成对应任务——
归一化统计量是**按任务算的**，用错任务会让策略输出整体偏移**且不报错**。

**2. `val_check_interval` 必须是 -1，训练进程内不能开评测。**
RLinf 的 env worker 在同一进程里同时构造 train 和 eval 两个环境实例，而 dexsim 引擎是**进程级单例**：
第二个实例建好后，第一个实例的 `get_qpos()` 从 (N,14) 变成 (N,28)（关节被重复注册），
rollout 会在 openpi 的 Normalize 处报 `shapes (28,) (14,)`。冒烟测试撞不到是因为它关了评测。
接入层在 `_wrap_obs` 里加了 `expected_state_dim`（默认 14）校验，撞上时直接报这个原因，
而不是让下游的广播错误糊过去。**评测用保存的 checkpoint 另起进程做。**

**3. `num_envs>1` 需要 EmbodiChain 的两处本地补丁**（见 `docs/tutorials/rlinf_ppo.md` 对应小节）：
原实现把两个索引张量逐元素配对，N=1 时 `[1]x[6]` 碰巧广播成功，N=4 直接 `IndexError`；
修法是先取行再取列。补丁在 `patches/`，不要直接改 EmbodiChain clone。

### 产物在哪

RLinf 每次启动建 `<RLINF_ROOT>/logs/<时间戳>-<配置名>/`，checkpoint 落在
`<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`
（**整模型含 value head，单个好几 GB**，注意磁盘）。RLinf 的 `get_model` 认这个布局——
把 `actor.model.model_path` 指到 `global_step_<N>` 目录即可续训或评测，不用再转格式。

### 环境变量

路径与机器地址一律不入库，全走环境变量，未 export 时脚本直接报错退出：
`RLINF_ROOT`、`RLINF_PYTHON` / `RLINF_VENV_PYTHON`、`RLINF_CONFIG_DIR`、`ROBOSYN_PATH`、
`ROBOSYN_TASK`、`ROBOSYN_PI05_TORCH_CKPT`；看门狗相关有 `RLINF_DEADLINE`、
`RLINF_KEEP_CHECKPOINTS`、`RLINF_MIN_DISK_FREE_GB`、`RLINF_MIN_FREE_GPU_MB`、`RLINF_PAUSE_PID`。

## 环境分层（最容易踩的坑）

**仓库刻意不提供统一环境**，仿真侧和训练侧的 torch/JAX 版本直接冲突，不要试图合并：

| 环境 | 位置 | 用途 |
|---|---|---|
| 仿真/采集/评估 | 仓库根 `.venv`（uv，Python 3.11 钉死） | `scripts/run_env.py`、`scripts/eval_policy.py`、`launch/*.sh` |
| 策略训练 | `policy/{act,dp,pi0,pi05}/`（各自 pyproject + uv.lock，已提交） | `finetune.sh` 内部走 `uv run --frozen`，首次运行自动建环境 |
| RLinf 训练 | `envs/rlinf/`（lock 文件 + `install_from_lock.sh`） | PPO 回路，与仿真侧解耦，走 `RLINF_VENV_PYTHON` |
| 重型策略 | `policy/{dm05,g05,motus,xr1}/` 的 `setup_env.sh` / README | 含 flash-attn、DeepSpeed 等自编译组件，不进 uv |

**本 worktree 没有自己的 `.venv`**，跑仿真侧的东西要先确认用的是哪个解释器。

评估时两边碰头的手法是：**把策略源码目录拼进 `PYTHONPATH` 而不是安装它**
（见 `policy/pi05/eval.sh` 与 `scripts/eval_policy.py:add_policy_dependency_paths`），
从而在仿真环境里调用训练侧代码，绕开版本冲突。这是有意设计，不要「顺手修好」成正常安装。

路径基准（仓库里不写机器绝对路径）：`REPO_ROOT` 由 `BASH_SOURCE`/`__file__` 上溯，
`EMBODICHAIN_ROOT`、`MODELS_ROOT` 均可用环境变量覆盖。YAML 里的路径一律**相对仓库根**，
所以对应命令必须从仓库根执行。EmbodiChain 必须与主仓库**同级 clone**
（注意本 worktree 是嵌套的，上溯不到那一层，用环境变量显式指）。

## 常用命令

```bash
bash launch/_print_available_tasks.sh          # 任务清单

# RLinf 侧
bash launch/rlinf_setup_env.sh                 # 按 lock 装训练环境
bash launch/rlinf_bench_envs.sh                # 环境并行吞吐基准（定 total_num_envs 的依据）
bash launch/rlinf_train.sh                     # 训练
bash launch/run_sample_loading_until_22.sh     # sample_loading rollout runner

# 评估（统一入口，各 policy 的 eval.sh 只是包装）
python scripts/eval_policy.py --config policy/<name>/deploy_policy.yml \
    --overrides --task_name click_bell --setting random --max_episodes 100
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

适配器**自己驱动环境**：`eval()` 内部把整个 action chunk 逐个 `env.step()`，每步检查
`is_task_success()` 与 `truncated.any()` 提前 break，返回最后一次的四元组。
语言指令不走 yml：`eval_policy.py` 从 gym_config 递归提取 instruction 写到 `env._current_instruction`。

**成功判定只认官方版**。评测口径 = 每步调用 `env.get_wrapper_attr("is_task_success")()`，
且必须未 truncated。`compute_task_state()` 返回的 success 位在多数任务里被刻意置 0，**不能拿来算成功率**；
`XxxTestEnv` 变体的 `is_task_success` 恒为 True，只供目视/采集。
`robosynchallenge/tasks/` 与 `origin/main` 逐字节一致，改这里等于改判据——不要动。
**本分支上不再使用本地改写的 `mixer_operating` / `sample_loading` 判定，已全部回退到主办方口径**——
PPO 的奖励直接建在这个判定上，换判定等于换奖励函数。
「成功率不对劲时第一个该查的地方」在 `docs/tutorials/rlinf_ppo.md` 有专节。

## 仓库约定

- **分支分三类**：`main` 是**测评分支**（各实验分支合流后跑评测的地方）；
  `official/main` 跟踪 `origin/main`（主办方 EDEM-AI），只负责把上游拉进来；
  `sim-recap`、`feat/rtc-async-pi05`、`feat/realtime-vla-pi05`、`ppo-post-training`（本分支）
  是实验分支，各管一摊。README 里 `<!-- branch-readme:begin/end -->` 之间那段是
  **每个分支各自维护**的分支说明，不要当冲突合掉。
- **各分支已开 worktree**，`git worktree list` 是权威清单。要改别的线就 `cd` 过去，
  别在本目录 `git checkout`（会报 already checked out）。各 worktree 有各自的 CLAUDE.md 与环境状态。
- 远端：`origin` = 主办方 EDEM-AI（只读上游），`mine` = 个人 fork（推这里）。
- 大产物一律 gitignore：`lerobot_dataset/`、`training_data/`、`eval_result/`、`/report/`、
  `outputs/`、`checkpoints`。报告类结论要留档就写进 `docs/`。
- 公开仓库卫生：内网 SSH 端点、私有 pip 源地址一律不入库，只写「需分发权限」。
- `policy/{g05,motus,xr1}` 的上游是 git submodule，clone 后需 `git submodule update --init`。
- **没有 lint/format 门禁**：`pyproject.toml` 的 `[tool.black]` 是个空节，没有 ruff/pre-commit 配置。
  别去找「跑一下 lint」的命令，按周围代码风格写即可。
- `scripts/README.md` 和 `launch/README.md` 是逐脚本的用途/参数手册，先查表再读源码。
