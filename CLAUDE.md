# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RoboSynChallenge：基于 EmbodiChain 的双臂（CobotMagic，14-DoF）操作挑战赛仓库——10 个仿真任务、
专家数据采集流水线、10+ 个策略的训练/部署接入，以及统一的评估协议。
面向使用者的完整说明在 [README.md](README.md)、[SETUP.md](SETUP.md)、`docs/`；本文件只记录
「读多个文件才能看出来」的东西。

**当前 worktree = `main`，测评分支。** 各实验分支在自己的 worktree 里，各有一份 CLAUDE.md；
分支分工见文末「仓库约定」。这里是各条线合流后跑评测的地方，所以 policy 目录最全
（`act dm05 dp g05 lila_wam motus pi0 pi05 pi05_lerobot smolvla xr1`），
`tests/` 与仓库根 `.venv` 也只在这条分支上。

## 环境分层（最容易踩的坑）

**仓库刻意不提供统一环境**，仿真侧和训练侧的 torch/JAX 版本直接冲突，不要试图合并：

| 环境 | 位置 | 用途 |
|---|---|---|
| 仿真/采集/评估 | 仓库根 `.venv`（uv，Python 3.11 钉死，torch 2.10+cu128） | `scripts/run_env_seeded.py`、`scripts/eval_policy_parallel.py`、`launch/*.sh` |
| 策略训练 | `policy/{act,dp,pi0,pi05}/`（各自 pyproject + uv.lock，已提交） | `finetune.sh` 内部走 `uv run --frozen`，首次运行自动建环境 |
| LeRobot pi0.5 | `policy/pi05_lerobot/`（uv，**Python 3.12**，lerobot 锁到含 MEM 的 git rev） | 上游要 3.12 而仿真侧钉 3.11，装不进同一解释器，所以没有 `sim` extra |
| 重型策略 | `policy/{dm05,g05,motus,xr1,lila_wam}/` 的 `setup_env.sh` / README | 含 flash-attn、DeepSpeed 等自编译组件，不进 uv |

评估时两边碰头有两种手法，都是有意设计，不要「顺手修好」成正常安装：

1. **把策略源码目录拼进 `PYTHONPATH` 而不是安装它**（见 `policy/pi05/eval.sh` 与
   `scripts/eval_policy_parallel.py:add_policy_dependency_paths`），在仿真环境里直接调训练侧代码。
   适用于纯 Python、且能跑在仿真侧 3.11 上的策略。
2. **跨进程**：Python 版本本身就冲突时（`policy/pi05_lerobot` 要 3.12），策略跑在自己的
   解释器里，与 `eval_policy.py` 用 stdio JSON 通信（`smolvla_worker.py`、`pi05_worker.py`）。

路径基准（仓库里不写机器绝对路径）：`REPO_ROOT` 由 `BASH_SOURCE`/`__file__` 上溯，
`WS_ROOT=dirname $REPO_ROOT`，`EMBODICHAIN_ROOT=$WS_ROOT/EmbodiChain`、`MODELS_ROOT=$WS_ROOT/models`，
均可用环境变量覆盖。YAML 里的路径一律**相对仓库根**，所以对应命令必须从仓库根执行。
EmbodiChain 必须与本仓库**同级 clone**。

## 常用命令

```bash
# 任务清单
bash launch/_print_available_tasks.sh

# 采集：<task> <setting(clear|random)> <format(3_0|2_1)>
bash launch/run_task_seeded.sh click_bell clear 2_1 --max_episodes 100 --headless
bash launch/collect_validated_batch.sh ...     # 带校验门，产出即训练就绪
python scripts/validate_lerobot_dataset.py ... # 首个失败门即非零退出，可当硬门禁

# 评估（统一入口，各 policy 的 eval.sh 只是包装）
python scripts/eval_policy_parallel.py --config policy/<name>/deploy_policy.yml \
    --overrides --task_name click_bell --setting random --max_episodes 100
cd policy/pi05 && bash eval.sh <task> <setting> <train_config> <model_name> <gpu_id>

# 训练
cd policy/act  && bash finetune.sh <dataset_root> outputs/train/act_x 0
cd policy/pi05 && bash finetune.sh pi05_click_bell my_exp 0
cd policy/pi05/train_scripts && ./train_click_bell.sh 0   # 10 个任务各有一键脚本
```

各 `finetune.sh` 头部注释就是该策略的完整用法/超参手册，改训练前先读它。

## 测试

pytest 装在根 `.venv`，直接跑即可：

```bash
.venv/bin/python -m pytest tests -q --ignore=tests/test_rtc_pi05.py
.venv/bin/python -m pytest tests/test_seeded_collection.py -q   # 单文件
```

历史上这里必须写成 `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ...`，因为 ROS2 jazzy
把 `/opt/ros/jazzy/lib/python3.12/site-packages` 泄漏进 `PYTHONPATH`，pytest 插件自动加载会去
import `launch_testing` 并崩在 `No module named 'lark'`。**2026-09-01 已把本机 ROS2 整套卸掉**
（450 个 `ros-jazzy-*` + colcon/rosdep，`~/.bashrc` 的 fishros source 块也删了），不再需要这两个前缀。
在还装着 ROS 的机器（4090 / pro6000 等）上仍要带。

已知状态（截至 2026-09-01，不是你改坏的）：
- `test_rtc_pi05.py` 需要 JAX，根环境 collect 就报 `No module named 'jax'`（会中断整个 run，
  所以上面加 `--ignore`），只能在 `policy/pi05` 环境里跑；
- `test_task_success_conditions.py` / `test_success_dataset_replay.py` / `test_sample_loading_success.py`
  共 18 个失败，因为判定已全量回退到官方版（见下），这些测试还在调已被删除的 `_evaluate_task_state`
  与本地打点接口。它们全部来自 `f70a7ef`，是给**本地改过的判定**写的；判定恢复官方版后这批测试
  失去对象，应删除或按官方判定重写，不要靠改生产代码去让它们变绿；
- 绿的 17 个：`test_constrained_randomization.py`、`test_linked_table_object_randomization.py`、
  `test_seeded_collection.py`、`test_pi05_lerobot_adapter.py`（12 个，不需要 checkpoint）。

## 架构要点

**任务 → 配置 → 环境**。`robosynchallenge/tasks/<task>/` 定义环境与专家策略，
`configs/<task>/<setting>/{gym_config.json,action_config.json}` 定义场景/相机/随机化与专家轨迹参数。
`action_config.json` 通常在任务根目录，找不到才回落到 setting 子目录（`find_action_config`）。
setting：`clear`（无随机化，日常验证）、`random`（官方评测口径）、`random_3p`（random + 仅录像用第三视角，
**不进模型观测**）、`aug_*` / `coverage_*`（自制采集配置）。
评测任务只有 `_print_available_tasks.sh` 列出的 10 个（低/中/高三档）；`tasks/_other_tasks/` 里那几个
（`pour_water`、`open_pan` 等）是历史/派生环境，不进评测，别拿它们的判定当参考。

`robosynchallenge/managers/{actions,datasets,events,observations}.py` 通过把模块名 append 进
`gym_utils.DEFAULT_MANAGER_MODULES` 注入 EmbodiChain 的 functor 注册表——这就是 gym_config 里
自定义 functor 名字能被解析的原因。新增 manager 必须同步这个列表（`scripts/eval_policy_parallel.py` 顶部有一份）。

**策略适配契约**。`scripts/eval_policy_parallel.py` 用 `importlib.import_module(f"policy.{name}")` 动态加载，
要求包顶层导出 `get_model(usr_args) / eval(env, model, obs) / reset_model(model)`
（通常靠 `policy/<name>/__init__.py` 里 `from .deploy_policy import *`）。
`eval` 返回 `(obs, info, truncated, inference_times)`，旧的 3 元组也兼容。
计时统一用 `policy/inference_timing.py`（带 cuda synchronize），不要各写一份。
接新策略：抄 `policy/Your_Policy/`。

适配器**自己驱动环境**，不是「返回动作让上层执行」：`eval()` 内部把整个 action chunk 逐个
`env.step()`，每步检查 `is_task_success()` 与 `truncated.any()` 提前 break，返回最后一次的四元组。
这决定了「一次推理执行几步」是策略侧的自由度（pi0.5 的 `pi0_step` / `phase_action_chunks`、
pi05_lerobot 的 `per_step` vs `chunk`），也是接新策略最容易写错的地方。
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
适配器首次调用时读它并 `model.set_language(...)`。

**配置与 overrides**。`deploy_policy.yml` 是唯一配置源，命令行 `--overrides` 后面是
`--key value` 成对的 REMAINDER，值会被 `eval()` 尝试解析（所以 `true`/`10`/`[1,2]` 都能写，
字符串保持原样）。`policy_name` / `task_name` / `setting` 必填。
每 episode 步数上限取 `max(deploy_config.max_steps, gym_config.max_episode_steps)`，不是 yml 里那个值。
并行评估用 `num_shards` / `shard_index`：种子 rng 在所有分片里照常逐个抽取、只跳过不属于自己的
episode，所以 N 个分片合并后与「单进程跑满 max_episodes」是同一批种子——改这段循环会静默破坏可比性。

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
完整判据与历史回退说明见 [docs/success_criteria.md](docs/success_criteria.md)：用旧几何版判定打过的标签
与官方口径不可直接对表。

**sim-RECAP 闭环**。rollout（评估器自动打成败标签）→ 合并专家池 → 价值函数训练（`third_party/evo_rl`
收编的 pistar06）→ 质检选 checkpoint → 把 value/advantage/indicator 写回数据集 → 发布 v2.1
reward/no-reward 两版 → ACP 微调 pi0.5。脚本是 `launch/recap/01..09_*.sh`，
一键版 `launch/run_sim_recap_round.sh`，手册 `docs/tutorials/sim_recap.md`，
以及项目级 skill `.claude/skills/sim-recap/`。
`eval_policy.py` 的 `rollout_save` / `rollout_save_path` 就是给这条链侧录用的：它会改写
gym_config 的 dataset functor（保留失败集），并在数据集目录写 `episode_success.json` sidecar，
供 `scripts/label_rollout_dataset.py` 消费。

## 仓库约定

- **分支分三类，别混**：

  | 分支 | 角色 | 说明 |
  |---|---|---|
  | `main` | **测评分支（本分支）** | 跑评测的地方。实验分支做完合回这里，所以它是各条线的汇合点 |
  | `official/main` | **官方同步分支** | 跟踪 `origin/main`（主办方 EDEM-AI），只负责把上游拉进来，不在上面开发 |
  | `sim-recap` / `feat/rtc-async-pi05` / `feat/realtime-vla-pi05` / `ppo-post-training` | 实验分支 | 各管一摊，见下 |

  `feat/lila-wam` 与 `feat/lerobot-pi05-mem` 已于 2026-09-01 合入 `main` 并删除本地分支
  （远端 `mine/feat/lila-wam` 还停在 `21fd0d4`）。
  README 里 `<!-- branch-readme:begin/end -->` 之间的那段是**每个分支各自维护**的分支说明，
  切分支改这块，不要把它当冲突合掉。
- **实验分支已各开 worktree，不要在本目录 `git checkout` 它们**（会直接报 already checked out）：
  `../RoboSynChallenge-sim-recap`、`../RoboSynChallenge-realtime-vla`、
  `.claude/worktrees/ppo-posttrain`。要改哪条线就 `cd` 过去；`git worktree list` 是权威清单。
  **各 worktree 有各自的 CLAUDE.md**，内容按该分支实际有的东西写，不要跨目录照搬。
  注意它们都没有自己的 `.venv`，也没有 `tests/`（这两样目前只在 `main` 上）。
- 远端：`origin` = 主办方 EDEM-AI（只读上游），`mine` = 个人 fork（推这里）。
- **评估产物落盘**：评估视频一律写到 `/root/workspace/eval_results/`（评估机上的绝对路径，仓库外，不进 git），
  按 `<task>/<model>/<framework>/<step>/` 分目录存放——顶层固定是 10 个任务名文件夹（`_print_available_tasks.sh`
  的清单），往下是模型（如 `pi05`）/ 推理框架（如 `jax`、`realtime_vla`）/ checkpoint 步数（如 `28000`），
  视频、`evaluation_metrics.json`、`report.md` 直接放在步数目录里，setting 写进 report。**每次评估都要写评估报告**：至少含官方公式算出的 Overall Score（见「官方榜单打分公式」）、checkpoint/配置、
  `max_episodes` 与种子口径、官方判定下的成功率、逐 episode 成败、失败模式归类，报告与视频同目录
  （`report.md`）；结论要长期留档的再拷一份进 `docs/`。仓库内的 `eval_result/` 只是本机临时产物，别当归档。
- 大产物一律 gitignore：`lerobot_dataset/`、`training_data/`、`eval_result/`、`/report/`、
  `outputs/`、`checkpoints`。报告类结论要留档就写进 `docs/`，别指望 `report/` 能被 clone 到。
- 公开仓库卫生：内网 SSH 端点、私有 pip 源地址一律不入库，只写「需分发权限」。
  同步/训练脚本的机器地址走环境变量（如 `RECAP_SYNC_REMOTE`、`DM05_REMOTE_HOST`），未 export 时脚本直接报错退出。
- `policy/{g05,motus,xr1,lila_wam}` 的上游是 git submodule，clone 后需 `git submodule update --init`。
- **没有 lint/format 门禁**：`pyproject.toml` 的 `[tool.black]` 是个空节，没有 ruff/pre-commit 配置，
  `.github/` 只有一个 PR 模板。别去找「跑一下 lint」的命令，按周围代码风格写即可。
- `scripts/` 有 40 个脚本、`launch/` 有 30 多个：`scripts/README.md` 和 `launch/README.md`
  是逐脚本的用途/参数手册，先查表再读源码。`scripts/eval_policy_parallel.py.fixed` 是未入库的游离副本，
  真正的入口永远是 `scripts/eval_policy_parallel.py`。
