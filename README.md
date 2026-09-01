<div align="center">
<h1>RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills</h1>

<h2 align="center"> 👉<a href="https://edem-ai.github.io/robosynchallenge.github.io/">Webpage</a> | <a href="https://edem-ai.github.io/RoboSynChallenge/html/">Document</a> | <a href="https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard">Leaderboard</a></h2>

![image](misc/robosynchallenge-pipeline.png)

</div>

---

<!-- branch-readme:begin -->

> **分支导航** — 本仓库按主题分支开发，每个分支的说明就在各自 README 的这个位置。
>
> **`main`（当前）** 测评 · [`official/main`](../../tree/official/main) 官方同步 · [`sim-recap`](../../tree/sim-recap) RECAP 价值函数 · [`feat/rtc-async-pi05`](../../tree/feat/rtc-async-pi05) 实时分块与异步执行 · [`feat/realtime-vla-pi05`](../../tree/feat/realtime-vla-pi05) 推理加速 · [`ppo-post-training`](../../tree/ppo-post-training) PPO 后训练

## 本分支：`main` — 测评分支

**跑评测的地方**，也是各条实验线的汇合点：实验分支做完合回这里，在统一口径下横向比较。
因此它的 policy 目录最全，`tests/` 与仓库根 `.venv` 也只在这条分支上。
共用的东西同样放这里：完整的中文项目说明、按需安装的 uv 训练环境、10 个任务的采集/训练/评估流程，
以及**回退到官方口径的成功判定**（本地改过的判定已全部撤回，避免与排行榜对不上）。

分支分三类：

| 分支 | 角色 |
|---|---|
| `main` | **测评分支**，实验分支合流后在这里跑评测 |
| `official/main` | **官方同步分支**，跟踪主办方 `EDEM-AI/RoboSynChallenge`，只负责把上游拉进来，不在上面开发 |
| `sim-recap` / `feat/rtc-async-pi05` / `feat/realtime-vla-pi05` / `ppo-post-training` | 实验分支，各管一摊 |

功能开发在实验分支上做，各自开了 worktree（`git worktree list` 是权威清单），
**每个 worktree 有各自的 `CLAUDE.md`**，按该分支实际有的东西写，不要跨目录照搬。

`feat/lila-wam` 与 `feat/lerobot-pi05-mem` 已于 2026-09-01 合入本分支并删除，内容如下。

### LiLa-WAM 接入与随机化覆盖度采集

- **LiLa-WAM 策略**：上游以 submodule 引入，附 deploy 入口、训练脚本、帧缓存、norm stats 与任务条件预计算（`policy/lila_wam/`，说明见 `docs/tutorials/policy/lila_wam.md`）
- **随机化覆盖度**：`scripts/analyze_random_coverage.py` 统计官方 random 配置的实际覆盖，`scripts/build_coverage_configs.py` 据此生成 93 组收窄范围的采集配置（分层补匀 + 缺口补采），产物在 `configs/<task>/coverage_*/`
- **约束随机化**：`randomize_rigid_object_pair_pose_constrained` 按 mesh 半尺寸保证成对物体的最小 xy 间隙；配套 6 个**直接调用生产判定函数**而非另写一份实现的测试
- **数据流水线**：种子化采集、v3.0→v2.1 转换、多视角质检、官方与自采数据合并（`scripts/`、`launch/`）
- **DM0.5 海光 DCU**：批大小探测、微调入口、checkpoint 回传与看门狗（`policy/dm05/`）

> 覆盖度配置的生成输入 `report/coverage/` 属本地产物不入库，因此生成的配置一并提交，否则 clone 后无法复现。

### LeRobot pi0.5（PyTorch）与 MEM 观测记忆

把 HuggingFace LeRobot 的 pi0.5 PyTorch 实现接成第二套 pi0.5 集成（`policy/pi05_lerobot/`），
与既有的 openpi/JAX 版 `policy/pi05` 并存，可在同一任务同一 setting 下直接对比。

- **跟最新上游**：锁定含 MEM 的 upstream commit（huggingface/lerobot#4076，短时程视觉与本体观测记忆），`setup_lerobot.sh` 会拒绝没有 `policies/pi05/memory.py` 的版本
- **MEM 决定执行节奏**：MEM 的历史环形缓冲每次策略调用只入队一帧，只有「每个环境步喂一次」才与训练一致。因此 eval 有 `per_step` / `chunk` 两种模式，带 MEM 的 checkpoint 默认走 `per_step`（实测 12 个环境步：`per_step` 喂满 12 帧，`chunk` 只喂 3 帧）
- **stride 跟数据集帧率**：`memory_stride` 以数据集帧计，本仓库数据 25fps 而上游默认 30，`finetune.sh` 直接从 `meta/info.json` 读 fps
- **独立 uv 环境 + worker 进程**：LeRobot 要求 Python ≥3.12 与 transformers v5，仿真侧钉死 3.11，装不进同一解释器；策略跑在自己的 uv 环境里，经 stdio JSON 与 `eval_policy.py` 通信
- **老 checkpoint 迁移**：过期 config 字段与写死的 tokenizer 路径这两类坑有现成解法，见策略 README

> 说明见 `policy/pi05_lerobot/README.md` 与 `docs/tutorials/policy/pi05_lerobot.md`。


---

<!-- branch-readme:end -->

# 目录

- [项目简介](#项目简介)
- [代码结构](#代码结构)
- [环境安装(uv 管理,按需安装)](#环境安装uv-管理按需安装)
- [任务与配置](#任务与配置)
- [数据:下载、采集与校验](#数据下载采集与校验)
- [训练](#训练)
- [评估](#评估)
- [支持的策略一览](#支持的策略一览)
- [评估结果](#评估结果)
- [更多文档](#更多文档)

# 项目简介

RoboSynChallenge 是基于 [EmbodiChain](https://dexforce.github.io/EmbodiChain/) 构建的
双臂机器人(CobotMagic,14-DoF)操作挑战赛,核心问题是:**在仿真中合成的操作技能,
能否泛化到域随机化乃至真实世界**。

本仓库提供完整的闭环工具链:

- **10 个双臂操作任务**的仿真环境与专家轨迹生成器(按低/中/高三档难度分级);
- **数据采集流水线**:专家演示 → LeRobot 数据集(v3.0/v2.1)→ 多重校验门 → 训练就绪;
- **10 个策略的训练/部署集成**:ACT、Diffusion Policy、pi0、pi0.5(openpi/JAX 与 LeRobot/torch 各一套)、
  DM0.5、G0.5、Motus、XR-1、LiLa-WAM,
  统一评估接口,任何策略实现 `deploy_policy.py` 即可接入;
- **标准化评估协议**:clear / random / random_3p 三种设置,固定判据与随机种子,结果可复现。

# 代码结构

```text
RoboSynChallenge/
├── robosynchallenge/     # 核心 Python 包:任务定义、管理器、域随机化、轨迹回放
│   ├── tasks/            #   10 个任务的环境与专家策略
│   ├── managers/         #   episode 管理、数据落盘
│   ├── Distractor/       #   干扰物资产与随机摆放
│   └── replay.py         #   轨迹回放
├── configs/              # 每任务一个目录:gym_config(场景/相机/随机化)+ action_config(专家动作)
│   └── <task>/{clear,random,random_3p,aug_*}/
├── launch/               # 数据采集/环境检查/可视化脚本(详见 launch/README.md)
├── scripts/              # 采集入口、评估入口、数据集工具(详见 scripts/README.md)
├── policy/               # 8 个策略的独立训练/部署环境(互不污染)
│   ├── act/  dp/         #   LeRobot 栈(uv 项目)
│   ├── pi0/  pi05/       #   openpi JAX 栈(uv 项目)
│   ├── dm05/ g05/ motus/ xr1/   # 各自的环境搭建脚本
│   ├── lila_wam/         #   LiLa-WAM(直接吃 LeRobot v2.1 训练)
│   └── Your_Policy/      #   接入自定义策略的模板
├── third_party/          # 收编的外部依赖(evo_rl:sim-RECAP 价值函数栈)
├── lerobot_dataset/      # 本地采集的数据默认落在这里
├── evaluation_results/   # 已发布 ACT/DP checkpoint 的百集评估结果(机器可读)
├── report/               # pi0.5 官方协议评估报告(random×100)
├── docs/                 # 完整教程(安装/采集/随机化/各策略训练)
├── SETUP.md              # 环境搭建速查(本页的展开版)
├── requirements.txt      # 仿真环境精确锁定依赖
└── pyproject.toml        # 根包定义 + 环境分层说明
```

# 环境安装(uv 管理,按需安装)

所有 Python 环境统一用 [uv](https://docs.astral.sh/uv/) 管理。
**本仓库刻意不提供"一把梭"的统一环境——用哪个策略就装哪个**,各策略环境互相独立、互不污染。

## 前置条件

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

按角色区分的额外前置:

| 你要做什么 | 需要什么 |
|---|---|
| 只训练策略 | 无。clone 本仓库即可,依赖全部来自公开源 |
| 仿真采集/评估 | ① [EmbodiChain](https://github.com/DexForce/EmbodiChain) clone 到**本仓库同级目录**;② `dexsim-engine` 闭源仿真引擎的私有 pip 源访问权限 |

目录布局约定(采集/评估用户):

```text
workspace/
├── EmbodiChain/          # 与本仓库同级
└── RoboSynChallenge/     # 本仓库
```

## 1. 仿真/采集/评估环境(仓库根目录)

```bash
cd RoboSynChallenge
uv venv                                       # .python-version 已钉死 3.11
source .venv/bin/activate
uv pip install -r requirements.txt            # 273 个包，全部来自公开 PyPI
uv pip install -e . --no-deps                 # 本仓库根包
uv pip install -e ../EmbodiChain --no-deps            # 仿真框架
uv pip install -e ../EmbodiChain/embodichain_tasks --no-deps

# 闭源引擎不在 requirements.txt 里，需单独拿 wheel（见 SETUP.md）
uv pip install dexsim_engine-0.4.3-cp311-*.whl
```

装好后可用 `bash launch/check_all_envs.sh` 依次拉起全部任务环境做冒烟检查。

## 2. 策略训练环境(按需,进哪个装哪个)

`policy/act`、`policy/dp`、`policy/pi0`、`policy/pi05`、`policy/pi05_lerobot`
都是**独立 uv 项目**,自带 `pyproject.toml` + `uv.lock`(已提交,保证可复现):

```bash
# 方式一:什么都不用装,直接跑训练脚本,首次运行自动按 lock 建环境
cd policy/pi05 && bash finetune.sh pi05_click_bell click_bell_v1 0

# 方式二:显式安装
cd policy/pi05
uv sync                # 只装训练依赖(公开源,任何人可装)
uv sync --extra sim    # 追加仿真评估依赖(需要 EmbodiChain 同级 clone + 私有源权限)
```

> 训练脚本内部用 `uv run --frozen`,严格按仓库自带的 uv.lock 安装,
> 没有 EmbodiChain 和私有源权限也能正常训练。
> 若你修改了某个 policy 的 `pyproject.toml`,需在有上述前置条件的机器上重新 `uv lock`。

> `policy/pi05_lerobot` 是唯一没有 `sim` extra 的:上游 lerobot 要 Python >=3.12,
> 而仿真侧钉死 3.11,两者装不进同一个解释器。它的评估靠跨进程——`eval_policy.py`
> 在仿真环境里跑,策略本体由 worker 在这个 uv 环境里跑,`eval.sh` 会自动把
> `policy/pi05_lerobot/.venv/bin/python` 传给 worker。

其余策略(dm05 / g05 / motus / xr1)依赖树庞大且含自编译组件(flash-attn、
DeepSpeed 等),各自目录下的 `setup_env.sh` 或 README 说明了环境搭建方式。

# 任务与配置

## 任务列表(按难度分级)

| 难度 | 任务 | 说明 |
|---|---|---|
| Low | `click_bell` | 按响桌铃 |
| Low | `handle_basket` | 双臂搬运提篮 |
| Low | `water_pouring` | 倒水 |
| Low | `table_rearrangement` | 桌面整理 |
| Mid | `items_handover` | 双臂物品交接 |
| Mid | `drawer_open_place` | 开抽屉并放入物品 |
| Mid | `mixer_operating` | 操作搅拌机 |
| High | `item_assembly` | 物品装配 |
| High | `manipulate_pipette` | 移液枪操作 |
| High | `sample_loading` | 样品装载 |

运行 `bash launch/_print_available_tasks.sh` 可随时查看任务清单。

## 配置目录

每个任务在 `configs/<task>/` 下有若干套配置,每套包含
`gym_config.json`(场景、相机内外参、光照/材质/物体位姿随机化)和
`action_config.json`(专家轨迹参数):

| 设置 | 含义 |
|---|---|
| `clear` | 无域随机化,固定场景 |
| `random` | 官方域随机化协议(评测口径) |
| `random_3p` | `random` + 仅录像用的第三视角相机(**不进入模型观测**,用于评估录像) |
| `aug_base` / `aug_near` / `aug_farright` | 相机视角增广采集配置 |

域随机化效果可用 `bash launch/run_visualize.sh <task>`(单任务)或
`bash launch/batch_run_visualize.sh`(全任务)可视化。详见
[docs/tutorials/domain_randomization.md](docs/tutorials/domain_randomization.md) 与
[docs/tutorials/configuration.md](docs/tutorials/configuration.md)。

# 数据:下载、采集与校验

## 直接下载

每任务 1000 条预采集轨迹托管在 HuggingFace:[数据集入口](https://edem-ai.github.io/robosynchallenge.github.io/#/data),
下载方法见 [docs/tutorials/download_data.md](docs/tutorials/download_data.md)。

## 自行采集

```bash
# 基础采集:<task> <setting> <format>,format 为 3_0 或 2_1(自动转换)
bash launch/run_task.sh click_bell clear 2_1 --max_episodes 100 --headless

# clear + random 混合采集并自动合并
bash launch/collect_combined_dataset.sh <task> ...

# 带校验门的采集(推荐,产出即训练就绪):
bash launch/collect_validated_batch.sh ...      # 单批采集,全部校验门通过才晋级
bash launch/collect_until_valid.sh ...          # 反复产出候选批直到整批通过校验
bash launch/collect_parallel_validated.sh ...   # 多分片并行采集→逐片校验→合并→再校验
```

数据默认写入 `lerobot_dataset/<task>/`。常用选项:`--max_episodes N`、`--headless`、
`--filter_visual_rand`(关视觉随机化)、`--filter_dataset_saving`(只跑不存盘)。
完整说明见 [launch/README.md](launch/README.md)。

## 数据集工具(scripts/)

| 脚本 | 用途 |
|---|---|
| `validate_lerobot_dataset.py` | 训练视角的严格校验(v2.1/v3.0),首个失败门即非零退出,可作采集流水线的硬门禁 |
| `convert_lerobot3.0_to_2.1.py` | LeRobot v3.0 → v2.1 格式转换 |
| `prepare_sim_real_cotrain.py` | 把 Real 数据对齐到 Sim 的 schema(fps/维度/键名)并合并,用于 sim-real 共训 |
| `add_lerobot_eef_pose.py` | 为数据集追加末端执行器位姿特征 |
| `camera_extrinsics_to_lootat.py` | 相机标定外参 → EmbodiChain eye/target/up 配置片段 |
| `visualize_distribution.py` | 数据分布可视化 |

轨迹回放:`bash launch/replay_task.sh`(支持 kinematic / dynamic / control 三种模式)。
各脚本详细用法见 [scripts/README.md](scripts/README.md)。

# 训练

每个策略目录下都有一个注释详细的 `finetune.sh`(用法、超参、数据格式、断点续训
都写在头部注释里),开箱即用:

```bash
# ACT / Diffusion Policy(从零训练,单张消费级显卡即可)
cd policy/act && bash finetune.sh <dataset_root> outputs/train/act_click_bell 0
cd policy/dp  && bash finetune.sh <dataset_root> outputs/train/dp_click_bell 0

# pi0 / pi0.5(官方基座全量微调,JAX 栈)
cd policy/pi0  && bash finetune.sh pi0_base_robosynchallenge_full my_exp 0
cd policy/pi05 && bash finetune.sh pi05_click_bell my_exp 0
# pi0.5 另有 10 个任务的一键脚本(内置配置名与数据检查):
cd policy/pi05/train_scripts && ./train_click_bell.sh 0

# DM0.5(OpenDM SFT,独立环境,见 policy/dm05/README.md)
cd policy/dm05 && bash finetune.sh <dataset_name> <num_gpus>

# G0.5 / Motus / XR-1(大显存机器,细节见各自 finetune.sh 头部注释)
bash policy/g05/finetune.sh 8 cobotmagic
bash policy/motus/finetune.sh 8
bash policy/xr1/finetune.sh <training_data_dir> <exp_name> 0
```

训练数据放到对应 policy 的 `training_data/` 下,例如
`policy/pi05/training_data/RoboSynChallenge/cobotmagic_Sim_click_bell/`。

各策略的图文教程:[docs/tutorials/policy/](docs/tutorials/policy/)
(act / dp / pi0 / pi05 / motus 各一篇)。

# 评估

评估在仿真环境里运行(需要 `--extra sim` 或根环境)。统一入口是
`scripts/eval_policy.py --config policy/<name>/deploy_policy.yml`,
每个策略目录下的 `eval.sh` 已做好包装:

```bash
# 示例:pi0.5
cd policy/pi05
uv sync --extra sim
bash eval.sh <task_name> <setting> <train_config> <model_name> <gpu_id>
# 例:bash eval.sh click_bell random pi05_click_bell my_exp 0 --max_episodes 100

# DM0.5 是服务式推理:先起模型服务,再在仿真环境跑评估
bash launch/run_dm05_server.sh <checkpoint_dir>
cd policy/dm05 && bash eval.sh <task_name> <setting> ...
```

`<setting>` 对应 `configs/<task>/<setting>/`:日常验证用 `clear`,
正式评测用 `random`(官方口径,每 10 步重随机化),
需要评估录像时用 `random_3p`。

## 接入自己的策略

复制 `policy/Your_Policy/` 模板,实现 `deploy_policy.py` 中的
`get_model / encode_obs / eval` 接口并填写 `deploy_policy.yml`,即可复用统一评估
流程。详见 [docs/tutorials/policy/your_own_policy.md](docs/tutorials/policy/your_own_policy.md)。

# 支持的策略一览

| 策略 | 类型 | 训练栈 | 训练显存需求 | 环境 |
|---|---|---|---|---|
| ACT | 轻量模仿学习(CVAE+Transformer) | LeRobot / torch | ~8 GB | uv(policy/act) |
| Diffusion Policy | 扩散策略 | LeRobot / torch | ~10 GB | uv(policy/dp) |
| pi0 | VLA(PaliGemma 3B + action expert) | openpi / JAX | 全量 ~80 GB,可调小 batch | uv(policy/pi0) |
| pi0.5 | VLA(pi0 升级版,开放世界泛化) | openpi / JAX | 全量 ~80 GB,可调小 batch | uv(policy/pi05) |
| pi0.5 (LeRobot) | 同上的 PyTorch 移植,带 MEM 观测记忆 | LeRobot / torch | 单卡 48 GB(bf16 + 梯度检查点) | 独立 Python(见 policy/pi05_lerobot) |
| DM0.5 | VLA(服务式推理) | OpenDM / torch | 多卡 SFT | conda(见 policy/dm05) |
| G0.5 | VLA(2B VLM + action expert) | GalaxeaVLA / torch | >70 GB/卡,官方 8 卡 | venv(见 policy/g05) |
| Motus | VLA(视频生成先验) | Motus / DeepSpeed | >80 GB/卡 | venv(见 policy/motus) |
| XR-1 | VLA(5.5B) | Xiaomi-Robotics / torch | 冻结 VLM 可单卡 48 GB | venv(见 policy/xr1) |
| LiLa-WAM | 世界-动作模型(冻结 DINOv3 + 0.2B DiT) | LiLa-WAM / torch | 单卡 24 GB | venv(见 policy/lila_wam) |

# 评估结果

## pi0.5(官方初赛协议,random × 100 集/任务)

完整报告与逐任务分析见 [report/README.md](report/README.md),原始数据在 `report/results.csv`:

| 任务 | 成功率 | ACT 基线 | DP 基线 |
|---|---|---|---|
| mixer_operating | **85%** | 77% | 69% |
| water_pouring | **80%** | 72% | 33% |
| items_handover | **79%** | - | - |
| table_rearrangement | **77%** | 63% | 16% |
| click_bell | **73%** | 37% | 44% |
| manipulate_pipette | **71%** | - | - |
| item_assembly | 16% | - | - |
| sample_loading | 3% | - | - |

## 已发布 ACT / DP checkpoint

100 集仿真评估结果见 [evaluation_results/README.md](evaluation_results/README.md),
机器可读文件固定了成功率、动作步数、推理耗时、HuggingFace checkpoint 版本与
`random` 协议配置。

完整榜单与评测设置:https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard

# 更多文档

| 文档 | 内容 |
|---|---|
| [SETUP.md](SETUP.md) | 环境搭建速查(仿真环境 + 训练环境的展开说明) |
| [docs/getting_started/](docs/getting_started/) | 安装、总览、代码结构 |
| [docs/tutorials/collect_data.md](docs/tutorials/collect_data.md) | 数据采集完整教程 |
| [docs/tutorials/task_trajectory.md](docs/tutorials/task_trajectory.md) | 任务轨迹生成原理 |
| [docs/tutorials/domain_randomization.md](docs/tutorials/domain_randomization.md) | 域随机化机制 |
| [docs/tutorials/configuration.md](docs/tutorials/configuration.md) | 配置文件字段说明 |
| [docs/tutorials/policy/](docs/tutorials/policy/) | 各策略训练/评估教程 |
| [docs/tutorials/sim_recap.md](docs/tutorials/sim_recap.md) | sim-RECAP:无人在回路的优势条件化迭代训练(价值函数 + ACP) |
| [launch/README.md](launch/README.md) | 采集/回放/可视化脚本手册 |
| [scripts/README.md](scripts/README.md) | 数据集工具手册 |
