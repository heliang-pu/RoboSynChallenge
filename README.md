<div align="center">
<h1>RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills</h1>

<h2 align="center"> 👉<a href="https://edem-ai.github.io/robosynchallenge.github.io/">Webpage</a> | <a href="https://edem-ai.github.io/RoboSynChallenge/html/">Document</a> | <a href="https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard">Leaderboard</a></h2>

![image](misc/robosynchallenge-pipeline.png)

</div>

---

# 目录

- [仓库说明](#仓库说明)
- [环境安装(uv 管理,按需安装)](#环境安装uv-管理按需安装)
- [数据集](#数据集)
- [训练](#训练)
- [评估](#评估)
- [支持的策略一览](#支持的策略一览)
- [已发布 Checkpoint 结果](#已发布-checkpoint-结果)

# 仓库说明

本仓库基于 [EmbodiChain](https://dexforce.github.io/EmbodiChain/) 构建 RoboSynChallenge
双臂操作挑战赛的仿真环境,包含 10 个任务(click_bell、water_pouring、item_assembly 等)
的数据采集、策略训练与统一评估,并集成了 8 个策略的完整训练/部署链路:
**ACT、Diffusion Policy、pi0、pi0.5、DM0.5、G0.5、Motus、XR-1**。

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

目录布局约定(评估用户):

```text
workspace/
├── EmbodiChain/          # 与本仓库同级
└── RoboSynChallenge/     # 本仓库
```

## 1. 仿真/采集/评估环境(仓库根目录)

```bash
cd RoboSynChallenge
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt            # 精确锁定的仿真依赖(含私有源包)
uv pip install -e . --no-deps                 # 本仓库根包
uv pip install -e ../EmbodiChain --no-deps    # 仿真框架
```

## 2. 策略训练环境(按需,进哪个装哪个)

`policy/act`、`policy/dp`、`policy/pi0`、`policy/pi05` 都是**独立 uv 项目**,
自带 `pyproject.toml` + `uv.lock`(已提交,保证可复现):

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

其余策略(dm05 / g05 / motus / xr1)依赖树庞大且含自编译组件(flash-attn、
DeepSpeed 等),各自目录下的 `setup_env.sh` 或 README 说明了环境搭建方式。

# 数据集

每个任务提供 1000 条预采集轨迹,托管在 HuggingFace:[数据集入口](https://edem-ai.github.io/robosynchallenge.github.io/#/data)。

也强烈建议自行采集(可控制随机化设置):

```bash
# 单任务采集(仿真环境中)
bash launch/run_task.sh <task_name> <setting>
# 带验证的批量采集
bash launch/collect_validated_batch.sh <task_name> <setting> <episodes>
```

训练用的 LeRobot 数据集放到对应 policy 的 `training_data/` 下,例如:

```text
policy/pi05/training_data/RoboSynChallenge/cobotmagic_Sim_click_bell/
```

# 训练

每个策略目录下都有一个注释详细的 `finetune.sh`,用法开箱即用:

```bash
# ACT / Diffusion Policy(从零训练,单张消费级显卡即可)
cd policy/act && bash finetune.sh <dataset_root> outputs/train/act_click_bell 0
cd policy/dp  && bash finetune.sh <dataset_root> outputs/train/dp_click_bell 0

# pi0 / pi0.5(官方基座全量微调,JAX 栈)
cd policy/pi0  && bash finetune.sh pi0_base_robosynchallenge_full my_exp 0
cd policy/pi05 && bash finetune.sh pi05_click_bell my_exp 0
# pi0.5 也提供 10 个任务的一键脚本:
cd policy/pi05/train_scripts && ./train_click_bell.sh 0

# DM0.5(OpenDM SFT,独立环境,见 policy/dm05/README.md)
cd policy/dm05 && bash finetune.sh <dataset_name> <num_gpus>

# G0.5 / Motus / XR-1(大显存机器,细节见各自 finetune.sh 头部注释)
bash policy/g05/finetune.sh 8 cobotmagic
bash policy/motus/finetune.sh 8
bash policy/xr1/finetune.sh <training_data_dir> <exp_name> 0
```

超参、数据格式、输出目录、断点续训方式都写在各 `finetune.sh` 的头部注释里。
接入自己的策略请参考 `policy/Your_Policy/` 模板与[官方文档](https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html)。

# 评估

评估在仿真环境里运行(需要 `--extra sim` 或根环境),每个策略目录下有 `eval.sh`:

```bash
# 示例:pi0.5
cd policy/pi05
uv sync --extra sim
bash eval.sh <task_name> <setting> <train_config> <model_name> <gpu_id>
```

`<setting>` 支持 `clear` / `random` / `random_3p`(第三方随机化协议)等,
对应 `configs/<task>/<setting>/gym_config.json`。

# 支持的策略一览

| 策略 | 类型 | 训练栈 | 训练显存需求 | 环境 |
|---|---|---|---|---|
| ACT | 轻量模仿学习 | LeRobot / torch | ~8 GB | uv(policy/act) |
| Diffusion Policy | 扩散策略 | LeRobot / torch | ~10 GB | uv(policy/dp) |
| pi0 | VLA(3B) | openpi / JAX | 全量 ~80 GB,可调小 batch | uv(policy/pi0) |
| pi0.5 | VLA(3B) | openpi / JAX | 全量 ~80 GB,可调小 batch | uv(policy/pi05) |
| DM0.5 | VLA | OpenDM / torch | 多卡 SFT | conda(见 policy/dm05) |
| G0.5 | VLA(2B) | GalaxeaVLA / torch | >70 GB/卡,官方 8 卡 | venv(见 policy/g05) |
| Motus | VLA | Motus / DeepSpeed | >80 GB/卡 | venv(见 policy/motus) |
| XR-1 | VLA(5.5B) | Xiaomi-Robotics / torch | 冻结 VLM 可单卡 48 GB | venv(见 policy/xr1) |

# 已发布 Checkpoint 结果

已发布 ACT 与 Diffusion Policy checkpoint 的 100-episode 仿真评估结果见
[`evaluation_results`](evaluation_results/README.md),结果文件固定了成功率、
动作步数、推理耗时、HuggingFace checkpoint 版本与 `random` 协议配置。

完整榜单与评测设置:https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard
