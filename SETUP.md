# RoboSynChallenge 环境搭建

## 总体结构：两个互相隔离的 Python 环境

这个项目**不能**用单一环境跑通。仿真侧和训练侧对 torch 的版本要求直接冲突：

| | 仿真 / 采集 / 评估 | 训练 (π0.5) |
|---|---|---|
| 环境 | conda `robosyn` | uv venv，位于 `policy/pi05/.venv` |
| Python | 3.11.15 | >=3.11,<3.12 |
| torch | **2.10.0+cu128** | **2.7.1** |
| JAX | 不装 | `jax[cuda12]==0.5.3` |
| 依赖声明 | `environment.yml` + `requirements.txt` | `policy/pi05/pyproject.toml` + `uv.lock` |
| 入口 | `scripts/run_env.py`、`scripts/eval_policy.py` | `policy/pi05/scripts/train.py` |

评估时两边会碰头：`policy/pi05/eval.sh` 的做法是把 openpi 的源码目录**拼进 `PYTHONPATH`** 而不是安装它，从而在仿真环境里调用策略代码，绕开版本冲突。这是有意设计，不要「顺手修好」。

## 目录布局

两个仓库必须是**同级目录**，`EmbodiChain` 以可编辑方式安装：

```
workspace/
├── EmbodiChain/          # 仿真框架 (github.com/DexForce/EmbodiChain)
└── RoboSynChallenge/     # 本仓库
```

`policy/pi05/eval.sh` 默认按 `EMBODICHAIN_ROOT=$WORKSPACE_ROOT/EmbodiChain` 定位，
放到别处需要显式导出该环境变量。

### 路径约定

仓库里不写任何机器绝对路径，统一按下面两个基准推导：

| 变量 | 含义 | 默认推导 |
|---|---|---|
| `REPO_ROOT` | 仓库根 | shell 用 `BASH_SOURCE` 上溯，Python 用 `__file__` 上溯 |
| `WS_ROOT` | 仓库父目录 | `dirname $REPO_ROOT` |
| `MODELS_ROOT` | 权重目录 | `$WS_ROOT/models` |
| `EMBODICHAIN_ROOT` | 仿真框架 | `$WS_ROOT/EmbodiChain` |

`models/`、`data/`、`outputs/`、`datasets/` 都与仓库**同级**：

```
workspace/
├── EmbodiChain/
├── RoboSynChallenge/     # 本仓库 = REPO_ROOT
├── models/               # = MODELS_ROOT
├── data/  outputs/  datasets/
```

四个变量都可以用环境变量覆盖。YAML 配置无法展开环境变量，里面的路径一律
**相对仓库根**（如 `../models/...`），因此那些配置对应的命令要从仓库根执行。

## 一、仿真环境

```bash
conda env create -f environment.yml
conda activate robosyn

# 三个本地包，必须可编辑安装（requirements.txt 里已剔除）
pip install -e ../EmbodiChain
pip install -e ../EmbodiChain/embodichain_tasks
pip install -e .
```

验证：

```bash
python -c "import embodichain, dexsim; print('ok')"
./launch/run_task.sh click_bell random 3_0 --max_episodes 1 --headless
```

### 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| NVIDIA 驱动 | >= 580.173.02 | 开发机实测版本 |
| CUDA | 12.8 | torch 2.10.0+cu128 自带，系统无需单独装 |
| Vulkan | 需可用 | 仿真渲染走 Vulkan，无 runtime 会在建环境时崩 |
| `dexsim-engine==0.4.3` | 需分发权限 | DexForce 闭源引擎，公开 PyPI 上没有 |

Python 版本必须钉死 3.11 —— `dexsim-engine` 只发布 cp311 wheel。

不要用 conda 装 torch 或科学计算栈：`dexsim-engine` 的二进制扩展是针对 pip 版
torch 的 ABI 编译的，混装 conda-forge 的 torch 会在 import 时报符号找不到。

## 二、训练环境

`policy/pi05` 是 vendored 的 [openpi](https://github.com/Physical-Intelligence/openpi)，
自带 `pyproject.toml` 和 `uv.lock`，用 uv 独立管理：

```bash
cd policy/pi05
uv sync            # 按 uv.lock 精确还原
```

### 预训练权重

权重不入版本库，统一放在**仓库外**的模型目录，开发机上是 `"$MODELS_ROOT"`：

```
models/
├── openpi-assets/checkpoints/pi05_base/params   # π0.5 底座（本项目用这个）
├── pi05_base/
├── pi0/
├── paligemma-tokenizer/
├── g05/  motus/  xr1/                           # 其他策略的权重
└── ...
```

`src/openpi/training/config.py` 里 `weight_loader` 默认指向
`gs://openpi-assets/checkpoints/pi05_base/params`。**无外网环境必须改成本地路径**，
否则训练会在加载权重时失败：

```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "$MODELS_ROOT/openpi-assets/checkpoints/pi05_base/params"
),
```

这里保留 `gs://` 作为提交进版本库的默认值，是为了不把某台机器的绝对路径固化到公开仓库里。
换机器时改这一行即可，或把模型目录软链到相同位置。

### 数据集路径解析

训练器按 `$HF_LEROBOT_HOME / <TrainConfig.data.repo_id>` 两段拼路径。
`finetune.sh` 把 `HF_LEROBOT_HOME` 固定为 `policy/pi05/training_data`，
而配置里 `repo_id="RoboSynChallenge/cobotmagic_Sim_click_bell"`，所以实际读的是：

```
policy/pi05/training_data/RoboSynChallenge/cobotmagic_Sim_click_bell
```

采集产物在 `lerobot_dataset/<task>/<name>`，两者对不上，需要软链接接线。

### 多卡

两条路径都支持，本仓库默认配置是单卡（`fsdp_devices=1`）：

```bash
# JAX（默认路径）—— 传多个卡号即可，无需改代码
./policy/pi05/finetune.sh pi05_base_robosynchallenge_full <exp_name> 0,1,2,3

# PyTorch DDP
cd policy/pi05
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

约束：`batch_size % 总卡数 == 0`，且 `总卡数 % fsdp_devices == 0`。
`fsdp_devices` 不是「用几张卡」，而是「拿几张卡切模型」——
网格形状为 `(总卡数 // fsdp_devices, fsdp_devices)`，两轴分别是数据并行和模型分片。
显存够就保持 1，切分要付 all-gather 的通信代价。

## 三、已知问题

### 采集超过一定集数会崩，并静默毁掉整批数据

`randomize_visual_material` 的材质复用路径显式排除了 `default_plane`：

```python
can_reuse = (not fallback_to_new
             and entity_cfg.uid != "default_plane"      # 地面永远进不来
             and isinstance(entity, (RigidObject, Articulation)))
```

于是地面每次随机化都新建材质，而清理调用的开关 `clean = fallback_to_new` 默认 False，
纹理只进不出。撞到 Vulkan 池上限 1024 时抛 `std::runtime_error` → `terminate` → SIGABRT。

**修法**：给各任务 `configs/*/random/gym_config.json` 的 `random_plane_material.params`
加一行 `"fallback_to_new": true`。对 `default_plane` 而言这个开关只翻转清理行为、
不改随机化逻辑（它本来就走 legacy 路径）。

阈值按**尝试次数**算而非保存成功的集数 —— `run_env` 会把失败的 attempt 丢弃重摇，
每次 attempt 照样烧纹理。所以专家成功率越低的任务崩得越早，实测 `item_assembly`
只跑 5 集就撞上了上限。

### 任何异常退出都会截断 parquet，但 meta 照常落盘

进程被 SIGABRT / SIGSEGV / OOM 杀掉时，parquet 的 footer 从未写出，
但 `meta/info.json` 和 `meta/stats.json` 已经正常写盘。结果是数据集
**看起来完好**（集数帧数视频俱全）实则 pyarrow 一读就报
`Parquet magic bytes not found in footer`。

采集后务必校验，判据是 parquet 文件首尾各 4 字节都必须是 `PAR1`：

```bash
python - <<'EOF'
import glob, json, os
import pandas as pd
for ds in glob.glob("lerobot_dataset/*/*/"):
    ij = os.path.join(ds, "meta", "info.json")
    if not os.path.exists(ij): continue
    info = json.load(open(ij))
    pqs = glob.glob(os.path.join(ds, "data", "**", "*.parquet"), recursive=True)
    ok = True
    for p in pqs:
        with open(p, "rb") as f:
            head = f.read(4); f.seek(-4, 2); tail = f.read(4)
        if head != b"PAR1" or tail != b"PAR1": ok = False
    rows = sum(len(pd.read_parquet(p)) for p in pqs) if ok else -1
    print(f"{'OK ' if ok and rows == info['total_frames'] else 'BAD'} {ds} "
          f"{info['total_episodes']} 集 parquet={rows} info={info['total_frames']}")
EOF
```

### clear 配置在 teardown 必定段错误

`clear` 配置跑完退出码 139，`random` 正常退出 0。触发条件是「视觉随机化没运行」
而非 clear 本身 —— random 加 `--filter_visual_rand` 同样崩。
后果是任何看退出码的自动化都会误判，且按上一条，退出异常就可能截断 parquet。

### 专家成功率在任务间差异极大

`run_env.py` 的 `_generate_until_saved_episode_target` 是「采到 N 个成功为止」的
**无界循环**，失败的 attempt 丢弃重摇。实测各任务成功率：

| 任务 | 尝试/成功 | 成功率 |
|---|---|---|
| drawer_open_place / manipulate_pipette / mixer_operating / table_rearrangement | 5/5 | 100% |
| handle_basket | 6/5 | 83% |
| items_handover | 9/5 | 56% |
| sample_loading | 17/5 | 29% |
| water_pouring | 38/0 | **0%** |

`water_pouring` 在 random 配置下专家根本完不成，采集会永远转下去。
批量采集务必给每个任务加 `timeout` 上限，否则一个任务能把并发槽位永久占死。

### 采集并发受内存约束而非核数

单个采集进程常驻约 5.5 GB。worker 数应按 `(可用内存GB - 8) / 6` 估算。
开太多会 OOM，而 OOM 会截断 parquet 毁掉整片数据 —— 见上文。
