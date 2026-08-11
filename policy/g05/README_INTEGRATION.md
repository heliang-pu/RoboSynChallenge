# G0.5 (星海图 Galaxea GalaxeaVLA) 接入 RoboSynChallenge

本目录把 [OpenGalaxea/GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) 的 G0.5 策略接到比赛的统一评估接口上。

> **一句话结论**：G0.5 的 14 维双臂 embodiment（左臂6 | 左夹爪1 | 右臂6 | 右夹爪1）
> 与比赛 `env.step` 的 `(1,14)` 绝对关节位置**逐位对应**，不需要手写 27→14 的重映射。
> 模型内部那个 27 维是 R1Pro 实体机的**分组布局**，由框架的 `GroupedPaddingMerger`
> 按 `parts_meta` 自动折叠/还原。
>
> **状态**：环境、数据、推理全链路已实跑通过。`g05-base` 零样本 1 集 `EXIT=0`
> （结果 `FAIL`，符合预期）。要拿成绩需微调，微调需 80G 卡。

---

## 0. 怎么用（可直接复制执行）

以下命令**全部在 `RoboSynChallenge` 仓库根目录**执行。

### 0.1 装环境（一次性，约 10~40 分钟）

```bash
cd "$REPO_ROOT"
bash policy/g05/setup_env.sh
```

一条命令做完四件事：`uv sync --frozen` 建 venv、补 torchcodec 缺的 `nvidia-npp-cu12`、
装 EmbodiChain 仿真栈（`eval_policy.py` 单进程同时要策略栈和仿真栈）、跑全部自检。

跑完自检应该全绿：

```
[2/3] 核心依赖自检          ok torch 2.7.1+cu128 ...
[2b/3] torchcodec 视频解码自检   ok torchcodec 原生库加载成功（av1 可解）
[2c/3] 仿真栈自检              ok scripts/eval_policy.py 可导入（embodichain / dexsim 就绪）
[3/3] g05 包 import 自检      ok g05.models.g05.inferencer.PolicyInferencer
```

然后跑冒烟测试（不需要权重，应该 7/7 全过）：

```bash
./policy/g05/GalaxeaVLA/.venv/bin/python policy/g05/smoke_test.py
```

> 装完**不要再单独跑 `uv sync`**：`--frozen` 会删掉所有不在 `uv.lock` 里的包，
> 整套仿真栈会被清掉。真清掉了就重跑 `setup_env.sh` 恢复。

### 0.2 准备权重

权重已下到 `"$MODELS_ROOT"/g05/`。软链进 GalaxeaVLA（配置里路径都是相对项目根的 `checkpoints/...`）：

```bash
cd "$REPO_ROOT"/policy/g05/GalaxeaVLA
mkdir -p checkpoints
ln -sfn "$MODELS_ROOT"/g05/action_tokenizer.pt          checkpoints/action_tokenizer.pt
ln -sfn "$MODELS_ROOT"/g05/g05-base                     checkpoints/g05-base
ln -sfn "$MODELS_ROOT"/g05/qwen3_5_2b_base_processor    checkpoints/qwen3_5_2b_base_processor
cd "$REPO_ROOT"
```

需要重新下载时（gated 仓库，必须先在 HF 网页同意协议，且只能走官方源 + token，hf-mirror 拿不到）：

```bash
huggingface-cli download OpenGalaxea/G05 --repo-type model \
    --local-dir "$MODELS_ROOT"/g05 \
    --include "g05-base/*" "qwen3_5_2b_base_processor/*" "action_tokenizer.pt"
```

### 0.3 准备训练数据

比赛数据 `lerobot_dataset/` 实测**已经是 LeRobot v3.0**，key 名和维度与 G0.5 完全一致，
**不需要转换数据本体**，只要生成配置：

```bash
./policy/g05/GalaxeaVLA/.venv/bin/python policy/g05/convert_lerobot_to_g05.py \
    scan lerobot_dataset --emit-config
```

产出 `policy/g05/configs/{data,task}/cobotmagic.yaml`。实测扫描到 10 个数据集 / 50 episodes / 12415 frames。

其他子命令：

```bash
# 单个数据集体检
./policy/g05/GalaxeaVLA/.venv/bin/python policy/g05/convert_lerobot_to_g05.py \
    inspect lerobot_dataset/handle_basket/cobotmagic_sim_handle_basket_000

# 万一拿到的是 v2.1 数据，迁移成 v3.0（视频软链，不复制）
./policy/g05/GalaxeaVLA/.venv/bin/python policy/g05/convert_lerobot_to_g05.py \
    migrate <v2.1目录> <输出目录>

# 转换器自测（造假数据，不依赖真实数据集）
./policy/g05/GalaxeaVLA/.venv/bin/python policy/g05/convert_lerobot_to_g05.py selftest
```

### 0.4 训练（微调）

```bash
bash policy/g05/finetune.sh 8 cobotmagic
```

调参示例：

```bash
bash policy/g05/finetune.sh 8 cobotmagic model.batch_size=8 model.max_epochs=5
```

> **本机 4090 跑不了全量微调**：官方要求 > 70 GB 显存（A100 80G / H20 96G），4090 只有 48 GB。
> 产物在 `policy/g05/outputs/cobotmagic/<exp_name>/`，checkpoint 文件名是 `step_<N>.pt`。

### 0.5 评估

```bash
bash policy/g05/eval.sh <task_name> <setting> [ckpt_path] [gpu_id] [额外参数...]
```

**零样本冒烟**（开箱即用，已实跑 `EXIT=0`）：

```bash
bash policy/g05/eval.sh click_bell clear \
    policy/g05/GalaxeaVLA/checkpoints/g05-base/checkpoints/model_state_dict.pt 0 \
    --max_episodes 1 --headless true
```

`deploy_policy.yml` 默认 `embodiment: robomindv2_Agilex`，借用 g05-base 里布局一致的
归一化统计量（原因见 8.1）。实测 1 集：38 次推理、平均 2.58 s/次、600 步、结果 `FAIL`。
**管线通了，但零样本做不出任务，要成绩得微调。**

**用微调产物评估**（推荐路径）：

```bash
bash policy/g05/eval.sh click_bell random \
    policy/g05/outputs/cobotmagic/<exp_name>/checkpoints/step_20000.pt 0 \
    --max_episodes 20
```

同时把 `policy/g05/deploy_policy.yml` 里这两项改成微调时用的 embodiment：

```yaml
sim_task: cobotmagic
embodiment: cobotmagic
```

评估显存官方推荐 4090（> 8 GB），本机够用。

---

## 1. 文件清单

| 文件 | 作用 |
|------|------|
| `deploy_policy.py` | 比赛接口三件套 `get_model` / `eval` / `reset_model`，含观测编码 |
| `g05_model.py` | **进程内**推理封装（绕开官方 WebSocket 服务），加载权重 / 组装 hydra 配置 / 解码动作 chunk |
| `deploy_policy.yml` | 评估配置（checkpoint 路径、replan_steps、精度等） |
| `eval.sh` | 评估入口 |
| `__init__.py` | `from .deploy_policy import *` —— 评估脚本按**包**导入策略，必须有 |
| `setup_env.sh` | 用官方 `uv sync` 建 `GalaxeaVLA/.venv` |
| `convert_lerobot_to_g05.py` | 数据体检 / 生成 G0.5 data+task 配置 / v2.1→v3.0 迁移 / 自测 |
| `smoke_test.py` | 不需要权重的冒烟测试（import / hydra 组装 / 切片落位 / av1 解码 / 真实加载路径） |
| `finetune.sh` | 微调入口，自动把自研配置软链进 GalaxeaVLA 的 configs 树 |
| `configs/{data,task}/cobotmagic.yaml` | 微调用的比赛 embodiment 定义 |
| `configs/{data,task}/robomindv2_Agilex.yaml` | 零样本用：同样的 14 维布局，但借用 g05-base 里已有的归一化统计量（见 8.1） |
| `GalaxeaVLA/` | 官方源码（git clone，未做任何修改） |

---

## 2. 进程内推理调用链

官方部署方式是 `scripts/serve_policy.py` 起 WebSocket 服务、客户端发 msgpack。
比赛评估脚本要求策略跑在同一进程里，所以这里直接复用官方的 `PolicyInferencer`：

```
deploy_policy.get_model(usr_args)
  └── G05.__init__                                        (g05_model.py)
        ├── _compose_cfg()
        │     hydra compose(configs/sim_robotwin.yaml, overrides=["task=robotwin"])
        │     → cfg.data（14 维 embodiment 定义）+ cfg.EVALUATION
        ├── _apply_checkpoint_model_config()
        │     读 <ckpt_run_dir>/.hydra/config.yaml
        │     → cfg.model / cfg.tokenizer（模型结构、action tokenizer 以权重为准）
        ├── instantiate(cfg.model.model_arch) + load_state_dict_safely()
        ├── build_processors(cfg) → MixtureProcessor{"robotwin": GalaxeaCoTProcessor}
        │     └── set_normalizer_from_stats(dataset_stats.json)
        └── PolicyInferencer(policy, processor, device)

deploy_policy.eval(env, model, obs)
  ├── encode_obs(obs)  三路相机 (1,H,W,4)→(H,W,3) + qpos (1,14)→(14,)
  ├── G05.update_observation_window()  → obs_dict
  ├── G05.get_action() → inferencer.infer([obs_dict])[0]
  │     → {left_arm:[T,6], left_gripper:[T,1], right_arm:[T,6], right_gripper:[T,1]}
  │     → 按 shape_meta 的 start_index 回填成 [T,14]
  └── 逐步 env.step((1,14))，执行 replan_steps 步后返回
```

`obs_dict` 的字段（照抄官方 `experiments/robotwin/galaxeafm_policy/deploy_policy.py`）：

```python
{
  "images": {cam_high|cam_left_wrist|cam_right_wrist: uint8 [T,3,H,W]},
  "state":  {left_arm:[T,6], left_gripper:[T,1], right_arm:[T,6], right_gripper:[T,1]},
  "task": "<语言指令>",
  "action": <全零占位>, "action_is_pad": <全 True>,
  "state_is_pad": [T]False, "image_is_pad": [T]False,
  "idx": 0, "frequency": 25.0,
  "embodiment": "robotwin",     # MixtureProcessor 靠这个选子 processor
}
```

---

## 3. 27 维语义与映射方案（核心结论）

### 27 维是什么

G0.5 的动作空间不是"27 个关节"，而是**分组槽位（grouped layout）**。
`g05-base` 权重的 `.hydra/config.yaml` 里 `action_dim: 27`，来自
`GroupedPaddingMerger` 的 `parts_meta` + `merge_spec`：

```yaml
parts_meta:  {left_arm: 7, right_arm: 7, left_gripper: 1, right_gripper: 1,
              left_ee_pose: 9, right_ee_pose: 9, lower_body: 7}
merge_spec:  {left_control:  [left_arm, left_ee_pose],    # 互斥，取先出现的，宽度 = max(7,9) = 9
              left_gripper:  [left_gripper],
              right_control: [right_arm, right_ee_pose],
              right_gripper: [right_gripper],
              lower_body:    [lower_body]}
```

打平后：

```
[ left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1) | lower_body(7) ] = 27
   0 ────────── 8   9              10 ─────────── 18   19             20 ────────── 26
```

`left_control` 是**互斥槽**：关节控制（`left_arm`）和末端位姿控制（`left_ee_pose`）共用，
谁在数据里出现就用谁，按 9 维对齐。所以"left_control(9) 是关节还是 EE 位姿"取决于 embodiment 声明了哪个 key。

### 比赛这边怎么落到 14 维

比赛 embodiment 只声明关节控制、不声明 `ee_pose` / `lower_body`：

```yaml
shape_meta:
  action:                              # state 同构
  - {key: left_arm,      lerobot_key: action, start_index: 0,  raw_shape: 6, shape: 6}
  - {key: left_gripper,  lerobot_key: action, start_index: 6,  raw_shape: 1, shape: 1}
  - {key: right_arm,     lerobot_key: action, start_index: 7,  raw_shape: 6, shape: 6}
  - {key: right_gripper, lerobot_key: action, start_index: 13, raw_shape: 1, shape: 1}
```

`GroupedPaddingMerger` 自动完成双向对齐，**不需要我们写任何映射代码**：

| 方向 | 行为 |
|------|------|
| 前向（喂 state） | `left_arm(6)` 右侧补 3 个 0 → 占满 `left_control(9)` 槽；`lower_body` 整组不存在 → 造 7 维虚拟零，并在 `proprio_dim_is_pad` 标记为 padding |
| 反向（取 action） | 按槽位切开 27 维 → `_restore_dict` 按 shape_meta 裁回原维度：`left_control[..., :6]` → `left_arm`，`lower_body` 因不在 shape_meta 里被直接丢弃 |

于是 **27 → 14 的实际对应关系**是：

```
模型输出 27 维                          比赛 (1,14) 绝对关节位置
left_control[0:6]   ───────────────►   [0:6]   左臂 6 关节
left_control[6:9]   ── 丢弃（padding）
left_gripper[0:1]   ───────────────►   [6]     左夹爪
right_control[0:6]  ───────────────►   [7:13]  右臂 6 关节
right_control[6:9]  ── 丢弃（padding）
right_gripper[0:1]  ───────────────►   [13]    右夹爪
lower_body[0:7]     ── 丢弃（比赛无下肢/底盘）
```

`g05_model.py` 里只做最后一步"部件 dict → 14 维打平"，用的就是 shape_meta 的 `start_index`：

```python
chunk[:, start : start + dim] = arr        # left_arm→0, left_gripper→6, right_arm→7, right_gripper→13
```

### 相机与状态也是天然对齐的

| 项 | 比赛 | G0.5 robotwin embodiment | 结论 |
|----|------|--------------------------|------|
| 相机 key | `cam_high` / `cam_left_wrist` / `cam_right_wrist` | 同名 | 直接透传 |
| 图像 | `(1,H,W,4)` uint8 → 取 `[0,...,:3]` | `raw_shape [3,480,640]` → resize `[3,256,256]` | 一致（480×640） |
| 状态 | `obs["robot"]["qpos"]` `(1,14)` | 14 维打平 | 逐位对应 |
| 动作 | `(1,14)` float32 绝对关节位置 | 同 | 逐位对应 |

---

## 4. 权重

**已就位**（gated 授权已开通，由独立进程下载）：

```
"$MODELS_ROOT"/g05/
├── action_tokenizer.pt              484M   RVQ 动作 tokenizer（共享）
├── g05-base/
│   ├── .hydra/config.yaml                  模型结构 + tokenizer 配置（部署时从这里读）
│   ├── checkpoints/model_state_dict.pt     主权重
│   └── dataset_stats.json                  归一化统计量
└── (还需 qwen3_5_2b_base_processor/)
```

GalaxeaVLA 的配置里这些路径都是**相对项目根目录**的 `checkpoints/...`，所以要软链过去：

```bash
cd policy/g05/GalaxeaVLA
mkdir -p checkpoints
ln -sfn "$MODELS_ROOT"/g05/action_tokenizer.pt checkpoints/action_tokenizer.pt
ln -sfn "$MODELS_ROOT"/g05/g05-base            checkpoints/g05-base
ln -sfn "$MODELS_ROOT"/g05/qwen3_5_2b_base_processor checkpoints/qwen3_5_2b_base_processor
```

完整权重集约 **55 GB**（含 RoboTwin checkpoint）；单个 G0.5 checkpoint 约 **11 GB**，
`action_tokenizer.pt` 约 484 MB，`qwen3_5_2b_base_processor/` 约 22 MB。

重新下载（gated，需先在 HF 网页同意协议并配 token）：

```bash
huggingface-cli download OpenGalaxea/G05 --repo-type model \
    --local-dir "$MODELS_ROOT"/g05 \
    --include "g05-base/*" "qwen3_5_2b_base_processor/*" "action_tokenizer.pt"
```

> gated 仓库**不能走 hf-mirror**，必须官方源 + token。

---

## 5. 环境

```bash
bash policy/g05/setup_env.sh          # 建 policy/g05/GalaxeaVLA/.venv
```

用官方方式 `uv sync --frozen --index-strategy unsafe-best-match`，python 3.10.16 / torch 2.7.1+cu128。

三个实测要点写死在脚本里：

1. **必须 unset 代理**。`download.pytorch.org` 直连 **65 MB/s**，走 `127.0.0.1:7897` 代理只有 **2.3 MB/s**，torch 单个 wheel 就 1.1 GB。
2. **用 `--frozen`**，绝不重写官方 `uv.lock`。官方 lock 的默认 registry 本来就是 `mirrors.aliyun.com/pypi`，已经是国内源，不需要再换 tuna。
3. **失败不自动回退**。曾经写过"`--frozen` 失败就退回普通 `uv sync`"，实际踩到坑：
   进程被 kill 也算失败，于是自动跑了会**改写官方 `uv.lock`** 的命令。现在改成报错停下。

安装完先跑冒烟测试（不需要权重）：

```bash
policy/g05/GalaxeaVLA/.venv/bin/python policy/g05/smoke_test.py
```

> 同一时刻只能有一个 `uv sync`，它们抢 `~/.cache/uv/.lock`。
> 装到一半没动静时先 `pgrep -af "uv sync"` 看是不是起了两个。

---

## 6. 训练

### 6.1 数据格式：不需要搬运数据本体

G0.5 读数据是**声明式切片**——数据保持标准 LeRobot 布局，由 `configs/data/<name>.yaml` 的
`shape_meta` 声明"每个部件从打平向量第几位开始、取几维"。

实测比赛数据 `RoboSynChallenge/lerobot_dataset/` **已经是 LeRobot v3.0**，
且 key 名和维度与 G0.5 完全一致，**无需任何数据转换**：

| 项 | 比赛数据实测 | G0.5 期望 |
|----|--------------|-----------|
| `codebase_version` | `v3.0` | 支持 `2.1` / `3.0`（`lerobot_ds_version` 开关） |
| 状态列 | `observation.state` `[14]` | 同 |
| 动作列 | `action` `[14]` | 同 |
| 视频列 | `observation.images.cam_{high,left_wrist,right_wrist}` | 同 |
| 分辨率 / fps | 480×640 / 25 | `raw_shape [3,480,640]`，fps 从 `meta/info.json` 读 |

生成配置：

```bash
python policy/g05/convert_lerobot_to_g05.py scan lerobot_dataset --emit-config
```

实测扫描结果：10 个数据集 / 50 episodes / 12415 frames / 全部 v3.0 / 25fps / av1。
产出 `policy/g05/configs/{data,task}/cobotmagic.yaml`。

其他子命令：

```bash
python policy/g05/convert_lerobot_to_g05.py inspect <dataset_dir>   # 单个体检
python policy/g05/convert_lerobot_to_g05.py migrate <v21> <v30>     # 真 v2.1 数据才需要
python policy/g05/convert_lerobot_to_g05.py selftest                # 造假数据自测，已通过
```

`migrate` 做的是：同 chunk 的 per-episode parquet 合并成 `data/chunk-000/file-000.parquet`、
生成 `meta/episodes/chunk-000/file-000.parquet` 索引（`dataset_from_index` / `dataset_to_index`）、
`tasks.jsonl`→`tasks.parquet`、视频按 v3.0 路径模板**软链**（不复制，数据几十 GB）。

### 6.2 启动微调

```bash
bash policy/g05/finetune.sh 8 cobotmagic
bash policy/g05/finetune.sh 8 cobotmagic model.batch_size=8 model.max_epochs=5
```

脚本会把 `policy/g05/configs/{data,task}/cobotmagic.yaml` **软链**进 `GalaxeaVLA/configs/`
（hydra 只认那个目录），并且拒绝覆盖已存在的非软链文件，保证不污染官方源码。

另外脚本替你处理了两个官方脚本的坑：

- **`G05_OUTPUT_DIR` 必须导出**。`configs/train.yaml` 的
  `hydra.run.dir: ${oc.env:G05_OUTPUT_DIR}/${task}/${exp_name}` **没有默认值**，
  不设会在 hydra 解析阶段直接 `KeyError`。脚本默认设成 `policy/g05/outputs`。
- **`logger.mode` 默认改成 offline**。官方默认 `wandb` + `online`，本机没登录会卡在联网重试。
  显式传 `logger.mode=online` 可覆盖。

### 6.4 微调产物布局（正好对接部署）

`datastatics_path: null` 且 `use_pretrained_norm_stats: false` → `finetune.py` 走
"从训练数据现算 norm stats"分支，并把结果写进产物目录。产物目录是自包含的：

```
$G05_OUTPUT_DIR/cobotmagic/<exp_name>/
├── .hydra/config.yaml        # 模型结构 + tokenizer 配置（部署时读这个）
├── dataset_stats.json        # 现算的归一化统计量
├── action_tokenizer.pt       # 从共享位置复制进来，配置里的路径被改写成这份
├── hf_processor/             # 同上
└── checkpoints/step_<N>.pt   # 权重
```

> **注意 checkpoint 文件名是 `step_<N>.pt`，不是 `model_state_dict.pt`。**
> 部署时 `ckpt_path` 指到 `.../checkpoints/step_<N>.pt` 即可，
> `g05_model.py` 用 `parent.parent` 回推 run_dir，三个文件都能找到。

### 6.3 显存要求

官方 `GalaxeaVLA/README.md` 的原话：

| 模式 | 显存需求 | 示例 GPU |
|------|----------|----------|
| Inference | > 8 GB | RTX 3090 / **RTX 4090（官方推荐）** |
| Fine-Tuning (Full) | > 70 GB | A100 (80GB) / H20 (96GB) |

也就是说：

- **推理/评估在本机 4090 上完全够用**（官方点名推荐 4090），
- **全量微调 4090 跑不动**，必须换 A100 80G / H20 96G。`finetune.sh` 启动时会打印这条提示。

OOM 时官方建议调小对应 task config 里的 `model.batch_size`。
多卡是 DDP（每卡一份完整模型副本），**不能**靠加卡降低单卡显存。

---

## 7. 评估

```bash
bash policy/g05/eval.sh <task_name> <setting> [ckpt_path] [gpu_id] [extra_opts...]
bash policy/g05/eval.sh click_bell random \
     policy/g05/GalaxeaVLA/checkpoints/g05-ft-cobotmagic/checkpoints/model_state_dict.pt 0
```

checkpoint 目录必须是官方约定布局，三个文件缺一不可：

```
<run_dir>/checkpoints/model_state_dict.pt
<run_dir>/.hydra/config.yaml          # 模型结构 + action tokenizer 配置
<run_dir>/dataset_stats.json          # 归一化统计量（也可放上层目录）
```

微调完之后把 `deploy_policy.yml` 里这三项一起改成新 embodiment：

```yaml
sim_task: cobotmagic
embodiment: cobotmagic
ckpt_path: <微调产物>/checkpoints/model_state_dict.pt
```

---

## 8. 风险与遗留问题

### 8.1 `g05-base` 零样本可以跑（此前判断有误，已用真实权重纠正）

拿到真实权重前我根据配置推断"base 不能零样本部署"，**两条依据里有一条是错的**，
实跑之后订正如下。

**错的那条：`num_output_cameras: 18` 不是"18 台相机"。**
它是**图像槽位总数** = `num_obs_steps × 相机数`。g05-base 实际是
`num_output_cameras: 18` / `num_obs_steps: 6` / `num_input_cameras: 3`，
即 6 个观测步 × 3 路相机 —— 相机数和比赛完全一致。
（`_preflight()` 里的校验已按 `num_obs_steps × 相机数` 改正。）

**对的那条：`dataset_stats.json` 里确实没有 `robotwin` / `cobotmagic` 的统计量。**
但它覆盖的 22 个预训练 embodiment 里，有 **8 个动作布局与比赛逐位一致**
（`left_arm 6 / left_gripper 1 / right_arm 6 / right_gripper 1`）：

```
bimanual_yam          galaxea_r1lite        robocoin_r1lite_1     robocoin_r1lite_2
robocoin_split_aloha  robomindv2_Agilex     robomindv2_Ark        robomindv2_UR5
```

所以**借用其中一个 embodiment 的归一化统计量就能零样本跑**。默认选
`robomindv2_Agilex` —— Agilex 正是比赛机器人 CobotMagic 的厂商，先验最接近。

`g05_model.py` 在 stats 对不上时会自动算出"哪些 embodiment 与当前动作布局兼容"
并列在报错里，不用手工翻 JSON。

**实跑结果**（`g05-base` 零样本，1 集）：

```
EXIT=0 | Task timeout!
Episode 1 policy eval timing: avg=2.584222s over 38 inference calls, env_steps=600/1000
[  1/1] FAIL  (success rate: 0/1 = 0.0%)
```

**管线通了，但任务没成功** —— 这符合预期：base 是多机型预训练权重，没见过这个任务、
这套相机外参和这个夹爪量纲，零样本做不出 `click_bell`。要拿成绩仍然必须微调
（见第 6 节），或拿到官方 `g05-robotwin20` 变体。零样本这条路的价值是**把整条链路验通**。

### 8.2 已验证 / 尚未验证

**已实跑验证**（`smoke_test.py` 6/6 通过）：

| 项 | 结论 |
|----|------|
| 环境 | venv 11 GB / 594 包；python 3.10.16、torch 2.7.1+cu128、CUDA 可用 |
| `deploy_policy` / `g05_model` import | 通过，四个接口函数齐全 |
| hydra 组装 `sim_robotwin`+`task=robotwin` | 通过，切片确为 14 维 3 相机，`action_size=32` |
| `encode_obs` | 比赛格式 → 3×(480,640,3) uint8 + (14,) |
| 27→14 落位 | `left_arm→0:6 / left_gripper→6 / right_arm→7:13 / right_gripper→13` 逐位正确，缺 key 会报错 |
| av1 视频解码 | torchcodec 解出 (3,480,640) uint8 |
| 数据体检 + 配置生成 | 10 个数据集全部 v3.0/25fps；生成的 yaml 可被 OmegaConf 加载，10 个目录全部存在 |
| `convert_lerobot_to_g05.py selftest` | 全绿（含 v2.1→v3.0 迁移） |
| 所有 py / sh | `py_compile` 与 `bash -n` 通过 |

**端到端也已实跑通过**（`g05-base` 零样本，`click_bell` / `clear` / 1 集，RTX 4090）：

| 项 | 结论 |
|----|------|
| 权重加载 | `instantiate(model_arch)` + `load_state_dict_safely` 通过 |
| `PolicyInferencer.infer()` | 输出 key 与 shape_meta 一致，27 维分组正确还原成 14 维 |
| 环境步进 | 600 步跑完，38 次推理，平均 **2.58 s/次**（`replan_steps=16`，600/16≈38 ✓） |
| 退出码 | `EXIT=0`，录像落盘 `eval_result/.../episode_000_seed_209652396_fail.mp4` |
| 任务结果 | `FAIL`（timeout）—— 预期内，零样本 base 权重做不出该任务 |

**仍未验证**：微调本身（需 80G 卡，见 6.3），以及微调产物的部署路径。

### 8.3 夹爪量纲

比赛夹爪范围是 `0~0.05`，而 `g05-robotwin20` 是在 RoboTwin 的 aloha 上训的，夹爪量纲不一定相同。
走微调路线时归一化统计量从比赛数据现算，这个问题自动消失；
零样本用 robotwin 权重时要留意，可能需要标定。
`deploy_policy.py` 已经按 `env.single_action_space` 的上下界做截断，防止越界值打飞关节。

### 8.4 控制频率

`control_frequency` 默认 **25.0**（比赛数据是 25fps）。这个值会喂给 action tokenizer。
若直接用官方 `g05-robotwin20` 权重零样本评估，官方 adapter 硬编码的是 **15.0**，
届时应改回 15.0 更贴近其训练设置。

### 8.5 torchcodec 加载失败 → 训练读不了视频（已实测，有修复）

`uv sync` 装完后跑冒烟测试，torchcodec **加载不了原生库**：

```
RuntimeError: Could not load libtorchcodec.
  FFmpeg version 7: libnppicc.so.12: cannot open shared object file
```

`libnppicc.so.12` 属于 `nvidia-npp-cu12`，它**不在官方 `uv.lock` 的依赖里**，
但 torchcodec 的 FFmpeg 后端需要它。

影响范围：

- **评估/部署不受影响** —— 观测直接来自仿真器，整条链路不解码任何视频。
- **训练受影响** —— `BaseLerobotDataset` 走 `get_safe_default_codec()`，
  它只用 `importlib.util.find_spec("torchcodec")` 判断，模块存在就返回 `"torchcodec"`，
  然后在 `from torchcodec.decoders import VideoDecoder` 处炸。
  而 `MultiLeRobotDataset` 构造时没有暴露 `video_backend` 参数，
  **没有配置项能切回 pyav**，所以只能把 torchcodec 修好。

**已修复并验证通过**，`setup_env.sh` 现在会自动处理，分两步：

1. `uv pip install --python .venv/bin/python nvidia-npp-cu12`
   （索引显式指到 aliyun —— `uv pip install` **不读 `uv.lock`**，默认走 pypi.org，国内直连极慢）
2. 装完还不够：它落在 `site-packages/nvidia/npp/lib`，**不在动态库搜索路径上**，
   必须把这个目录加进 `LD_LIBRARY_PATH`。`eval.sh` / `finetune.sh` 都已经加了。

`smoke_test.py` 用 `ctypes.CDLL(..., RTLD_GLOBAL)` 预加载这些 `.so`，
所以它不依赖外部环境变量也能独立跑通。

复验结果：

```
ok    5. 比赛 av1 视频解码  (torchcodec 解 av1 成功 (3, 480, 640) <- cobotmagic_Sim_click_the_bell_000)
```

av1 编码本身不是问题：机器上 ffmpeg 带 `libdav1d`，视频流是 `codec=av1 640x480 25fps`。

### 8.6 目录名 `policy/g05/` 与 GalaxeaVLA 顶层包 `g05` 撞名（已修复）

`scripts/eval_policy.py` 会把 `RoboSynChallenge/policy` 加进 `sys.path`，
于是裸写 `import g05` 会命中**我们这个适配器目录**而不是 `GalaxeaVLA/src/g05`，触发循环导入：

```
policy/g05/__init__.py -> deploy_policy -> g05_model
  -> `from g05.models...` 命中 policy/g05/__init__.py（尚未初始化完）
  -> ImportError: cannot import name 'CAMERA_KEYS' from partially initialized module 'g05_model'
```

只调 `sys.path` 顺序不保险（谁先进 path 取决于调用方）。`g05_model.py` 里的
`_bind_galaxea_g05_package()` 直接按绝对路径把真正的 GalaxeaVLA 包注册进 `sys.modules["g05"]`，
让后续所有 `from g05.xxx import` 确定落到 GalaxeaVLA 上。

> 这个坑只在**评估脚本的真实加载路径**上暴露，冒烟测试前 6 项用的是另一种导入方式，测不出来。
> 已补 `smoke_test.py` 第 7 项专门守这个回归。

### 8.7 仿真栈与策略栈同进程共存

`eval_policy.py` 是单进程的：同一个解释器里既要有 G0.5 策略栈，也要有 EmbodiChain 仿真栈。
`setup_env.sh` 已自动处理，几个要点：

- **一律 `uv pip install`（不进 `uv.lock`）+ `--no-deps`**，保住锁定的 torch 2.7.1+cu128。
  实测装完 torch / torchvision / numpy / transformers 版本均未被改动。
- **顺序**：仿真栈必须装在 `uv sync` 之后。`uv sync --frozen` 会删掉所有不在 lock 里的包，
  装完再跑 `uv sync` 会把仿真栈整套清掉。
- **`dexsim_engine==0.4.3`** 走私有源 `http://pyp.open3dv.site:2345/simple/`。
  注意 `UV_DEFAULT_INDEX` 环境变量会**盖掉** `--index-url`，
  踩过一次 `dexsim-engine was not found in the package registry`，
  必须把默认索引整个换成私有源。
  它会顺带替换 `coacd/mujoco/open3d/trimesh` —— 那几个是 GalaxeaVLA 自带 GalaxeaManipSim 用的，
  我们走 EmbodiChain，不受影响。
- **`warp-lang` 必须钉 1.14.0**：dexsim 0.4.3 只兼容这个版本（dexsim 会带进来 1.13.0，需覆盖）。
- **`polars` 必须钉 1.31.0**：新版 1.43.2 装出来会报 `Polars binary is missing!`。
- **`tensordict==0.13.0`**（配 torch 2.7.1，`robosynchallenge.managers.actions` 要），
  它还需要 `pyvers` / `orjson`。
- **`pytorch_kinematics` 有版本陷阱**：GalaxeaVLA 钉的 `pytorch_kinematics_ms==0.7.3`
  提供的模块名同样叫 `pytorch_kinematics`，但**缺 `forward_kinematics_tensor`**，
  EmbodiChain 运行时会炸；PyPI 上的 0.10.0 也缺。必须用仓库根 `.venv` 里的**补丁版**
  （`setup_env.sh` 会自动检测并复制）。GalaxeaVLA 自己的 `src/` 从不 import 它，替换是安全的。

### 8.8 拿到真实权重后修的三个坑

前面三项都是"没有权重时写的代码，在真实 `.hydra/config.yaml` 上才暴露"。

**① `processor.tokenizer_params.pretrained_model_name_or_path` 是 `null`**

仓库的 `configs/model/g05.yaml` 里这个字段本来是插值
`${model.model_arch.hf_processor_path}`，训练落盘时被拍平成了 `null`。
原来的代码直接 `str(None)` 去拼路径，报 `GalaxeaVLA/None 的 tokenizer.json 不存在`。
现在按原插值回填成 `model_arch.hf_processor_path`。
（官方 `experiments/robotwin/galaxeafm_policy/deploy_policy.py` 有同样的写法，
拿 `g05-base` 跑也会踩同一个坑。）

**② `# @package _global_` 必须是 yaml 第一行**

生成 task 配置时我把一行注释放到了它前面，hydra 就不再把它当全局包，
`override /model: g05` 被解释成 `model@task.model`，报
`Could not override 'model@task.model'`。前面连注释都不能有。
`convert_lerobot_to_g05.py` 的 selftest 已加断言守这条。

**③ 汇总输出被 stdout 缓冲吞掉**

仿真器在进程退出阶段的清理会把缓冲区里的
`Episode ... SUCCESS/FAIL` / `success rate` 整段吞掉，日志停在 tqdm 进度条上，
看起来像"跑了一半就没了"，但退出码是 0、录像也已落盘。
`eval.sh` 现在 `export PYTHONUNBUFFERED=1`。

### 8.9 data yaml 里的 `action_state_merger` 是"死配置"

生成的 `configs/data/cobotmagic.yaml` 里写了 `PaddingActionMerger(14)`，
但 `build_processors()` 有一条显式规则：只要 `cfg.model.processor.action_state_merger` 存在，
就**整体替换**掉 data 层的同名配置。所以真正生效的永远是模型层的 `GroupedPaddingMerger`。
这里保留该字段是为了与官方 `configs/data/robotwin.yaml` 保持结构一致（官方也是这么写的），
改它不会有任何效果——要改分组布局得改 `configs/model/g05.yaml` 或 checkpoint 的 hydra 配置。

### 8.10 许可

G0.5 是 **非商用**许可（`LICENSE-G0.5`，LicenseRef-G0.5-Community-1.0）。比赛用途需自行确认合规。
