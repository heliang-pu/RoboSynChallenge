# Motus × RoboSynChallenge 接入说明

把 [thu-ml/Motus](https://github.com/thu-ml/Motus)（统一隐动作世界模型，MoT 三专家 ~8B）接到
RoboSynChallenge 评测框架。本目录**只新增文件**，没有修改比赛官方文件，也没有修改 `Motus/`
克隆体本身——上游 `Motus/inference/robotwin/Motus/` 是自包含的推理栈，我们直接 import 复用。

```
policy/motus/
├── __init__.py                     # from .deploy_policy import *（评测器 import 的是包）
├── deploy_policy.py                # get_model / eval / reset_model
├── motus_model.py                  # MotusPolicy 封装类（进程内推理）
├── deploy_policy.yml               # 评测配置
├── eval.sh                         # 评测入口
├── setup_env.sh                    # 建 .venv（py3.10 + torch2.7.1cu128 + EmbodiChain 仿真栈）
├── prepare_data.py                 # LeRobot v3.0 → Motus 原生训练格式
├── finetune.sh                     # Stage-3 SFT 启动脚本（需 >80G 显存机器）
├── configs/
│   ├── robosyn_infer.yml           # 我们微调 checkpoint 的模型配置（chunk 48）
│   ├── robotwin2_infer.yml         # 官方 Motus_robotwin2 的模型配置（chunk 16）
│   ├── robosyn_finetune.yaml       # 训练配置
│   └── stat.json                   # prepare_data.py --emit-stats 生成
├── tests/test_offline.py           # 离线自检（无权重/无 GPU/无仿真器）
└── Motus/                          # 上游克隆（只读）
```

---

## 0. 怎么用（每条都可直接复制执行）

下面所有命令都在 **4090 (fmc3-0-outer)** 上、以 `"$REPO_ROOT"` 为
工作目录执行。

### 0.1 环境安装

一条命令建好 venv（Python 3.10 + torch 2.7.1cu128 + Motus 依赖 + EmbodiChain 仿真栈）：

```bash
cd "$REPO_ROOT"
bash policy/motus/setup_env.sh
```

变体：

```bash
# 训练机上再补 tensorboard / wandb
bash policy/motus/setup_env.sh --with-train

# 只做数据转换、不需要仿真器时跳过 EmbodiChain（快很多）
bash policy/motus/setup_env.sh --no-sim
```

装完自检（脚本末尾会自动跑一遍，也可以单独执行）：

```bash
cd "$REPO_ROOT"
PYTHONPATH="$EMBODICHAIN_ROOT" \
  policy/motus/.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import eval_policy; print('eval_policy OK')"

policy/motus/.venv/bin/python policy/motus/tests/test_offline.py
```

> 脚本会在 `.venv/.cuda-shim/bin/` 放一个只报版本号的 `nvcc` 假桩。原因见 §8 风险 3：
> Motus 的 `utils/common.py` 在模块级 `import deepspeed`，而 deepspeed 无论安装还是 import
> 都会去执行 `$CUDA_HOME/bin/nvcc -V` 读版本号，这台机器没有 CUDA toolkit。
> `DS_BUILD_OPS=0` 下不会编译任何 CUDA 算子，所以假桩是够用的。

### 0.2 数据准备

第一步把比赛的 LeRobot v3.0 数据转成 Motus 原生格式，并统计 14 维量程：

```bash
cd "$REPO_ROOT"
policy/motus/.venv/bin/python policy/motus/prepare_data.py \
    --lerobot-root "$REPO_ROOT"/lerobot_dataset \
    --output-root  "$WS_ROOT"/data/motus_robosyn \
    --emit-stats
```

第二步预编码 T5（`RobotWinTaskDataset` 强制要求 `umt5_wan/` 非空；需要能加载 umT5-xxl 的机器）：

```bash
cd "$REPO_ROOT"
policy/motus/.venv/bin/python policy/motus/prepare_data.py --t5-only \
    --output-root "$WS_ROOT"/data/motus_robosyn \
    --wan-path    "$MODELS_ROOT"/motus/Wan2.2-TI2V-5B
```

只转某几个任务、或先小规模试跑：

```bash
policy/motus/.venv/bin/python policy/motus/prepare_data.py \
    --lerobot-root "$REPO_ROOT"/lerobot_dataset \
    --output-root  "$WS_ROOT"/data/motus_robosyn \
    --tasks handle_basket drawer_open_place --limit-episodes 2 --overwrite
```

### 0.3 评估

```bash
cd "$REPO_ROOT"
bash policy/motus/eval.sh <task_name> <setting> <ckpt_dir> <model_name> <gpu_id> [额外参数]
```

用我们微调出来的 checkpoint（默认 `model_config: configs/robosyn_infer.yml`）：

```bash
cd "$REPO_ROOT"
bash policy/motus/eval.sh handle_basket random \
    "$WS_ROOT"/outputs/motus-robosyn/checkpoints/checkpoint_step_20000 \
    motus 0 --max_episodes 20
```

用官方 `Motus_robotwin2` 做零样本 sanity check（**必须**同时换模型配置和 `action_repeat`）：

```bash
cd "$REPO_ROOT"
bash policy/motus/eval.sh handle_basket random \
    "$MODELS_ROOT"/motus/Motus_robotwin2 \
    motus 0 --max_episodes 5 \
    --model_config configs/robotwin2_infer.yml --action_repeat 3
```

常用调参（都可以直接追加在命令末尾）：

```bash
--num_inference_timesteps 6    # 去噪步数调小，换推理速度
--execute_steps 48             # 一次推理执行满整个 chunk，减少重规划次数
--headless true                # 无显示环境
--save_video_debug true        # 落盘模型预测的未来帧，便于诊断
```

### 0.4 训练

**本机 4090 48G 训不了，需要 >80GB/卡的机器。** 在训练机上：

```bash
cd "$REPO_ROOT"
bash policy/motus/setup_env.sh --with-train

# 8 卡，用 configs/robosyn_finetune.yaml，DeepSpeed ZeRO-1
bash policy/motus/finetune.sh 8 policy/motus/configs/robosyn_finetune.yaml zero1
```

改卡数 / 换 ZeRO 策略 / 指定输出目录：

```bash
RUN_NAME=motus-robosyn-z2 OUTPUT_DIR="$WS_ROOT"/outputs/motus-z2 \
  bash policy/motus/finetune.sh 4 policy/motus/configs/robosyn_finetune.yaml zero2
```

开训前请确认 `configs/robosyn_finetune.yaml` 里这几项指向真实路径：
`dataset.dataset_dir`、`model.wan.*`、`model.vlm.checkpoint_path`、`finetune.checkpoint_path`。

---

## 1. 推理调用链

进程内单函数调用，没有 server/client 拆分。

```
deploy_policy.eval(env, model, obs)
  └─ MotusPolicy.update_observation_window(img_arr, state)
  └─ MotusPolicy.get_action()
       └─ Motus.inference_step(first_frame, state, num_inference_steps,
                               language_embeddings, vlm_inputs)
            → (predicted_frames, predicted_actions[1, chunk, 14])
```

### 加载的组件

| 组件 | 来源 | 说明 |
|---|---|---|
| Motus 主干（WAN 视频专家 + 动作专家 + 理解专家，~8B） | `ckpt_path/mp_rank_00_model_states.pt` | `load_pretrained_backbones=False`，**全部权重来自 checkpoint** |
| Wan2.2 VAE | `wan_path/Wan2.2_VAE.pth` | 编解码条件帧/预测帧 |
| umT5-xxl 文本编码器 | `wan_path/models_t5_umt5-xxl-enc-bf16.pth` + `wan_path/google/umt5-xxl` | 只在 `set_language()` 时用一次，用完即释放 |
| Qwen3-VL-2B processor | `vlm_path` | **只用 config/tokenizer/processor**，权重同样来自 checkpoint |

> `wan_path` / `vlm_path` 里的大权重文件其实不需要（除了 VAE 和 T5）——上游 README 也这么写。

### 一个重要的代码事实

`models/motus.py` 里有 **三个** `inference_step` 定义（880 / 1006 / 1199 行），但后两个整段被
`'''` 包住是死代码。AST 校验确认 `Motus` 类上只有 **880 行那个**是活的：普通 Euler flow-matching
积分，`timesteps = linspace(1.0, 0.0, n+1)`，默认 `num_inference_timesteps: 10`。
所以 dpm++ / UniPC solver 相关的配置项都是无效的，别去调。

### 输入格式

- `first_frame`：`[1, 3, 384, 320]` float32 ∈ [0,1]，**三视图 T 型拼接后再等比缩放补边**（见 §3）
- `state`：`[1, 14]` 当前绝对关节位置
- `language_embeddings`：`list[Tensor[S, 4096]]`，umT5 编码的 **带场景前缀** 的指令
- `vlm_inputs`：`[{input_ids, attention_mask, pixel_values, image_grid_thw}]`，同一张条件帧 + 同一句指令

场景前缀必须带上，训练时 `robotwin_converter.py` 把它写进了 `metas/*.txt`：

```
The whole scene is in a realistic, industrial art style with three views: a fixed rear
camera, a movable left arm camera, and a movable right arm camera. The aloha robot is
currently performing the following task: <instruction>
```

---

## 2. 动作语义与 14 维映射

**输出 = 绝对关节位置（absolute qpos），不是增量，不做反归一化。**

`action_chunk_size = num_video_frames × video_action_freq_ratio`

| 配置 | chunk | downsample | 用途 |
|---|---|---|---|
| `configs/robotwin2_infer.yml` | 8×2 = **16** | 3 | 官方 Motus_robotwin2 |
| `configs/robosyn_infer.yml` | 8×6 = **48** | 1 | 我们的微调（25fps 原速，覆盖 1.92 s） |

### 14 维就是恒等映射

比赛 `env.step` 要 `(1,14)` float32 绝对 qpos = `[左臂6, 左夹爪1, 右臂6, 右夹爪1]`，
下标 6/13 是夹爪。本地 LeRobot 数据集 `meta/info.json` 的维度命名是
`left_joint1..left_joint7, right_joint1..right_joint7`，`joint7` 即夹爪——**顺序完全一致，
不需要任何置换**。只要用我们自己的数据微调，模型输出直接喂 `env.step` 即可。

### 夹爪量程实测：0~1，不是 0~0.05

任务书里写的"左夹爪 0~0.05"与实测不符。用 `prepare_data.py --emit-stats` 扫完全部 10 个任务
共 **12415 帧**，第 6、13 维的 `action` 量程恰好是 **[0.0000, 1.0000]**，且 `q01=0.0 / q99=1.0`
（双峰二值分布）。旁证：`configs/<task>/action_config.json` 里 `left_eef` / `right_eef` 是
`execute_open` / `execute_close` 两个离散节点，本来就是归一化开合量，不是米制开口宽度。

这和 RoboTwin2 `stat.json` 的夹爪约定（min 0.0 / max 1.0）**完全一致**，所以：

- `gripper_scale` 保持默认 **1.0**，不要设 0.05；
- 零样本迁移时夹爪通道天然对齐，剩下的差异只在手臂关节零位与运动学。

全部 14 维实测量程见 `configs/stat.json`（`--emit-stats` 生成）。

`deploy_policy.py` 里保留了三个可选修正（默认全关，实测下来都不需要）：

- `gripper_scale`：夹爪整体缩放
- `gripper_limits`：夹爪截断，如 `[0.0, 1.0]`
- `action_clip`：逐关节上下限截断（可直接抄 `stat.json` 的 q01/q99）

### 执行节奏

`execute_steps`（默认 24/48）控制一次推理执行多少步后重新推理；`action_repeat` 控制每个动作
保持几个 env step。用 `robotwin2_infer.yml`（downsample=3）时 **必须设 `action_repeat: 3`**，
否则动作会以 3 倍速播放。用 `robosyn_infer.yml`（downsample=1）时保持 `action_repeat: 1`。

---

## 3. 三视图拼接（最容易踩错的地方）

Motus 吃的是**单张拼接图**，不是三路独立输入。布局（`data/utils/multi_camera_concat.py`
与 `add_cam_concatenated_to_lerobot_dataset.py::_stitch_frames`）：

```
+-----------------------+   上： cam_high 原尺寸           (H × W)
|       cam_high        |   下： 双腕各缩到 H/2 × W/2
+-----------+-----------+   合计：1.5H × W
| cam_left  | cam_right |
+-----------+-----------+
```

再用 `resize_with_padding` 等比缩放补边到 `384 × 320`。

关键性质：**比例拼接让 480×640（比赛）和 240×320（RoboTwin）落到完全相同的画布**——
720×640 和 360×320 都等比缩到 360×320，上下各补 12 行黑边。也就是说几何上和训练分布严格对齐。
`tests/test_offline.py` 的第 1 组用例就在断言这件事。

两个上游坑：

1. **上游 `inference/robotwin/Motus/deploy_policy.py` 不能直接用**：它写死
   `cv2.resize(left_img, (160, 120))`，那是给 320×240 的 RoboTwin 相机准备的。对 640×480
   输入，下排宽 320 和上排宽 640 拼不起来，`np.concatenate(axis=0)` 直接报错。
   我们的 `build_three_view()` 用的是比例版（`H//2, W//2`）。
2. **Motus 的 LeRobot loader 的三相机拼接对我们的数据是错的**：
   `lerobot_dataset.py::load_concatenated_view` 取 `bottom_h = 腕部相机原始高度`——它假设腕部
   视频本来就是从拼接图切出来的半尺寸图。我们三路都是 480×640，走那条路会拼成 960×640，
   腕部画面比训练时大一倍。这也是我们不走 LeRobot 直读、改用转换脚本的原因之一。

---

## 4. 归一化：两条训练路径不兼容（最关键的结论）

| 训练数据路径 | 是否归一化 | 对应部署设置 |
|---|---|---|
| `dataset.type: robotwin`（原生格式） | **否** | `action_normalization: none` |
| `dataset.type: lerobot`（LeRobot 直读） | **是**，min-max 到 [0,1] | `action_normalization: minmax` + `stat_path` |

证据：

- `data/robotwin2/robotwin_agilex_dataset.py:333-334` 两行归一化调用被注释掉了，state 和 action
  都是原始 qpos。
- `data/lerobot/lerobot_dataset.py` 结尾则实打实调了 `normalize_actions(...)`，用
  `data/utils/stat.json[embodiment_type]` 的 min/max。

推论：官方 `Motus_robotwin2` 是走 robotwin 路径训的，所以工作在**原始 qpos 空间**。
上游 `deploy_policy.py` 里那个 `utils/stat.json` + `_normalize_actions/_denormalize_actions`
是**死代码**（`current_state_norm` 算了但从没用过，输出也没反归一化）——不是 bug，只是残留。

我们选 robotwin 原生格式训练，因此 `action_normalization: none`。
**如果哪天改用 LeRobot 直读微调，必须同步把这个开关切到 `minmax`，否则动作会差一个仿射变换。**

---

## 5. 显存配置（48G 4090 可跑推理）

**以下是 2026-08-06 冒烟实测值**（4090 48G，另有常驻任务占 11.4G）：

| 阶段 | device 总占用 | 其中 torch | 说明 |
|---|---|---|---|
| 起点（仅常驻任务） | 11.5 G | 0 | 基线 |
| 建完 env（仿真器 + 渲染器） | 14.0 G | 0 | +2.5 G |
| 加载完 Motus（113s） | 31.6 G | 17.6 G | 权重 ~15G ckpt → 17.6G bf16 常驻 |
| **推理稳态** | **34.9 G** | 17.6 G | 13 次推理全程零增长，无泄漏 |

扣掉常驻任务，**整个评测进程约 23.5 G**，与上游"预编码 T5 ~24GB"一致，48G 卡上余量充足。
上游"不预编码 ~41GB"的模式在本机会 OOM（第一次冒烟就是栽在加载 T5 时 CUDA OOM）。

`motus_model.py` 做了自动化：`t5_mode: auto`（默认）在 `set_language()` 时按需加载 T5 →
编码 → **落盘缓存到 `cache/t5/<sha1>.pt`** → 立刻 `del` + `empty_cache()`。
实测缓存文件 4.2 MB；**第二次起直接命中缓存，9.4G 的 umT5 编码器根本不加载**，
上表的稳态值就是命中缓存后的结果。首次跑新指令若显存紧张，用 `--t5_mode cpu` 在 CPU 上
编码一次把缓存刷出来，之后即可回到 `auto`。

其他取值：`cpu`（在 CPU 上编码，最省显存）、`cache_only`（没缓存就报错，适合严格控显存）、
`keep`（保留 T5，~41 GB）。

`eval.sh` 里设了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，避免"加载→释放 T5"
这两个阶段之间的显存碎片。

---

## 6. 数据：LeRobot 字段映射与转换

### 本机数据的实际情况（和最初假设不同）

`RoboSynChallenge/lerobot_dataset/` 下**已经有 10 个任务的样例数据**（每个 5 episodes）：

```
observation.state              float32[14]   测量关节位置
observation.qvel / qf          float32[14]
action                         float32[14]   指令关节位置
observation.images.cam_high        video 480×640×3, AV1, 25fps
observation.images.cam_left_wrist  video 480×640×3, AV1, 25fps
observation.images.cam_right_wrist video 480×640×3, AV1, 25fps
```

**但 `codebase_version` 是 `v3.0`，不是 v2.1。** v3.0 把多个 episode 打包进
`data/chunk-000/file-000.parquet` 和每相机一个 `videos/<key>/chunk-000/file-000.mp4`，
元数据在 `meta/episodes/**.parquet`。Motus 依赖 `lerobot==0.3.2`，只认 v2.1
（`episode_000000.parquet` + `meta/episodes.jsonl`）。**直读不可行。**

### 相机 key 映射

好消息是名字本身完全对得上，不需要改名：

| 我们的 key | Motus 期望 | 结论 |
|---|---|---|
| `observation.images.cam_high` | 同名 | ✓ |
| `observation.images.cam_left_wrist` | 同名 | ✓ |
| `observation.images.cam_right_wrist` | 同名 | ✓ |
| `observation.state` | `observation.state` / `actions` / `action` | ✓ |
| `action` | `action` / `actions` | ✓ |

真正的障碍是 **v3.0 目录布局** 和 **三相机拼接几何**，不是字段名。

### 转换脚本

`prepare_data.py` 不依赖 lerobot（直接读 parquet + PyAV 解码），把 v3.0/v2.1 转成 Motus
原生格式：

```
<output_root>/clean/<task>/
    videos/{i}.mp4      # T 型三视图，360×320，fps 保留
    qpos/{i}.pt         # torch float32 [T, 14]
    metas/{i}.txt       # 带场景前缀的指令，一行一条
    umt5_wan/{i}.pt     # list[Tensor[S,4096]]，与 metas 行号对齐
```

```bash
# 1) 转视频 + qpos + metas，并统计 min/max 写进 configs/stat.json
python policy/motus/prepare_data.py \
    --lerobot-root "$REPO_ROOT"/lerobot_dataset \
    --output-root  "$WS_ROOT"/data/motus_robosyn \
    --emit-stats

# 2) 预编码 T5（需要能加载 umT5-xxl 的机器；RobotWinTaskDataset 强制要求 umt5_wan/ 存在）
python policy/motus/prepare_data.py --t5-only \
    --output-root "$WS_ROOT"/data/motus_robosyn \
    --wan-path    "$MODELS_ROOT"/motus/Wan2.2-TI2V-5B
```

`--qpos-source` 默认 `action`（记录的指令值），可选 `observation.state`（测量值）。
选 `action` 的理由：Motus 原生格式用**同一个数组**同时当 state 输入和 action 目标，而
比赛 `env.step` 消费的正是指令值；更重要的是夹爪通道——在 click_bell 样例里
`action[13]` 干净地恒为 0，而 `state[13]` 在 -0.064~0.126 之间抖，拿测量值当目标会逼模型
去拟合噪声。代价是推理时喂进去的 state 是测量值、训练时是指令值，两者在仿真里除夹爪外
几乎一致（上游 RoboTwin 因为 qpos 就等于 action，不存在这个区分）。

---

## 7. 训练

**本机（RTX 4090 48G）训不了。** 上游明确要求 >80 GB/卡（A100-80G / H100 / B200）。
`finetune.sh` 是给训练机写的，脚本头部也写了这条警告。

```bash
bash policy/motus/setup_env.sh --with-train        # 补 deepspeed
bash policy/motus/finetune.sh 8 policy/motus/configs/robosyn_finetune.yaml zero1
```

起点用 Stage-2 checkpoint `motus-robotics/Motus`。`train.py` 走的是
`Motus.load_pretrain_weights()`，它期望路径 **`<checkpoint_path>/pytorch_model/mp_rank_00_model_states.pt`**
（比推理的 `load_checkpoint` 多一层 `pytorch_model/`），并且会跳过
`action_expert.input_encoder.*` / `action_expert.decoder.*`。这两个正好是唯一依赖 chunk 大小的
张量，所以**即使从 chunk=16 的 `Motus_robotwin2` 起步也能加载**。

> `configs/robosyn_finetune.yaml` 的 `common` 段必须和 `configs/robosyn_infer.yml` 逐字一致。
> `ActionExpert` 的 `pos_embedding` 是按 `chunk_size + 1 + num_registers` 注册的 buffer，
> 对不上时 `load_checkpoint` 会直接报 size mismatch。

---

## 8. 遗留风险

1. ~~未冒烟~~ **已冒烟通过（2026-08-06）**：`click_bell / clear` 零样本 1 集完整跑完——
   13 次推理 / 600 env steps / `Task timeout!` → `[1/1] FAIL`，平均 9.26s 一次推理，
   录像 `episode_000_seed_209652396_fail.mp4` 已落盘。FAIL 是**预期**结果，因为用的是
   RoboTwin2 本体的零样本权重（见风险 2）；能跑满 600 步并正常判定即说明链路是通的。
   **仍未验证的是微调之后的成绩。**
2. **零样本跨本体仍不可靠**（但比预想的好一点）。夹爪约定实测与 RoboTwin2 一致（都是 0~1），
   图像几何也严格对齐，所以剩下的 gap 只在手臂关节零位/运动学和任务分布。可以用
   `model_config: configs/robotwin2_infer.yml` + `action_repeat: 3`（**不要**动 `gripper_scale`）
   跑个 sanity check，但别指望成绩。**真正的路径是微调。**
3. **deepspeed + nvcc 假桩**。Motus 的 `utils/common.py` 在模块级 `import deepspeed.comm.comm`，
   所以 deepspeed 是**推理**依赖而不只是训练依赖。deepspeed 只发 sdist（无 wheel），安装和
   运行时 import 都会执行 `$CUDA_HOME/bin/nvcc -V`，而这台机器没有 CUDA toolkit
   （torch 只带运行时库，`nvidia-cuda-nvcc-cu12` 只有 `ptxas` 没有 `nvcc`）。
   解决办法是 `setup_env.sh` 在 `.venv/.cuda-shim/bin/` 放一个只打印版本号的 `nvcc`，
   `motus_model._ensure_cuda_home()` 在 import Motus 前自动选中它。
   `DS_BUILD_OPS=0` 下不编译任何 CUDA 算子，所以假桩是安全的；但**如果将来真要 JIT 编译
   deepspeed 融合算子（例如上训练机），必须换成真的 CUDA toolkit**。
4. **`pytorch_kinematics` 必须从基准 venv 拷，不能装 PyPI 版**。PyPI 的 0.10.0 缺
   `Chain.forward_kinematics_tensor`，而 EmbodiChain 会调它，装了 PyPI 版会在运行期炸。
   工作区基准 venv `"$REPO_ROOT"/.venv` 里是打过补丁的 0.10.0。
   这个包是纯 Python（无 `.so`），所以从 3.11 的 site-packages 拷到本 venv 的 3.10 是安全的。
   `setup_env.sh` 已自动处理，也可手动执行：

   ```bash
   BASE="$REPO_ROOT"/.venv/lib/python3.11/site-packages
   DEST="$REPO_ROOT"/policy/motus/.venv/lib/python3.10/site-packages
   rm -rf $DEST/pytorch_kinematics $DEST/pytorch_kinematics-*.dist-info
   cp -r $BASE/pytorch_kinematics $BASE/pytorch_kinematics-*.dist-info $DEST/
   # 校验
   $DEST/../../../bin/python -c "from pytorch_kinematics.chain import Chain; print(hasattr(Chain,'forward_kinematics_tensor'))"
   ```

   自检脚本会显式检查这一项，输出 `pytorch_kinematics: WRONG BUILD` 就是装错了。
5. **仿真栈版本对齐是手工的**。`dexsim_engine` 的依赖解析会拉到比 `policy/pi05/.venv`
   更新的 mujoco/newton/warp/trimesh/polars（其中 polars 1.43.2 装出来缺二进制、一 import
   就 warning）。`setup_env.sh` 里把这几个显式 pin 回 pi05 的版本。**唯一没对齐的是
   scikit-learn**：pi05 用 1.9.0，但它要求 Python ≥ 3.11，而 Motus 要 3.10，所以这里停在
   1.7.x。如果 EmbodiChain 有代码依赖 sklearn 1.9 的新 API，会在这里炸。
6. **transformers 版本**。上游 pin `==5.0.0rc0`（预发布），`setup_env.sh` 装不上会回退到
   `>=4.57,<5`（Qwen3-VL 最低要求）。两者的 `AutoProcessor` 行为若有差异，可能影响
   `vlm_inputs` 的 `image_grid_thw`。注意本 venv 的 transformers(5.0.0rc0) 和
   tokenizers(0.23.0rc0) 都比 pi05 的(4.53.2 / 0.21.1)新很多，如果 EmbodiChain 侧也用
   transformers，存在冲突可能。
7. **flash-attn 缺失会直接崩，上游那个"兜底"根本不在调用路径上**（已修，冒烟时踩到的真错误）。
   `bak/wan/modules/attention.py` 里有两个函数：`attention()` 是带 SDPA 兜底的分发器，
   `flash_attention()` 结尾是一句裸的 `assert FLASH_ATTN_2_AVAILABLE`。而 Motus 的
   `model.py`(3 处)、`action_expert.py`、`und_expert.py` **全都直接 import 并调用
   `flash_attention`**，绕开了分发器——所以没装 flash-attn 时第一步去噪就 AssertionError。
   修法：`motus_model._install_sdpa_attention_fallback()` 在 import 完 Motus 之后，把所有
   绑定了该符号的模块（实测 6 个：`wan.modules.attention` / `wan.modules` /
   `wan.modules.model` / `wan.distributed.ulysses` / `models.action_expert` /
   `models.und_expert`）里的 `flash_attention` 换成 `_sdpa_flash_attention`。
   替换实现比上游的兜底更严格：额外正确处理了 `softmax_scale`、`q_scale` 和 `k_lens` 补齐
   掩码，所以数值上对齐 flash-attn 路径而不只是近似。PyTorch 的 SDPA 内部本来就会选融合
   kernel，实测 384×320 视频 latent 上单次推理 ~9.3s，速度可接受。
   **没有修改 `Motus/` 克隆体，是运行期 monkey-patch。** 装上真 flash-attn 后该 patch 自动跳过。
8. **`env.close()` 会把进程带走，吞掉缓冲区里的报错**（EmbodiChain/DexSim 侧问题，非本策略）。
   实测：官方 `eval_policy.py` 的最后一段 `Evaluation Results Summary` **永远打不出来**，
   因为它位于 `finally: env.close()` 之后；`dmesg` 里能看到 `librender-optix.so` 的
   general protection fault。这正是前两次冒烟看起来像"0 步静默退出"的原因——真实的
   AssertionError 当时就在没来得及 flush 的缓冲区里。
   `eval.sh` 现在强制 `PYTHONUNBUFFERED=1`，逐行 flush，报错和每集 SUCCESS/FAIL 都保得住。
   **判定结果请以每集的 `[ 1/1] FAIL/SUCCESS` 行为准，不要等最后的汇总行。**
9. **顶层模块名冲突**。上游用了 `models` / `utils` / `wan` 这种裸名。`_import_motus()` 会检测
   并驱逐已被别的包占用的同名模块，同时打 warning。已实测：在 `import eval_policy`
   （即 embodichain / robosynchallenge 全部加载完）之后再 `_import_motus()` 不冲突。
10. **推理慢，实测 9.26s 一次**（8B × 10 步去噪 + VAE 编解码 + VLM，SDPA 而非 flash-attn）。
    冒烟里 13 次推理覆盖 600 env steps，折合 ~5 env step/s。用 `robotwin2_infer.yml` 时
    chunk 只有 16、`action_repeat=3`，所以一次推理管 48 步；换成 `robosyn_infer.yml`
    （chunk 48、repeat 1）时 `execute_steps` 默认 24，**一次推理只管 24 步，等效帧率会掉一半**，
    建议把 `execute_steps` 调到 48。想更快就调小 `num_inference_timesteps`（10 → 6）。
11. **AV1 解码**。比赛视频是 AV1，`prepare_data.py` 优先用 PyAV（`av` 包，带 libdav1d）。
   若换成 OpenCV 的 `VideoCapture` 可能解不出来。
12. **数据规模**。本地只有每任务 5 条样例（10 任务共 50 episodes / 12415 帧，已全部转换成功）。
   上游在 RoboTwin2 上的 87% 是用每任务 50 clean + 500 randomized、50 个任务合并多任务训练
   得到的，数据量差两个数量级。
13. **动作向量顺序未独立验证**。`configs/<task>/action_config.json` 的 `scope` 字典顺序是
   `right_arm, left_arm, left_eef, right_eef`，与数据集的 `left_joint1..7, right_joint1..7`
   不一致。我按任务书给的契约和 pi05 参考实现（都是直通不做置换）实现，两者一致；但如果
   评测时出现"左右臂互换"的现象，第一个要查的就是这里。

---

## 9. 快速自检

```bash
# 语法
cd "$REPO_ROOT"/policy/motus
python3 -m py_compile __init__.py deploy_policy.py motus_model.py prepare_data.py

# 离线用例（三视图几何 / 动作后处理 / 归一化往返 / v3.0 转换往返）
.venv/bin/python tests/test_offline.py

# 评测（权重就位后）
bash policy/motus/eval.sh click_bell random \
     "$MODELS_ROOT"/motus/Motus_robotwin2 motus 0 --max_episodes 5
```
