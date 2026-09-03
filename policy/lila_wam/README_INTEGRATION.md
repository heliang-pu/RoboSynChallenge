# LiLa-WAM × RoboSynChallenge

把 [LiLa-WAM](https://github.com/teee000/LiLa-WAM)(*Lightweight Latent Reasoning
World-Action Model*,[arXiv:2608.03701](https://arxiv.org/abs/2608.03701))接进
RoboSynChallenge,**直接吃本仓库的 LeRobot v2.1 数据集训练**,并按统一评测接口
(`get_model` / `eval` / `reset_model`)接入 `scripts/eval_policy.py`。

模型本身:冻结的 DINOv3 ViT-L/16 + 0.2B 可训练的 DiT 动作专家,流匹配出 32 步动作块,
外加"未来帧特征预测"辅助监督;任务条件走 VTT(Visual Transition Token,视觉转移向量),
**不吃语言输入**。全模型 0.5B 参数,单张 24GB 卡可训。

---

## 目录

```text
policy/lila_wam/
├── LiLa-WAM/                 # 上游仓库(git submodule,不改动)
├── configs/
│   ├── robosyn_3cam.yaml     # 默认:cam_high + 双腕相机
│   └── robosyn_cam_high.yaml # 单相机,结构上与上游完全一致
├── lerobot_v21.py            # LeRobot v2.1 读取 + JPEG 帧缓存
├── lila_dataset.py           # torch Dataset(替换上游的 RoboTwin HDF5 loader)
├── lila_model.py             # 建模型(多相机 wrapper)
├── lila_infer.py             # 推理引擎(EmbodiChain 观测 → 动作块)
├── build_frame_cache.py      # 帧缓存构建 CLI
├── compute_norm_stats.py     # 归一化 min/max 统计 CLI
├── precompute_task_cond.py   # VTT 任务条件向量 CLI
├── train_lila_wam.py         # 训练入口
├── finetune.sh               # 一条命令跑完整流水线
├── eval.sh / deploy_policy.* # 评测接入
├── setup_env.sh              # 环境搭建 + DINOv3 权重下载
└── tests/test_lila_wam.py    # 数据链路测试(无需 GPU / 权重 / 网络)
```

---

## 1. 环境

**训练**用独立环境:

```bash
bash policy/lila_wam/setup_env.sh --download-encoder
```

**评测**直接用**仓库根 venv**——EmbodiChain / dexsim 本来就装在那儿
(README 的"仿真/采集/评估环境"),只需补两个 LiLa-WAM 要的包:

```bash
uv pip install --python .venv/bin/python "transformers>=4.56,<5" omegaconf
```

实测这一步是纯增量的(新增 transformers / tokenizers / omegaconf /
antlr4-runtime 四个包,零升级零降级),不会动到现有仿真栈。`eval.sh` 默认就用根
venv。只有在根 venv 不可用的机器上,才需要 `setup_env.sh --with-sim` 把整套仿真栈
装进 policy 环境——那条路要私有源权限,而且会多占几个 GB。

上游 README 写的是 conda + Python 3.10,这里改成 uv + 3.11:评测要和 EmbodiChain
跑在同一个解释器里,而仿真栈的 pin 需要 ≥3.11;LiLa-WAM 本身没有 3.10 专属依赖。

**DINOv3 权重是 gated 的**:先去
<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> 同意协议,再
`hf auth login`(或 `export HF_TOKEN=...`),`--download-encoder` 才拉得下来。
权重落在 `policy/lila_wam/dinov3/dinov3-vitl16-pretrain-lvd1689m/`,
路径在 config 的 `model.vision_encoder.checkpoint_path`。

`transformers` 是硬门槛:DINOv3 的建模代码 **4.56 才进 transformers**。
setup_env.sh 先试上游 pin 的 `5.0.0rc0`,拿不到就退到最后一个 4.x。

---

## 2. 数据

要求 **LeRobot v2.1**(`meta/` + `data/*.parquet` + `videos/*.mp4`),也就是
`bash launch/run_task.sh <task> <setting> 2_1 ...` 的直接产物。用到这些特征键:

| 键 | 用途 |
|---|---|
| `observation.state` (14) | 本体感受(proprio)。左右臂各 joint1..7,第 7 维是夹爪 |
| `action` (14) | 动作块的监督目标,同样的排布 |
| `observation.images.cam_high` 等 | 相机;`camera_names` 里的**第一个**是主相机 |

v3.0 数据集会被直接拒掉并提示转换命令:

```bash
python scripts/convert_lerobot3.0_to_2.1.py --input <v3.0 目录> --output <v2.1 目录>
```

### 为什么要建帧缓存

训练是在全体帧上随机采样的,而随机寻址 AV1 视频远远跟不上。
`build_frame_cache.py` 把每集视频**解一次**,按训练分辨率存成 HDF5 里的 JPEG buffer
(`frames/<camera>` 是变长 uint8),训练时只做一次 O(1) 读 + 一次 JPEG 解码——
和上游 RoboTwin HDF5 的访问模式、开销完全一致。

实测(本机,sample_loading,320×240,三相机):**20 集 / 7440 帧 ≈ 14 秒 / 346 MB**,
按这个比例 350 集大约 4 分钟 / 6 GB。缓存默认落在
`policy/lila_wam/cache/<task>/frames_<W>x<H>.h5`,可用 `dataset.cache_dir` 改。

缓存带指纹(数据集、相机集合、分辨率、JPEG 质量),换了任何一项都会拒绝复用,
要重建就加 `--overwrite`。

---

## 3. 训练

```bash
# stage 1:2e-4,上游建议跑 11~12 轮就停
bash policy/lila_wam/finetune.sh \
     lerobot_dataset/click_bell/cobotmagic_Sim_click_bell click_bell 0 --epochs 12

# stage 2:降到 4e-5 再跑 3~4 轮。必须用 --init_from(只load权重、重置优化器和
# lr 调度器),用 --resume 会接着 stage1 的 2e-4 schedule 走
bash policy/lila_wam/finetune.sh \
     lerobot_dataset/click_bell/cobotmagic_Sim_click_bell click_bell 0 \
     --epochs 4 --learning_rate 4e-5 \
     --init_from policy/lila_wam/checkpoints/sft_<时间戳>/checkpoint_epoch_12.pt
```

`finetune.sh` 依次做:生成 config → 建帧缓存 → 统计归一化参数 → 预计算 VTT → 训练。
每步幂等,重跑不会重复干活;只想重训可以 `LILA_SKIP_PREP=1 bash finetune.sh ...`。

**多任务/多数据集共训**:直接写一份 config(`dataset.dataset_dir` 和
`dataset.task_names` 都是列表),然后把 config 当第一个参数传:

```bash
bash policy/lila_wam/finetune.sh policy/lila_wam/configs/my_multitask.yaml multitask 0
```

产物在 `policy/lila_wam/checkpoints/sft_<时间戳>/`,里面除了
`checkpoint_epoch_<N>.pt` 还有 `config.yaml` + `norm_stats.json`,
所以评测只要指到这个目录就够了。

### 单相机 vs 三相机

| | `robosyn_cam_high.yaml` | `robosyn_3cam.yaml`(默认) |
|---|---|---|
| 相机 | 仅 cam_high | cam_high + 左右腕 |
| 与上游结构 | 完全一致(不建相机 embedding) | 多一个可学习的 per-camera embedding |
| 默认 batch | 128 | 48(冻结编码器的前向翻了三倍) |

本仓库其它策略(ACT/DP/pi0.5)评测时都吃三路相机,精细操作任务(移液枪、样品装载)
很依赖腕部视角,所以默认给三相机;想复现论文设置或显存紧张就用单相机那份。

---

## 4. 评测

```bash
bash policy/lila_wam/eval.sh <task_name> <setting> <checkpoint> <model_name> [gpu] [extra...]

bash policy/lila_wam/eval.sh click_bell random \
     policy/lila_wam/checkpoints/sft_2026-08-29_12-00-00 lila_wam 0 --max_episodes 20
```

`checkpoint` 传 run 目录会自动挑 epoch 最大的那个,`config.yaml` / `norm_stats.json`
从同目录读。VTT 向量按 **`task_name`** 查
`dataset.task_cond_dir/<task_name>/task_cond.npy`;如果 checkpoint 训练时数据集
登记的任务名和评测任务名不一样,用 `--task_cond_name` 覆盖。

动作直接用数据集的单位(min-max 归一化 → 反归一化后就是 env 的动作单位),
和 ACT 一样透传给 `env.step`,不做额外的夹爪缩放。

---

## 5. 和上游的差异(以及为什么)

| 处 | 上游 | 这里 |
|---|---|---|
| 数据源 | RoboTwin 2.0 HDF5(`joint_action/vector` + `endpose/*`) | LeRobot v2.1(parquet + mp4),经 JPEG 帧缓存 |
| 状态维度 | 16(双臂 endpose 7+1) | **14**(`observation.state`,关节角),`common.state_dim: 14` |
| 相机 | 单相机 `head_camera` | 可配 N 路;>1 时各相机 token 在 token 维拼接,并加一个零初始化的可学习 per-camera embedding |
| 归一化 | `utils/stat-500-all.json` | 从 LeRobot 数据现算,输出仍用 `robotwin2` 这个顶层 key(上游 `VLAWrapper.load_norm_stats` 是按名字读的) |
| VTT | 从 HDF5 首末帧算 | 从帧缓存首末帧算,`<out>/<task>/task_cond.npy` 格式不变 |
| register token 数 | 配置缺失时默认 4 | 按编码器 config 读,缺失时按 0(默认 4 只对 DINOv3 成立,换 DINOv2 会静默切掉 4 个真 patch) |
| 训练循环 | `train.py` | `train_lila_wam.py`,schedule / 优化器 / **checkpoint 格式完全一致**,`--resume` / `--init_from` 语义不变 |

上游的建模代码(`models/`)一行没改,是 submodule 原样引用。为了不让
`models` / `utils` / `dataloader` 这些通用名污染评测进程的顶层命名空间,
`_upstream.py` 把 checkout 挂在私有的 `lila_upstream` 命名空间下。

多相机的 camera embedding 是**零初始化**的,所以第 0 步的行为和单相机版本完全一致,
之后才学出来;它挂在 action model 上,因此会进优化器、也会存进 checkpoint。
加它的原因是:DINO patch 特征本身不带"这是哪路相机"的信息,
perceiver adapter 的可学习 query 需要能区分不同视角的 token。

---

## 6. 已验证 / 未验证

已验证:

- `tests/test_lila_wam.py`(12 项,无需 GPU/权重/网络):meta 解析、v3.0 拒绝、
  缓存的集/帧对齐、动作块末尾的 clamp+mask、样本不跨集边界、
  future frame 偏移与 clamp、VTT 装载、各类配置错误的报错。
- `tests/test_multicamera.py`(9 项,stub 掉编码器,同样不需要权重):
  单相机不建 camera embedding(结构与上游一致)、多相机时它零初始化且进
  state_dict / optimizer、token 数随相机数线性增长且每路落在正确的槽位、
  相机数不匹配会报错、梯度能回到 camera embedding。
- 真实数据链路(本机 4090 跑过,数据集 `lerobot_dataset/simrecap_sample_loading_round1`):
  - 建缓存:前 20 集 / 7440 帧 / 三相机 = 14 秒、346 MB;Dataset 取样后
    state 和 action chunk 与源 parquet 逐值对齐。
  - 归一化统计:全部 350 集,349 集入统计、1 集被 [-π, π] 越界过滤器剔除。
  - VTT 预计算:DINOv3 ViT-L/16 加载正常(hidden=1024, patch=16, registers=4),
    产出 dim=1024 的任务向量。
  - 训练:三相机模式起来了(camera embedding 已挂上),可训参数 **211.95M**
    (与论文的 0.2B 对得上),flow-matching + future-feat 两个 loss 都在降,
    checkpoint 正常落盘。
  - 推理:`LilaWamInference` 加载 checkpoint,吃 EmbodiChain 形状的观测
    (`sensor/<cam>/color` 是 batched HWC RGBA、`robot/qpos` 14 维),
    出 (32, 14) 动作块、(16, 14) 执行段,receding-horizon 队列和 reset 行为正确。

未验证:**训练收敛和评测成功率**——那要完整跑一遍 stage1+stage2 再进仿真评测,
不在本次适配范围内。

另外一个已知行为(与上游一致,没有改):流匹配采样出来的动作反归一化后
**不保证落在数据集 min/max 区间内**,上游也不做 clamp。未收敛的模型尤其明显。

---

## 7. 常见问题

**`vision encoder weights not found`** —— 没下 DINOv3,或 config 里的
`model.vision_encoder.checkpoint_path` 指错了。跑
`bash policy/lila_wam/setup_env.sh --download-encoder`。

**`frame cache ... was built for a different dataset/camera/size combination`**
—— 改了数据集/相机/分辨率。加 `--overwrite` 重建。

**显存不够** —— 冻结编码器的前向是大头,和相机数成正比。先降
`training.batch_size`,再考虑加 `training.grad_accum_steps`,或者换单相机 config。

**`missing VTT vector for task 'xxx'`** —— 该任务的条件向量没算。
`python policy/lila_wam/precompute_task_cond.py --config <config.yaml>`,
或者评测时用 `--task_cond_name` 指到已有的任务名。
