# XR-1 (Xiaomi-Robotics-1) 接入 RoboSynChallenge

本目录是把小米 **Xiaomi-Robotics-1** 适配到 RoboSynChallenge 评测框架的全部代码。
不改动比赛官方任何文件；XR-1 上游源码（`Xiaomi-Robotics-1/`）也保持原样，
需要改写的地方一律在运行时打补丁。

```
policy/xr1/
├── deploy_policy.py            # 比赛统一接口 get_model / eval / reset_model
├── deploy_policy.yml           # 评测配置
├── xr1_model.py                # 模型封装：进程内加载 + 前后处理
├── eval.sh                     # 评测入口（5 个位置参数）
├── setup_env.sh                # 建 .venv 装依赖
├── convert_lerobot_to_xr1.py   # LeRobot -> XR-1 训练格式转换器
├── finetune.sh                 # 后训练入口
├── test_roundtrip.py           # 端到端回归测试（转换 -> 上游 Dataset -> 解码）
└── Xiaomi-Robotics-1/          # 上游源码（未改动）
```

---

## 一、最重要的三个结论

### 1. 官方发布的权重不能用 `AutoModel` 加载

上游 `deploy/server.py` 写的是
`AutoModel.from_pretrained(path, trust_remote_code=True)`，但 HuggingFace 上
`XiaomiRobotics/Xiaomi-Robotics-1-5B` 这个 repo 里**只有三个文件**：
`.gitattributes` / `README.md` / `model_states.pt`（10.2 GB）——
没有 `config.json`，没有 remote code，没有 processor。`AutoModel` 根本无从下手。

`model_states.pt` 是 DeepSpeed 风格的裸 state dict（`{"module": {"model.xxx": tensor}}`），
对应的是**另一条**加载路径 `xr1/mibot/server/deploy.py`：

```python
model = MIMODEL.build(cfg.model.params.model)   # 用 mmengine 注册表搭骨架
ckpt  = torch.load(".../mp_rank_00_model_states.pt")
model.load_state_dict(strip_prefix(ckpt["module"], "model."))
```

`xr1_model.py` 走的就是这条路（同时保留了 HF 分支，将来官方若发布 HF 格式可直接用）。
顺带一提，`scripts/deploy.sh` 里调的还是 `deploy/server.py`，与 README 自相矛盾，
上游这块是乱的，别照抄。

### 2. 模型只吐归一化动作，反归一化统计量必须部署侧自带

`xr1.forward()` 在 eval 分支直接 `return self._generate(...)`，出来的是归一化后的
`(30, 60)`。真正的反归一化在 `runtime/server.py` 里：

```python
action = denormalize_action(action * mask, mean, std) * mask
```

而 `mean/std/q01/q99` 存在**训练配置**里（`configs/data/*.yaml` 的 `train_datasets`）。
所以换了数据集就必须换统计量，否则动作幅度整体错。
`convert_lerobot_to_xr1.py` 会把统计量同时写进 `xr1_stats.json`（部署用）和
`xr1_data.yaml`（训练用）；后训练产出的 `config.py` 里也自带一份，
`xr1_model.py` 加载后训练检查点时会优先读它，不用手工同步。

### 3. 60 维动作是末端位姿增量，不是关节 —— 我们做了槽位复用

`mibot/utils/io.py` 的 `ACTION_PARTS` 定死了打包结构：

| 维度 | 原语义 | 本适配层塞的东西 |
|---|---|---|
| 0:3 | 左末端位置增量 | 左臂关节 1-3 |
| 3:6 | 左末端旋转增量（轴角） | 左臂关节 4-6 |
| 6 | 左夹爪 | 左夹爪（原样） |
| 8:11 | 右末端位置增量 | 右臂关节 1-3 |
| 11:14 | 右末端旋转增量（轴角） | 右臂关节 4-6 |
| 14 | 右夹爪 | 右夹爪（原样） |
| 16 | 腰 | 未用（置 0） |
| 17:20 | 底盘速度 | 未用（置 0） |
| 其余 40 维 | 保留 | 恒 0 |

**状态侧不需要任何技巧**：`compose_state()` 本来就是纯关节+夹爪打包
（`state[0:7]`=左臂关节、`state[7]`=左夹爪、`state[8:15]`=右臂关节、`state[15]`=右夹爪），
本赛题 14 维 qpos 正好一一对应，左右各空出第 7 个关节位填 0。

`action_shape = (30, 60)` 和 `state_shape = (1, 60)` 在 `XR1.__init__` 里是写死的，
改不了维度，所以只能复用槽位。

---

## 二、为什么选槽位复用而不是真做末端位姿

考虑过的另一条路是"忠实还原语义"：转换时用 FK 从关节算末端位姿写进 JSON，
推理时把模型输出的末端目标用 IK 解回关节。放弃的原因：

1. **评测回路里要跑 IK**。env.step 收的是绝对关节位置，模型每次吐 30 步，
   意味着每次推理要解 30 次 IK。慢，且 IK 失败/跳解会直接毁掉整条轨迹。
2. **EmbodiChain 的 IK 可用性没验证过**，FK 有 `robot.compute_fk`，IK 没有对等保证。
3. **转换期也要 FK**，等于把转换器绑死在仿真器和机器人模型上；
   数据集和仿真环境不在同一台机器时无法离线转换。
4. 槽位复用的编解码是**严格可逆**的（下面有实测），信息一点不丢。

代价是：**预训练权重零样本输出没有意义**（模型以为自己在输出末端增量，
我们按关节解释），必须微调后才有用。这一点在任务约束里是可接受的。

### 编码方案（严格可逆，非近似）

难点在于上游 `JsonDataset._arm_action` 把相对化写死了，只吃末端位姿字段：

```python
rotm = proprios.{arm}_ee_rotm[frame]                 # 锚点帧
pos_part = rotm.T @ (actions.{arm}_ee_pos[t] - proprios.{arm}_ee_pos[frame])
rot_part = rotm2aa(rotm.T @ actions.{arm}_ee_rotm[t])
```

我们用**指数映射**把关节三元组塞进 SO(3)，让上游那套相对化数学原样成立。
转换器写进 JSON 的是（`s` = `rot_scale`）：

```
proprios.{arm}_ee_pos [f] = q123[f]              proprios.{arm}_ee_rotm[f] = exp(s · q456[f])
actions .{arm}_ee_pos [f] = a123[f]              actions .{arm}_ee_rotm[f] = exp(s · a456[f])
```

于是上游算出来的监督目标天然是（R_a = exp(s · q456[锚点])）：

```
dims[0:3] = R_aᵀ (a123[t] − q123[锚点])
dims[3:6] = log(R_aᵀ · exp(s · a456[t]))
dims[6]   = 夹爪增量
```

`xr1_model.XR1.decode_action` 是它的逆：

```
q123[t] = q123[锚点] + R_a · dims[0:3]
q456[t] = log(R_a · exp(dims[3:6])) / s
夹爪[t] = 夹爪[锚点] + dims[6]
```

两个关键性质：

* **位置槽落在锚点旋转坐标系下**，这正好和真实末端增量的语义同构
  （真实情况也是"增量表达在当前末端坐标系里"），不是硬凑。
* **30 步共用同一个锚点**（上游是 `target[frame:frame+steps] − pos[frame]` 广播），
  不是逐步累积。所以推理时一次加到当前 qpos 上即可，不需要串行累加。

`rot_scale` 由转换器按数据自动定，保证 `‖s · q456‖ < π`（指数映射的单射范围），
留了到 2.8 rad 的安全余量。它会写进 `xr1_stats.json` 和 `xr1_data.yaml`，
部署侧读同一个值，训练/推理不会错配。

### 实测精度

`test_roundtrip.py` 走的是**上游真实的 `JsonDataset`**（不是我自己的复刻），
读转换产物再解码回绝对 qpos，与 LeRobot 原始 `action` 逐点比对：

```
A 上游打包 vs 本地打包   : 4.707e-04     float32 舍入
B 归一化往返精度损失     : 6.333e-08
C 解码回绝对 qpos 目标   : 3.369e-04     弧度，约 0.02 度
状态还原（未截断维）     : 1.192e-07
```

A/C 的残差已定位到底：上游 `rotm2aa_batch` 全程 float32，把本地计算强制降到
float32 后与上游**逐位相同**（`diag_precision.py` 实测 `0.000e+00`）。
也就是说编码语义零差异，残差纯粹是 float32 舍入，0.02 度远低于机器人控制分辨率。

状态那栏还会报"被 q01/q99 截断 N 维"——这是分位数归一化的固有行为
（1% 分位数以外的值本来就要裁剪），不是 bug。

---

## 三、安装

```bash
bash policy/xr1/setup_env.sh                    # 只做评测
bash policy/xr1/setup_env.sh --with-flash-attn  # 还要微调
bash policy/xr1/setup_env.sh --skip-sim         # 只装策略栈，不装仿真栈
```

在 `policy/xr1/.venv` 建 Python 3.11 环境，装 torch 2.8.0 + transformers 4.57.1
（XR-1 明确要求这个版本）+ `xr1`(mibot) 包 + **EmbodiChain 仿真栈**。

**仿真栈是必须的**：`scripts/eval_policy.py` 在**同一个解释器**里同时 import 策略栈
和仿真栈（gymnasium / EmbodiChain / dexsim / robosynchallenge），只装 XR-1 的话
eval.sh 会直接挂在 `import gymnasium`。实跑结果：

```
python: 3.11.14   torch: 2.8.0+cu128   transformers: 4.57.1
mibot / mibot.models.VLA.XR1 / decord 0.6.0 / mmengine 0.10.7 / deepspeed 0.18.9 / lightning 2.5.3  均 ok
flash_attn: 未安装 -> 推理使用 sdpa
```

### 版本锁定（三个必须锁的地方）

| 包 | 锁定值 | 不锁会怎样 |
|---|---|---|
| `warp-lang` | **1.14.0** | 1.16.0 会让 dexsim 的 `_patched_findsource` 无限递归，`import dexsim` 直接 **SIGSEGV(139)**，无任何报错信息 |
| `gymnasium` | 0.29.1 | EmbodiChain 只写 `>=0.29.1`，会装到 1.x；1.x 有 API 变更，对齐 pi05 已跑通的版本 |
| `toppra` / `polars` | 0.6.3 / 1.31.0 | EmbodiChain pyproject 里的显式 pin，但我们是 `--no-deps` 装的它，得手动落实 |

装仿真栈时用 constraint 文件锁死 `torch==2.8.0` / `torchvision==0.23.0` /
`transformers==4.57.1` / `numpy==2.1.3` / `warp-lang==1.14.0`，
任何要动这几个的依赖一律退回 `--no-deps`。**实测零版本妥协**：
装完仿真栈后 torch 仍是 2.8.0+cu128、transformers 4.57.1、numpy 2.1.3。

### pytorch_kinematics 必须复制，不能 pip 装

EmbodiChain 的 solver 会调 `Chain.forward_kinematics_tensor()`，
而**这个方法只存在于比赛仓库根部 `.venv` 里那份打过补丁的构建**
（`pytorch_kinematics/chain.py:507`）。PyPI 上的 `0.10.0` 和 EmbodiChain
警告里提到的 `0.7.6` 都没有它，pip 装上去会在 solver 初始化时报 AttributeError。

**最阴的地方是打补丁那份的版本号也写作 `0.10.0`**，光看 `pip list` 分辨不出来。
唯一可靠的判据是这个方法在不在：

```bash
policy/xr1/.venv/bin/python -c \
  "import pytorch_kinematics as pk; print(hasattr(pk.chain.Chain,'forward_kinematics_tensor'))"
# 必须是 True
```

`setup_env.sh` 因此是从 `$BASELINE_VENV`（默认 `RoboSynChallenge/.venv`）
整包复制 `pytorch_kinematics/` 和 `pytorch_kinematics-*.dist-info/`，
并在校验段落里对这个方法做了断言。换机器时若基准 venv 不在原处，
用 `BASELINE_VENV=<path> bash policy/xr1/setup_env.sh` 指定。

顺带澄清一个误判：numpy 2.1.3 与 dexsim/EmbodiChain **没有** ABI 冲突
（pi05 用 numpy 1.26.4 纯属它自己的依赖需要），上面那个段错误是 warp 版本问题，不是 numpy。

两个踩过的坑已在脚本里处理：

* **deepspeed 构建期要 nvcc**。这台机器没装系统级 CUDA，脚本会自动去
  `/usr/local/cuda*` 和各 conda env 里找（本机命中
  `/home/phl/miniconda3/envs/RoboTwin`，CUDA 12.1）。实在找不到就自动跳过
  deepspeed，只保证推理可用。
* **安装顺序**。deepspeed 的 `setup.py` 在构建期就 `import torch`，
  所以先装 torch 再用 `--no-build-isolation` 装其余的。

### GitHub release 直连极慢，务必走加速镜像

flash-attn 的预编译 wheel 只发在 GitHub release（PyPI 上只有源码包），
而这台机器直连 GitHub CDN 实测只有 **24~27 KB/s**，250MB 要下 2.5 小时。
换加速镜像后是 **1.2~1.3 MB/s，快约 45 倍**，三分钟下完：

```bash
W="flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
curl -sL -o "$W" \
  "https://gh-proxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/$W"
python3 -c "import zipfile; zipfile.ZipFile('$W').testzip()"   # 装之前先验完整性
uv pip install --python .venv/bin/python --no-deps "$W"
```

可用镜像：`gh-proxy.com`、`ghfast.top`（都是 `https://<镜像>/<完整 GitHub URL>` 的形式）。
**这条对所有 GitHub-release-only 的包都适用**，别再傻等直连。

wheel 文件名必须与环境三项对齐，错一项就装不上或 import 崩：
`cu12torch2.8`（torch 版本）、`cxx11abiTRUE`（`torch._C._GLIBCXX_USE_CXX11_ABI`）、`cp311`（Python）。

### flash-attn：推理可选，训练必须

* **推理可以不装**。`attn_implementation: auto` 检测不到就退回 sdpa。
  评测时 batch=1、单条序列，sdpa 结果正确。
* **训练必须装**。`CustomCollate` 会把一个 batch 的多条样本
  **拼接成一条长序列**，靠 `cu_seq_lens_q/k` 做变长注意力隔离，
  而这套参数只有 flash-attention 的 varlen kernel 认。
  换成 sdpa **不会报错**，但样本之间会互相注意到，等于训了个错的东西。
  `finetune.sh` 因此会硬性检查 flash-attn，缺了直接拒绝启动。

---

## 四、评测

```bash
bash policy/xr1/eval.sh <task> <setting> <train_config> <model_name> <gpu> [--overrides...]

# 微调后
bash policy/xr1/eval.sh click_bell random click_bell posttrain 0 --max_episodes 50
# 零样本（只验证链路，动作语义无意义，建议限幅防止把仿真打飞）
bash policy/xr1/eval.sh click_bell random none none 0 --max_joint_delta 0.2
```

权重解析优先级：`model_path` > `checkpoints/<train_config>/<model_name>` > `pretrained_ckpt`。

`deploy_policy.yml` 里几个要注意的字段：

| 字段 | 说明 |
|---|---|
| `backbone_path` | 本地 Qwen3-VL-4B-Instruct 目录。**必填** |
| `stats_path` | 转换器产出的 `xr1_stats.json`。留空会先按 `train_config_name` 去 `training_data/<name>/xr1_stats.json` 自动找，再找不到才退回官方 demo 的 `load_washer.yaml`（只能验证链路） |
| `xr1_step` | 每次推理执行几步，默认 10（模型 horizon 是 30 步 @ 25fps = 1.2s） |
| `max_joint_delta` | 每步关节增量上限，`null` 不限。零样本时建议 0.2 |

前处理与上游 `runtime/client.py` 逐字对齐：三路图各自
`resize_image(factor=32, max_pixels=160000)`（640×480 → 448×320），
prompt 措辞、`/no_cot` 后缀、`<cot></cot>` 助手轮一个字都没改——
这些进了 tokenizer，改了就和训练分布对不上。

**backbone 其实只用到配置和 tokenizer/processor**：`XR1._build_model` 走的是
`Qwen3VLForConditionalGeneration._from_config(...)`，只读 config 随机初始化，
权重全部来自 `model_states.pt`。所以 backbone 目录缺 safetensors 也能跑，
但 `config.json` / tokenizer / preprocessor 那几个文件必须齐。

---

## 五、训练数据转换

```bash
policy/xr1/.venv/bin/python policy/xr1/convert_lerobot_to_xr1.py \
    --repo_dir lerobot_dataset/click_bell_aug_base \
    --out_dir  policy/xr1/training_data/click_bell \
    --instruction "Click the bell"
```

产出 `data/episode_XXXXXX.json`（每集一个）、`videos/`、`xr1_stats.json`、`xr1_data.yaml`。

几个实现要点：

* **比赛数据是 LeRobot v3.0，不是 v2.1**。v3 把整个 chunk 的所有集拼进
  一个 parquet 和一个 mp4，靠 `meta/episodes/*.parquet` 里的
  `dataset_from_index` / `videos/<key>/from_timestamp` 定位。转换器两种布局都支持。
* **视频必须转码**。比赛数据是 **AV1**，decord 0.6.0 打不开，
  实测报 `cannot find video stream with wanted index: -1`。
  默认 `--video_mode transcode` 会把每个**源视频文件整段**转成 h264
  （不是按集裁剪——整段转一次既避免按时间戳 seek 的对齐风险，也快得多），
  各集仍用 `start` 帧偏移索引，转码后会校验帧数一致，不一致直接中止。
  数据本身若是 h264 可以用 `--video_mode link` 跳过转码。
* **统计量只统计真实存在的步**。上游对超出集尾的步做"重复末帧"填充，
  但 `action_mask` 把它们排除在损失外，所以归一化统计也必须排除，
  否则短集的末帧会被重复计入把 mean/std 拉偏。
* **恒定维会被压成 padding**。`validate_quantiles` 要求非填充维必须
  `q99 > q01`，而像 click_bell 这种夹爪全程不动的任务会触发断言，
  转换器统一把这类维的 q01/q99 置 0 走 padding 分支。

---

## 六、后训练

```bash
bash policy/xr1/finetune.sh policy/xr1/training_data/click_bell click_bell 0 \
     trainer.max_steps=20000
```

位置参数是 `<数据目录> [实验名] [GPU] [hydra overrides...]`，
输出落在 `policy/xr1/checkpoints/project_<PROJECT>/<实验名>/`，
里面的 `config.py` + `last.ckpt/checkpoint/mp_rank_00_model_states.pt`
正好是 `xr1_model.py` 认的后训练布局，评测时 `--model_path` 指过去即可。

处理掉的几件事：

* **上游 README 让跑 `scripts/train.sh`，但发布的仓库里没有这个文件**。
  这里直接调 hydra 入口 `tools/train.py`。
* **自定义数据配置**通过 `hydra.searchpath` 注入
  （拷到 `policy/xr1/configs/data/<名字>.yaml`），不用往上游 `configs/` 里塞东西。
  已实测 `--cfg job` 能正确 compose 出 data/model/trainer 三段。
* **`Qwen/Qwen3-VL-4B-Instruct` 在源码里是写死的 hub id**
  （`XR1._build_model` 和 `CustomCollate` 都有）。训练侧没法像推理那样打补丁，
  脚本会在 `policy/xr1/.hf_home` 下造一份指向本地目录的缓存软链，
  配合 `HF_HUB_OFFLINE=1` 让离线机器也能解析。
* **必须在 `Xiaomi-Robotics-1/xr1/` 目录下运行**：`process_save_cfg` 会往
  `./assets/config.py` 写一份配置，cwd 不对会直接报错。
* `WANDB_MODE` 默认 `offline`，免得没登录时卡住。

---

## 七、已知风险 / 没做完的事

1. **没有跑过真正的推理和训练**（任务范围不含冒烟测试）。

   已验证（都在 CPU 上实跑过）：
   - 环境装得起来，`mibot.models.VLA.XR1` 能 import
   - 数据转换与上游 `JsonDataset` 的对接、decord 能读转码后的视频
   - 编解码数值精度（见上）
   - hydra 配置能 compose 出完整的 data/model/trainer 三段
   - **权重键结构吻合**：`model_states.pt` 共 1135 条参数，全部带 `model.` 前缀，
     剥掉后的 11 个顶层子模块与 `xr1._build_model` 定义的完全一致（无缺无余），
     DiT 36 层、VLM text 36 层（`DiT.forward` 要求前者 ≤ 后者）。
     `load_state_dict(strict=True)` 应当能过。

   **仍未验证**：真实前向、显存占用、推理延迟、训练能否收敛。

2. **零样本必然不可用**。槽位复用让预训练权重的输出语义完全对不上，
   接口和形状是对的，数值是胡说。必须微调。
   零样本试跑请务必带 `--max_joint_delta 0.2`。

3. **显存**。5B 模型 bf16 权重约 10 GB（`model_states.pt` 实际 10.2 GB），
   加上 DiT 激活和 KV cache，4090 48G 单卡评测应该够；
   训练要 deepspeed + 梯度检查点，能跑多大 batch 需实测。

4. **`MAX_LENGTH` 会静默丢样本，这是最容易踩的坑**。
   `CustomCollate` 对超出长度预算的样本直接 `continue`，**不报错也不警告**。
   实测单样本 **484 token**（三路 640×480 → 448×320，每图
   `grid=[1,20,28]` → 140 token，三图 420，加 prompt 约 64；
   训练样本再多约 35 个 `<state>`/`<a_i>`/`<score>`）：

   | MAX_LENGTH | 一个 batch 实际装得下 |
   |---|---|
   | 4096（默认） | 8 条 |
   | 8192 | 16 条 |
   | 16384 | 33 条 |

   所以官方 demo 的 `batch_size: 48` 配默认 `MAX_LENGTH` 会白白丢掉 40/48。
   转换器生成的 yaml 默认写 `batch_size: 8`；要加大就同步调
   `MAX_LENGTH=16384 bash policy/xr1/finetune.sh ...` 并把 `batch_size` 一起提上去。
5. **`freq_excluded_dims: [17,18,19]`** 是底盘速度维，我们没用到，保持默认即可。
6. **夹爪范围**。解码后按 `[0, 0.05]` 米裁剪（`DEFAULT_GRIPPER_LIMITS`），
   若换机器人要同步改。
7. **`rot_scale` 必须训练/推理一致**。它跟着 `xr1_stats.json` 和后训练
   `config.py` 走；如果手工换了统计量文件，务必确认 `rot_scale` 也是配套的，
   否则关节 4-6 会整体缩放错。
8. **多任务混合训练**没做。当前转换器一次处理一个数据集目录、一条指令。
   要混合多任务，把多个 `out_dir/data` 一起写进 `paths`，
   但统计量得重新在合并集上算（现在没有合并统计的工具）。

---

## 八、EEF 路线（真实末端位姿编码）

槽位复用（第二章）和 EEF 是**两条可 A/B 的路线**，由转换器的 `--encoding` 选择，
部署侧 `decode_mode: auto` 会自动跟随 stats 里记录的编码方式，不用手工对齐。

| | `--encoding slot` | `--encoding eef`（默认） |
|---|---|---|
| 动作槽位含义 | 关节 1-3 / 4-6 搬进末端槽位 | 真实末端位置增量 + 轴角增量 |
| 与预训练语义 | 不一致，零样本无意义 | **一致**，能吃到预训练先验 |
| 依赖 | 无 | URDF + FK（转换期）、IK（部署期） |
| 部署解码 | 增量直接加到关节 | 末端目标 → `compute_ik` |
| `rot_scale` | 按数据自动定，保证 ‖s·q456‖<π | 恒为 1（真旋转不缩放） |
| 风险 | 语义错，必须微调 | IK 可能失败 |

### FK 的三个关键事实（都从源码核实，别凭直觉改）

1. **末端不是 gripper_base**。EmbodiChain 给 CobotMagic 配的是
   `end_link=<side>_link6`、`root=<side>_arm_base`，FK/IK 串链用的是单臂
   URDF `CobotMagicArm/CobotMagicNoGripper.urdf`（链名是通用的 `base_link`→`link6`，
   左右臂共用同一条链，差异只在基座变换）。
2. **有一个 TCP 变换**：绕 Z 转 180° + Z 偏移 0.143 m，`get_fk()` 返回的是**乘过 TCP** 的位姿。
   踩过的坑：`OPWSolver.__init__` 把 TCP 初始化成单位阵，cfg 里的 `tcp` 只有走
   `OPWSolverCfg` 的工厂方法才生效；直接构造解算器会差整整一个 TCP
   （表现为恒定 143 mm + 180° 偏差）。**TCP 是右乘，不会在增量编码里抵消**，必须两侧一致。
3. **坐标系不是问题**。`compute_fk` 返回 arena 系，`xr1_fk.py` 返回臂基座系，
   两者差一个常量刚体变换 T。但 XR-1 编码的是相对锚点帧的增量：
   `p_rel = R_aᵀ(p_t − p_a)`、`R_rel = R_aᵀR_t`，在 T 左乘下严格不变
   （`validate_fk.py` 实测 1.5e-15）。所以训练数据用臂基座系完全等价，
   **不需要 arena→base 外参**；部署侧则全程走 arena 系，自洽。

### 交叉验证结果

`validate_fk.py` 拿 200 帧真实数据，比对两套**完全独立**的实现
（pytorch_kinematics URDF 链式乘法 vs EmbodiChain OPW 解析公式）：

```
配置               位置误差 max/mean (mm)      旋转误差 max/mean (deg)
left/state         0.0001 / 0.0000            0.0379 / 0.0146
right/state        0.0001 / 0.0000            0.0371 / 0.0148
left/action        0.0001 / 0.0000            0.0357 / 0.0129
right/action       0.0001 / 0.0000            0.0340 / 0.0098
```

`test_eef_roundtrip.py` 再走一遍**上游真实 JsonDataset**，把打包动作解回绝对末端位姿
与 FK(action) 逐点比对：**位置 0.00004 mm / 旋转 0.024°**。旋转那 0.02~0.04° 是
OPW 走 float32、PK 走 float64 的精度差，不是语义差。

### 命令

```bash
# 转换（1000 集约 15 分钟，视频转码 20 并行）
policy/xr1/.venv/bin/python policy/xr1/convert_lerobot_to_xr1.py \
    --repo_dir /home/phl/workspace/datasets/cobotmagic_Sim_sample_loading \
    --out_dir  policy/xr1/training_data/sample_loading_eef \
    --encoding eef --video_workers 20 \
    --instruction "Pick up the test tube, and it to the other arm, and insert it to the rack."

# 微调
bash policy/xr1/finetune.sh policy/xr1/training_data/sample_loading_eef sample_loading_eef 0

# 看 loss（wandb offline，summary 要很久才 flush，这个直接读 datastore）
policy/xr1/.venv/bin/python policy/xr1/read_train_loss.py

# 评测
bash policy/xr1/eval.sh sample_loading random sample_loading_eef posttrain 0
```

### 训练：为什么必须冻结 VLM

实测参数量 5.50B = VLM 4.83B + DiT/projector 0.68B：

| 方案 | 显存需求 | 单卡 4090（剩 37G）|
|---|---|---|
| 全参微调 | 权重 11G + 梯度 11G + Adam 66G ≈ **88G** | 放不下；CPU RAM 也只有 44G，offload 同样放不下 |
| **冻结 VLM** | 权重 11G + Adam 8.1G + 激活 ≈ **34G** | 可行（实测稳定占 45.6/49.1G，含用户任务 11.4G）|

入口是 `policy/xr1/train_xr1.py`（注册 `FrozenVLMRunner`），**没改上游 `tools/train.py`**；
要全参微调（多卡）用 `RUNNER=BaseRunner`。这也是 VLA 后训练的常规做法：
VLM 当冻结的视觉-语言特征提取器，动作专家去适配新本体。

两个环境坑已固化进 `finetune.sh`：
* **deepspeed import 期就要 `CUDA_HOME`**（探测 CUDA op 兼容性），没有会抛
  `MissingCUDAException`，看起来像缺包其实不是。已自动探测。
* **fused_adam JIT 编译**：CUDA 12.1 的 nvcc 只认 gcc≤12，Ubuntu 24.04 是 gcc 13。
  已设 `CUDAHOSTCXX=/usr/bin/g++-12`；另留 `OPTIMIZER=torch.optim.AdamW` 兜底（不需编译）。

### 部署：IK 解码与失败策略

`decode_mode: eef_ik` 时每步：

1. `robot.compute_fk(当前臂关节, name="{side}_arm")` 拿 arena 系当前末端位姿 `(p_a, R_a)`；
2. `p_t = p_a + R_a·dims[0:3]`、`R_t = R_a·exp(dims[3:6])`
   （30 步共用同一锚点，上游是广播差分不是逐步累积）；
3. `robot.compute_ik(pose=T_t, joint_seed=上一步的解, name="{side}_arm")`；
4. **返回码 == 1 才接受**；失败则保持上一步关节角并计数，避免跳变。
   种子始终用上一步的解，保证轨迹连续。

失败率通过 `model.ik_failure_rate()` 暴露，每集 reset 时打印
`[XR1] 上一集 IK 失败率 x/y = z%`。

### 解码级重放验证（pro6000，20 集）

静态验证（数值往返）不等于能用，所以补了一轮**重放**：读转换后 JSON 的末端位姿 →
上游同款相对化 → 本适配层的 decode 数学 → EmbodiChain OPW `get_ik` 逐步解
（种子链式，与部署实现一致）→ 重建 qpos → 与原始数据集 action 逐帧比对。
脚本 `replay_eef_ik.py`，20 集均匀跨 1000 集，锚点步长 30（整段 horizon，
是最难的情形；部署每 10 步就重锚一次，真实误差只会更小）。

**① 重建误差（rad）**

| | median | p95 | max |
|---|---|---|---|
| 全部 12 关节 | 1.34e-05 | 4.44e-04 | **1.06e-02** |

误差极小（p95 = 4.4e-04 rad ≈ 0.025°），max 落在 `left_j4` / `left_j6`
两个腕关节上——腕部近奇异时关节大幅变化只对应末端极小变化，属尾部事件。

**② IK 失败率：302/14880 = 2.03%**

**③ 最差 3 集**：ep000105 (max 1.06e-02)、ep000789 (6.63e-03)、ep000684 (5.06e-03)——
注意这三集 **IK 失败均为 0**，说明「误差」和「失败」是解耦的两件事。

**④ 判定：按严格阈值（max<0.01 rad 且失败=0）为 FAIL**，但根因已查清，
不是转换器或解码器的 bug：

1. **目标本身完全可达**。目标位姿是由真实关节角经 FK 生成的，
   直接拿 JSON 原始位姿做 IK 是 **0/14880 失败**。
2. **微扰扫描**定位了敏感度：给原始目标加噪声，
   `1e-7 → 0%`、`1e-6 → 0.68%`、`1e-5 → 2.62%`。
   即 OPW 的接受判据在这批目标上处于**刀刃状态**，1e-6 量级扰动就开始拒解。
3. **扰动来自上游的 float32 对数映射**。`mibot.utils.io.rotm2aa_batch` 全程 float32，
   而本数据集每步相对旋转角**中位数只有 0.0012 rad**；`arccos` 在 1.0 附近病态，
   float32 会把误差放大到约 1e-4 rad，解码后目标位姿偏差约 1e-5 m——
   正好落在上面 2.62% 那一档，与实测 2.03% 吻合。
   （用我自己的 float64 简化实现反而更差：4.29%，因为它缺 θ≈π 分支；
   重放脚本已改成与生产链路逐行一致的 float32 版，否则测的就不是真实链路。）

**对部署的意义**：这 2% 是**数值层面的地板值**，不是模型带来的。实际评测时
模型自身的预测误差远大于 1e-5，IK 失败将主要由模型误差主导。
现有兜底（失败则保持上一步关节角、种子始终用上一步解）足以吸收这类单步失败——
证据是误差最大的三集恰好零失败，两者不相关。

若冒烟时失败率明显高于 2%，优先试这两个缓解（均未实测）：
失败时改用锚点 qpos 重试一次 IK；或对解码出的旋转矩阵做 SVD 重正交化后再送 IK。

### 在 pro6000 上建评估环境（与 4090 的四处差异）

4090 被训练占满时，早期冒烟放在 pro6000 做。装法一样，但有四处要注意：

```bash
R=/workspace/users/fmc3-8-workspace/Chen/robosynchallenge/RoboSynChallenge
cd "$R" && BASELINE_VENV="$R/.venv" UV_BIN=/snap/bin/uv \
    bash policy/xr1/setup_env.sh --skip-deepspeed
```

1. **`UV_BIN=/snap/bin/uv`**：pro6000 的 uv 装在 snap 里，不在 `~/.local/bin`。
2. **`--skip-deepspeed`**：只做推理，不需要 deepspeed，省掉 nvcc 依赖。
   推理也不需要 flash-attn（sdpa 结果正确，batch=1 无序列打包）。
3. **dexsim 私有源装不上**：pro6000 的 snap uv 对 HTTP(非 HTTPS) 源处理不同，
   会报 "not found in registry" 即使源可达（curl 实测 HTTP 200）。
   `setup_env.sh` 已加兜底——自动从 `$BASELINE_VENV` 整包复制 dexsim，
   和 pytorch_kinematics 同一套路。
4. **两个额外依赖**：`lxml` 和 `arm_pytorch_utilities`，都是 pytorch_kinematics
   的运行时依赖。因为 pk 是复制进来的（不走 pip），它的依赖不会被自动解析，
   得手动补。4090 上是被别的包顺带装上了才没暴露。

**权重路径跨机器自动重定位**：两台机器的模型根不同
（`/home/phl/workspace/models` vs `/home/fmc3-0|fmc3-8/workspace/models`），
但 `deploy_policy.yml` 只有一份。`deploy_policy.py` 会在配置路径不存在时，
把 `models/` 之后的部分接到本机候选根上重试，并打印重定位日志，
所以**不需要为换机器改 yml**（改了另一台就坏）。可用 `XR1_MODELS_ROOT` 强制指定。

pro6000 实测验收（`RTX PRO 6000 Blackwell, sm_120`）：

```
dexsim 0.4.3 / eval_policy 链 + xr1 adapter / mibot.models.VLA.XR1   均 OK
torch=2.8.0+cu128  transformers=4.57.1  numpy=2.1.3  warp=1.14.0
gymnasium=0.29.1   embodichain=0.2.3    pytorch_kinematics 定制版=True
xr1_fk 可用（零位末端 [0.1986, 0.0, 0.2257]）
GPU sm_120 bf16 matmul 实跑通过   ← torch 2.8+cu128 支持 Blackwell，已验证不是只能 import
```

### 夹爪量程必须从数据里读

sample_loading 的夹爪是 **0~1 归一化**，不是最初以为的 0~0.05 米
（click_bell 夹爪全程为 0，所以那边看不出来）。硬编码 `(0, 0.05)` 会把夹爪指令
整个裁掉。转换器现在把实测量程写进 `xr1_stats.json`，部署侧自动读取；
全量 sample_loading 实测是 `[-0.0385, 1.0861]`。
