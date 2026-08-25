# 用 RLinf 对 pi0.5 做 PPO / GRPO 后训练

把 RoboSynChallenge 的仿真任务接进 [RLinf](https://github.com/RLinf/RLinf)，对已有的 pi0.5
SFT checkpoint 做强化学习后训练。奖励直接用官方 `is_task_success()`，不自定义判定、不做
reward shaping。

RLinf 在 LIBERO 上报的数字是 pi0.5 从 77.1% 提到 97.9%；那是它自己的 SFT 基座和任务集，
这里能到多少要自己跑出来。

## 为什么要转成 PyTorch

RLinf **从第一个 commit 起就是纯 PyTorch**（FSDP / Megatron），全仓 689 个 commit 里只有两个
文件跟 JAX 有关，都是 checkpoint 转换器。它对 π₀/π₀.₅ 的支持方式一直是"把 JAX 权重转成
PyTorch，用自己的重实现训练"，没有 JAX 训练路径可用。

好消息是转换是**单向**的：openpi 的 `create_trained_policy` 靠目录里有没有 `model.safetensors`
自动切后端，两条路走完全相同的 transform 和 norm_stats。所以 RL 训完的 PyTorch checkpoint
可以直接放回 `policy/pi05/checkpoints/<config>/<model>/<step>/`，用现成的 `eval.sh` 评测，
不需要转回 JAX。

## 前置

| 组件 | 位置 | 说明 |
|---|---|---|
| RLinf | `~/workspace/RLinf` | `RLINF_ROOT` 可覆盖 |
| EmbodiChain | `~/workspace/EmbodiChain` | `EMBODICHAIN_ROOT` 可覆盖；worktree 下同级目录假设不成立，脚本会自动回退查找 |
| pi0.5 SFT checkpoint | `policy/pi05/checkpoints/...` | openpi 的 orbax 格式 |

RLinf 环境安装见其官方文档（`bash requirements/install.sh embodied --model openpi --env embodichain`）。
**注意**：它默认从 DexForce 私有源装 `embodichain>=0.2.4` 的 wheel。本仓库的任务是针对
`~/workspace/EmbodiChain` 这个工作副本写的，建议在 RLinf 的 venv 里改成 editable 安装本地版本，
否则任务代码和引擎版本可能对不上。

## 第一步：转换 checkpoint

```bash
policy/pi05/.venv/bin/python scripts/convert_pi05_jax_to_torch.py \
    --checkpoint-dir policy/pi05/checkpoints/pi05_base_robosynchallenge_full/mixer_operating/28000 \
    --config-name pi05_base_robosynchallenge_full \
    --output-path /home/phl/workspace/models/pi05_pt/mixer_operating_28000
```

产出 `model.safetensors`（bf16 约 7.2G）+ `config.json` + `assets/`（norm_stats）。

这个脚本包装了 RLinf 的转换器，替它绕开两个坑：

1. **openpi 包被遮蔽**。RLinf 的转换器同目录下有个也叫 `openpi` 的包，直接
   `python .../convert_openpi_jax_to_python.py` 运行时 `sys.path[0]` 是脚本目录，
   会把真正的 openpi 顶掉，报 `ModuleNotFoundError: No module named 'openpi.models'`。
2. **assets 拷不到**。它从 `checkpoint_dir.parent/assets` 找 norm_stats，而 openpi 训练产物的
   布局是 `<step>/assets`（与 `<step>/params` 同级）。拷不到就没有归一化统计量，策略输出会完全错，
   而且不报错。

另外脚本加了一道硬校验：`--checkpoint-dir` 路径里不含 `pi05` 直接拒绝运行。转换器靠这个字符串
判断走 pi0.5 的 adaptive-norm 分支还是 pi0 的 RMSNorm 分支，判断错会**静默**产出一个结构合法
但权重错位的模型。

**首次运行前**要给 openpi 的 venv 打 transformers 补丁（openpi 的 PyTorch 路径要求）：

```bash
SP=policy/pi05/.venv/lib/python3.11/site-packages/transformers
cp -r policy/pi05/src/openpi/models_pytorch/transformers_replace/* "$SP"/
```
覆盖 siglip / gemma / paligemma 共 5 个文件，需要 `transformers==4.53.2`。改前建议备份原文件。

### 验证转换没出错

```bash
V=policy/pi05/.venv/bin/python
NS=<输出目录>/assets/RoboSynChallenge/cobotmagic_Sim_mixer_operating

$V scripts/verify_pi05_torch_vs_jax.py --backend jax \
    --checkpoint-dir <JAX 的 28000 目录> --config-name pi05_base_robosynchallenge_full \
    --norm-stats-dir "$NS" --out /tmp/jax.npz
$V scripts/verify_pi05_torch_vs_jax.py --backend torch --no-compile \
    --checkpoint-dir <转换输出目录> --config-name pi05_base_robosynchallenge_full \
    --norm-stats-dir "$NS" --out /tmp/torch.npz
$V scripts/verify_pi05_torch_vs_jax.py --compare /tmp/jax.npz /tmp/torch.npz
```

两边喂完全相同的 observation、prompt、flow-matching 初始噪声和 norm_stats。

**怎么判读结果**：不要拿绝对误差跟一个拍脑袋的阈值比，而是跟这条流水线自身的可复现性下限比。
mixer_operating/28000 的实测：

| 对比 | 最大绝对误差 | 余弦相似度 |
|---|---|---|
| torch fp32 vs bf16（同后端，仅精度） | 3.2e-03 | 0.99999988 |
| JAX vs torch（跨后端） | 3.1e-02 | 0.99997944 |
| torch 开 compile vs 关 compile（**同后端同权重**） | **5.6e-02** | 0.99993250 |

跨后端误差比"同一后端开关 `torch.compile`"还小 —— 转换引入的偏差低于流水线自己跟自己的一致性。
闭环验收：转换后 checkpoint 在 mixer_operating random 上 20 集 **16/20 = 80%**，JAX 基线 85%。

`--norm-stats-dir` 必须显式给：`create_trained_policy` 默认按 config 的 `repo_id` 找 assets 子目录，
而 base config 的 repo_id 是 click_bell，多任务基座下各任务 checkpoint 的 assets 目录名却是各自的
任务名，对不上。用 per-task config（`pi05_mixer_operating`）时能自动找到。

## 第二步：给 RLinf 打补丁

```bash
python scripts/patch_rlinf_env.py --rlinf-root ~/workspace/RLinf
python scripts/patch_rlinf_env.py --check     # 幂等，可反复跑
python scripts/patch_rlinf_env.py --revert    # 撤销
```

RLinf 的 `get_env_cls()` 是硬编码的枚举映射，没有插件注册口，所以必须改它两处（共 20 行）：

- `rlinf/envs/__init__.py`：加 `ROBOSYNCHALLENGE` 枚举 + 一个返回我们环境类的分支
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`：注册 `pi05_robosynchallenge` 配置

实现全部在本仓库的 `robosynchallenge/rlinf_env/`，RLinf 那边只是挂钩。升级 RLinf 后重跑一次即可。

## 第三步：训练

```bash
export ROBOSYN_PI05_TORCH_CKPT=/home/phl/workspace/models/pi05_pt/mixer_operating_28000

bash launch/rlinf_train.sh ppo   --dry-run   # 先做前置检查，不启动
bash launch/rlinf_train.sh ppo
bash launch/rlinf_train.sh grpo
```

启动脚本负责把 `examples/rlinf/*.yaml` 链进 RLinf 的 config 目录（hydra 只从那里找配置），
并设好 `PYTHONPATH` / `EMBODICHAIN_PATH` / `ROBOT_PLATFORM=ALOHA` / EGL 离屏渲染。
`--dry-run` 会把所有前置条件检查一遍并给出具体修复命令 —— 这些条件少一个都会让训练在
跑起来几分钟后才以难懂的方式失败。

## 换任务

改 `examples/rlinf/robosynchallenge_ppo_pi05.yaml` 里 `env.train` / `env.eval` 的两处：

```yaml
gym_config_path: ${oc.env:ROBOSYN_PATH}/configs/<task>/random/gym_config.json
max_episode_steps: <官方值>
```

`max_episode_steps` 必须用官方值（取自各任务 `random/gym_config.json`，对应上游的
`fix(eval): honor task episode limits`）：

| task | max_episode_steps |
|---|---|
| mixer_operating / water_pouring | 500 |
| click_bell / table_rearrangement | 361 |
| drawer_open_place | 900 |

同时把 `robosynchallenge/rlinf_env/dataconfig.py` 里 `register()` 的 `repo_id` 换成对应任务
（决定 norm_stats 在 `<checkpoint>/assets/<repo_id>/` 下的查找路径，换错会加载到别的任务的
归一化统计量）。

### 先打哪个任务

PPO 靠"偶尔成功 → 拉高该轨迹概率"，稀疏奖励下基线成功率太低就起不来。按官方最新基线
（`recalculated_at: 2026-08-21`）和你自己 README 里的 pi0.5 数字：

- **适合起步**：mixer_operating、water_pouring、items_handover、table_rearrangement、
  click_bell、manipulate_pipette —— pi0.5 基线 71%~85%，和 RLinf 验证过的 LIBERO 77.1% 同区间
- **不适合**：item_assembly（16%）、sample_loading（3%）—— 32 路并行一轮平均只有 1 条成功轨迹，
  信噪比撑不起策略梯度

## 奖励与终止怎么接的

链路：`EmbodiedEnv.get_info()` → `compute_task_state()` → `{success, fail, elapsed_steps, metrics}`，
然后 `base_env.step()` 里 `terminateds = success | fail`。RLinf 的 `_record_metrics` 读的正是
`infos["success"]` / `infos["fail"]` —— 两边接口天然对齐。

唯一的错位是：RoboSynChallenge 的任务为了数据采集，故意把 `compute_task_state` 的 success 压成
全 False 让 episode 跑满，真实成败只由官方 `is_task_success()` 给出。

`install_official_reward()` 在**环境实例**上补这个差（不动 `robosynchallenge/tasks/` 下任何文件
—— 那是官方赛题代码，改了会和上游冲突）：

- `compute_task_state` 改成返回官方 `is_task_success()`，于是成功当步就终止 episode
- `get_reward` 给稀疏奖励：成功那一步 1.0，其余 0

**关于重复调用的安全性**：原 `compute_task_state` 和 `is_task_success` 都可能触碰有副作用的统计
（如 mixer_operating 的 `_update_button_contact_history`）。官方实现本身就假定
`is_task_success` 会被反复调用并做了防重复计数 —— handle_basket 用
`_hb_last_success_check_env_step` 按环境步差累加而不是每次 `+= 1`。官方评测循环也同样在每步之外
额外调用它。所以这个用法是被官方代码支持的。

日志里会同时记 `official_success` 和 `task_reported_success`，两者长期不一致就说明这个假设在某个
任务上不成立，需要查。

## 观测通路

RLinf 的 actor 吃的契约（`RLinf/rlinf/data/schema/embodied_types.py:111-117`）：

```
main_images        [N_ENV, H, W, C]
wrist_images       [N_ENV, H, W, C] 或 [N_ENV, N_IMG, H, W, C]   <- 双腕走后者
states             [N_ENV, D]
task_descriptions  list[str]
```

RLinf 自带的 EmbodiChain 适配器只产出 `{"states": ...}`（它是为 CartPole 写的），
`RoboSynChallengeVLAEnv._wrap_obs` 补齐了图像和语言：三路相机 `cam_high` /
`cam_left_wrist` / `cam_right_wrist`，RGBA 切成 RGB，在**环境侧**缩放到 224×224
（原图 640×480，不缩放跨进程要多搬 4.6 倍数据）。

## 一个必须知道的配置陷阱

`actor.model.openpi.config_name` 要写 **`pi05_robosynchallenge`**，不要用 RLinf 自带的
`pi05_aloha_robotwin`。

两者相机 uid 和 14 维双臂动作空间完全一致，delta mask 也都是 `make_bool_mask(6, -1, 6, -1)`
（每臂 6 关节做 delta、夹爪绝对），看起来可以直接套用。但 robotwin 那份用的是
`aloha_policy.AlohaInputs(adapt_to_pi=True)`，会做 ALOHA 特有的关节翻转；RoboSynChallenge 的
SFT checkpoint 是用 `libero_policy.EmbodiChainInputs` 训的，不做这个变换。混用**不会报错**，
只会让动作空间悄悄错位。

## EmbodiChain 需要两处本地补丁才能跑 num_envs>1

这两处改在 `~/workspace/EmbodiChain` 工作区里（**未提交** —— 那是 DexForce 的仓库），
如果重新 checkout 或升级 EmbodiChain 就会丢，PPO 会在并行 rollout 时"莫名"崩掉。
两处都值得整理成最小复现提给 DexForce。

**1. `managers/events.py` `get_pose()`：高级索引误用**

```python
# 原来:两个索引张量逐元素配对,N=1 时 [1]x[6] 碰巧广播成功,N=4 时直接 IndexError
entity.get_qpos()[env_ids, entity.get_joint_ids(control_part)]
# 修为:先取行再取列
entity.get_qpos()[env_ids][:, entity.get_joint_ids(control_part)]
```

**2. `randomization/spatial.py` + `events.py`：FK/IK 未透传 `env_ids`（部分重置崩溃）**

`randomize_robot_eef_pose` 把 qpos 切成 `len(env_ids)` 行，但调 `compute_fk` / `compute_ik`
时不传 `env_ids`，它们默认按全部环境校验 batch。**全量 reset 时两者相等，测不出来**；
episode 中途只有部分环境终止、auto-reset 只带子集时报
`Joint positions batch size mismatch. Expected 32 but got 2`。

这个 bug 的表现极具迷惑性：短测（几步内没有环境终止）永远通过，长测必崩 ——
看起来像"引擎不稳定"，实际是确定性 bug。验证方式：8 环境 × 120 步随机动作
（随机动作让环境频繁终止，部分重置持续发生），修复后全程无崩溃。

**排障心得**：dexsim 引擎崩溃经常不产生 Python traceback（C++ 层直接终止进程，
退出码可能还是 0）。所以失败日志必须保留 —— `launch/rlinf_bench_envs.sh` 会把失败档位的
完整日志存到 `/tmp/rsc_bench_failures/`，别删。

## 成功率不对劲时第一个该查的地方

mixer_operating 的按钮位置偏移量在两个版本里不一致：

| | `button_offset_x` | `_y` | `_z` |
|---|---|---|---|
| 官方 main + 上游（当前使用） | `0.11175` | `-0.006` | `0.042` |
| sim-recap 工作区未提交的标定 | `-0.1176` | `0.0` | `0.056` |

x 的符号相反，y/z 也都改了 —— 是有人重新标定过，不是笔误。当前接入按"用官方判定"走官方版，
实测能判出成功（20 集 16 次）。但如果 PPO 的成功率曲线看着可疑（比如涨得太容易、或者视频里
机器人按的位置明显不对），**先查这个偏移量**：判定用错了位置，PPO 会照着错的信号优化，
而成功率曲线本身看不出异常。

## 其他踩过的坑

- **`deploy_policy.yml` 里 `pytorch_device: cpu`**。JAX 路径不读这个字段，所以一直没被咬到；
  转成 PyTorch 后它会把 3B 模型钉死在 CPU 上推理。用转换后的 checkpoint 评测时必须显式传
  `--pytorch_device cuda`。
- **`stdbuf` 对 Python 无效**。Python 用自己的 io 缓冲，不是 libc stdio。跑评测/训练一律加
  `PYTHONUNBUFFERED=1`，否则 `print()` 全被缓冲吞掉，看起来像卡死。
- **`cmd | tail` 的退出码是 `tail` 的**。管道会掩盖真实失败，直接重定向到文件。
- **评测静默退出码 0 ≈ `reset()` 抛了异常**。异常触发 `eval_policy.py` 外层 `finally` 里的
  `env.close()`，dexsim 的 C++ 引擎直接 `exit(0)`，traceback 还没打印进程就没了。
  用最小复现（只建环境 + `reset()`，不加载策略）才能看到真因。
- **评测要传 `--headless True`**。`deploy_policy.yml` 的 `headless: false` 只适用于有显示器的
  交互式场景，官方脚本 `08_eval.sh` / `01_rollout.sh` 都强制传 True。
- **GPU 仿真是支持的**。`sim_device="cpu"` 只是 `SimulationManagerCfg` 的默认值，单环境评测路径
  没去覆盖它；EmbodiChain 自带的 push_cube RL 配置写的是 `"device": "cuda:0"` + `num_envs: 64`。

## 还没做完的

- **RLinf venv 没装**，所以端到端训练还没真正跑起来。上面的链路（配置解析、补丁、前置检查）
  都验过，但只到 `--dry-run`。
- **`total_num_envs` 是照抄 robotwin 的占位值 256，没有实测依据**。RLinf 的 EmbodiChain 例子是
  无渲染 CartPole，LIBERO 是轻量渲染 + 200 步 episode，都不可比；这里是三路 640×480 相机 +
  500 步 + 每 10 步随机光照。这个数定不下来，所有 batch size 都是猜的。需要先测带渲染时
  `num_envs` 从 1→8→16→32→64 的 FPS 和显存曲线。
- **`num_steps`（flow 积分步数）、`noise_level` 用的是 robotwin 的值**（5 / 0.3），没有针对
  RoboSynChallenge 调过。
