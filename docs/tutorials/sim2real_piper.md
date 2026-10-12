# Sim2Real：双臂 PiPER 上部署 pi0.5

本教程面向这样一套真机：两条 PiPER 机械臂，标准 70 mm 夹爪，三个 D435 相机，摆放方式与比赛的 CobotMagic 相同。流程是仿真数据加真机数据混合微调 pi0.5，再部署到真机。

整体链路：

```text
仿真数据 (V100 夹爪)  ─┐
                       ├─ 混合采样 + 夹爪换算 ─ pi0.5 微调 ─ serve_policy.py ─ websocket ─ 真机客户端 ─ pyAgxArm / D435
真机数据 (70 mm 夹爪) ─┘
```

## 1. 约定

策略的状态和动作都是 14 维，顺序与仿真一致：

| 下标 | 含义 | 单位 |
|---|---|---|
| 0–5 | 左臂关节 1–6 | 弧度，绝对角度 |
| 6 | 左夹爪 | 归一化开口，0 闭合，1 全开 |
| 7–12 | 右臂关节 1–6 | 弧度，绝对角度 |
| 13 | 右夹爪 | 归一化开口，0 闭合，1 全开 |

控制频率是 25 Hz。

夹爪的“1”在不同地方代表的物理开口不同，这是最容易出错的地方：

| 来源 | 夹爪值 1.0 对应的开口 |
|---|---|
| 仿真 CobotMagic V100 夹爪 | 100 mm |
| 你的 PiPER 70 mm 夹爪 | 70 mm |
| `pi05_robosynchallenge_sim2real` 训练和部署 | 70 mm |

混合训练时，仿真数据的夹爪值乘以 100/70 再截断到 1，这样同一个数值在两边代表同样的物理开口。仿真里夹住 35 mm 的物体是 0.35，换算后是 0.5，真机上 0.5 正好也是 35 mm。

## 2. 检查真机和仿真是否对齐

**零位。** 仿真里的零位是 PiPER 标准的折叠收拢姿态，和真机零位一致，所以关节零点偏移默认都是 0。

**关节方向。** 零位一致不能证明每个关节的正方向一致。用抓拍加叠图的方法一次检查方向、底座位置和相机外参：

1. 在真机上抓拍几个姿态。机械臂只读不动，用主臂或手把从臂摆到不同姿态，每个姿态按一次回车。

   ```bash
   cd policy/pi05
   python -m examples.cobotmagic_piper_real.capture_snapshot \
       --cam-high-serial <顶部相机序列号> \
       --cam-left-wrist-serial <左腕相机序列号> \
       --cam-right-wrist-serial <右腕相机序列号> \
       --out-dir snapshots
   ```

   姿态建议覆盖每个关节都明显偏离零位的情况，比如关节 2 抬起 0.5 rad、关节 5 转 0.5 rad。

2. 在装有 EmbodiChain 的机器上，把仿真摆到同样的关节角并叠图：

   ```bash
   python scripts/sim2real/overlay_real_sim.py \
       --gym_config configs/click_bell/clear/gym_config.json \
       --action_config configs/click_bell/action_config.json \
       --snapshots policy/pi05/snapshots/*.npz \
       --out_dir sim2real_overlay
   ```

   每张输出图从左到右是真机、仿真、两者半透明叠加。

3. 看叠加图判断问题：

   | 现象 | 原因 | 处理 |
   |---|---|---|
   | 机械臂完全重合 | 已对齐 | 不用改 |
   | 某个关节弯向相反方向 | 关节方向相反 | 在部署参数里把该关节的 sign 改成 -1 |
   | 整幅画面平移或旋转 | 顶部相机外参不准 | 重新标定，用 `scripts/camera_extrinsics_to_lootat.py` 写回配置 |
   | 手腕视角偏了 | 手腕相机安装位置不同 | 改配置里 `cam_left_wrist`/`cam_right_wrist` 的 `pos`/`quat` |

   改相机配置时要改你训练用的那个 gym 配置，并重新生成仿真数据。

## 3. 检查数据

混合之前先确认两份数据的单位一致：

```bash
python scripts/sim2real/inspect_lerobot_dataset.py \
    lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell \
    lerobot_dataset/piper_real/click_bell
```

脚本会打印帧率、图像字段，以及状态和动作每一维的分布，并对下面几种情况报警：

- 夹爪不是 0 到 1 的归一化值，比如按米或毫米存储。
- 关节角超过弧度的合理范围，通常是存成了角度。
- 两份数据的帧率不同。

真机数据要满足和仿真一样的键名：`observation.qpos`、`action`、`cam_high.color`、`cam_left_wrist.color`、`cam_right_wrist.color`。夹爪需要按 70 mm 归一化。如果你的真机数据是别的格式，先转换再混合。

比赛官方的 `cobotmagic_Real_*` 真机数据也可以混进来，但要先确认它录制时用的夹爪。如果和仿真一样是 100 mm 夹爪，给它设 `gripper_scale=100/70`。如果是 70 mm 夹爪，设 `1.0`。检查脚本只能看出数值是否归一化，看不出归一化用的是哪种夹爪。

## 4. 混合微调

配置在 `policy/pi05/src/openpi/training/config.py` 的 `pi05_robosynchallenge_sim2real`。按你的数据修改：

```python
mixture_sources=(
    _mixture.DatasetSource(
        repo_id="RoboSynChallenge/cobotmagic_Sim_click_bell",
        weight=0.5,
        gripper_scale=_mixture.SIM_V100_TO_PIPER_70MM,  # 100 / 70
    ),
    _mixture.DatasetSource(
        repo_id="piper_real/click_bell",  # 你的真机数据
        weight=0.5,
        gripper_scale=1.0,
    ),
),
```

`weight` 是每个批次里各数据源的样本比例。真机数据通常少很多，会被自动重复采样，不会被仿真数据淹没。仿真数据的每一帧也都会被用到。真机数据很少时可以从 0.5 起步，再试 0.3 和 0.7 对比。

数据放到 `policy/pi05/training_data/` 下，目录名对应 `repo_id`，然后：

```bash
cd policy/pi05
export HF_LEROBOT_HOME=$PWD/training_data
uv run scripts/compute_norm_stats.py --config-name pi05_robosynchallenge_sim2real
bash finetune.sh pi05_robosynchallenge_sim2real my_run 0
```

规范化统计会按混合后的比例计算，和训练时看到的分布一致。所有数据源需要相同的帧率，否则一个动作块在不同数据源里对应的时长不同，训练日志会给出警告。

## 5. 真机部署

**安装客户端依赖**，在机器人电脑的 `policy/pi05` 目录下：

```bash
pip install -r examples/cobotmagic_piper_real/requirements.txt
pip install -e packages/openpi-client
```

**激活 CAN。** 只需要两条从臂的 CAN。按 pyAgxArm 的 [CAN 手册](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/can_user.md) 用 `can_muti_activate.sh` 把它们命名为 `can_left` 和 `can_right`，波特率 1000000。

**确认固件版本**，并选对 `--firmware`：

| 固件 | 参数 |
|---|---|
| S-V1.8-2 及更早 | `default` |
| S-V1.8-3 到 S-V1.8-7 | `v183` |
| S-V1.8-8 | `v188` |
| S-V1.8-9 及以后 | `v189` |

配错时客户端会在启动日志里提示正确的值。

**启动策略服务端**，可以在 GPU 服务器上：

```bash
cd policy/pi05
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_robosynchallenge_sim2real \
    --policy.dir=checkpoints/pi05_robosynchallenge_sim2real/my_run/20000
```

**先空跑。** `--dry-run` 会读相机和关节、请求策略、打印动作，但不驱动机械臂：

```bash
python -m examples.cobotmagic_piper_real.main \
    --prompt "Click the bell" \
    --host <服务端IP> \
    --cam-high-serial <序列号> --cam-left-wrist-serial <序列号> --cam-right-wrist-serial <序列号> \
    --firmware v188 \
    --dry-run
```

看日志里每个动作块的第一帧，关节角应该和当前状态接近，夹爪值在 0 到 1 之间。

**正式运行**时去掉 `--dry-run`。如果第 2 节发现某个关节方向相反，加上对应参数，例如：

```bash
--left-joint-signs 1 1 1 1 -1 1 --right-joint-signs 1 1 1 1 -1 1
```

客户端的默认行为：

- 每个回合开始时用 3 秒平滑移动到零位，夹爪闭合，与仿真起始状态一致。用 `--reset-pose current` 可以从当前姿态直接开始。
- 每个动作块执行 10 步后重新推理，与仓库里仿真评测的设置相同，用 `--action-horizon` 调整。
- 每个控制周期每个关节最多变化 0.1 rad，即 2.5 rad/s。策略请求更大的跳变时会被限幅并打印警告，频繁出现通常说明状态或单位没对齐。
- 关节目标会被限制在 PiPER 关节限位内。
- 退出时机械臂保持使能。PiPER 失能后抬起的关节会直接掉落，所以要先手动把机械臂放回安全姿态再断电。

**安全提示。** 首次运行时降低速度，比如 `--speed-percent 30`，并让人守在急停旁边。

## 6. 常见问题

**夹爪总是开不到位或夹不紧。** 检查训练配置里每个数据源的 `gripper_scale`，以及部署时的 `--gripper-full-mm`。用 `pi05_robosynchallenge_sim2real` 训练的模型配 70。只用未换算的仿真数据训练的模型配 100，超过 70 mm 的指令会被硬件上限截断。

**动作明显滞后或抖动。** 服务端推理时客户端会停住等待。可以加大 `--action-horizon` 减少推理次数，或者把服务端放到离机器人更近的机器上。

**仿真评测混合训练的模型。** 模型输出的夹爪是 70 mm 归一化，仿真里需要乘以 0.7 才是同样的物理开口。仓库现有的仿真评测脚本不做这个换算。
