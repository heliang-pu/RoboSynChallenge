# `random` 评测配置的物体生成范围修正（可达性）

分支：`fix/random-spawn-reachability`（从 `main` `8fc3081` 切出，worktree `../RoboSynChallenge-spawn-fix`）。
分析日期：2026-09-02。改动只涉及 `configs/<task>/random/gym_config.json` 与 `configs/<task>/random/gym_config.json`
里物体 pose 随机化事件的 `position_range`；旋转范围、`clear`/`coverage_*`/`aug_*` 配置、`robosynchallenge/tasks/` 均未动。

## 1. 问题

官方 `random` 配置里物体的 `position_range` 是轴对齐矩形，没有对照机械臂工作空间。CobotMagic 两条 Piper 臂的基座在
机器人坐标系 (0.233, ±0.300)（`EmbodiChain/embodichain/lab/sim/robots/cobotmagic.py`），连杆 0.285 + 0.251 m，
腕部到 TCP 0.091 + 0.143 m。以 click_bell 为例：铃铛范围 x∈[0.40, 0.85]、y∈[-0.30, 0.30]，最远角 (0.85, 0.30)
离右臂基座 0.86 m，右臂无论如何够不到。专家在这些位置直接 IK 失败——采集时被静默重试掉，评测时则成为必败集。
评测（`scripts/eval_policy.py --setting random`）与采集用的是同一份配置。

## 2. 方法

判据 = **专家动作图能否生成**（`scripts/analyze_rigid_spawn_range.py`），会走完 `action_config.json` 里全部
`get_ik_ret` 校验（抓取/预抓取/抬起/放置/交接每个关键位姿的 IK、关节限位、`is_qpos_flip`），比「离基座多远」严格。
用官方 `random` 配置逐物体扫网格（间距约 3 cm，每点 3 次试验，其余物体按各自事件照常随机，机器人初始关节/末端扰动照常），
点分三类：`#` 三次全可行、`+` 部分可行（多半是伙伴物体随机到不可行处或旋转不利）、`.` 三次全失败。
新范围 = 不含任何 `.` 点、尽量多保留 `#` 点的最大轴对齐矩形，边界取到实测网格点本身（不外扩），只收不放。

扫描在本机 4090 上只跑了一部分就按要求停掉了（不再占卡）：click_bell 314/336 点、duck / pipette / beaker1 / pen /
holder / cube 全部点、cup 45/54、fork 24/54、spoon 18/54、bottle 9/36；basket、milk（见 §4）、drawer、fork/spoon 的远端、
mixer 两物体、rack、guijiao 没有扫。没扫到的物体用两类旁证判断：官方成功专家数据的位置分布
（`scripts/analyze_random_coverage.py` 2026-08-27 的审计，覆盖 8 个任务共 7000+ 集）与上面测出的臂可达半径。
**改后配置没有再跑仿真验证**（用户要求不占卡），见 §5。

工具修复：`analyze_rigid_spawn_range.py` 原来自己拼 `SimulationManagerCfg`，与现版 EmbodiChain 启动器脱节
（`AttributeError: renderer`、`KeyError: enable_rt`），已改为与 `run_env.py` 同一条 `config_to_cfg` 路径，
并在 env 缺 `action_config` 属性时补挂（见 §4 handle_basket）。

## 3. 逐任务结果与改动

坐标为世界系 (x 前, y 左)，单位 m。行 = y 从高到低，列 = x 从低到高。

### click_bell — `button`（右臂俯压）

原 x[0.40,0.85] y[-0.30,0.30]，314 点中 112 点全失败（x=0.85 整列另测过，全失败），失败节点全是右臂 IK
（`button_prepress_pre_qpos`）。可行域是以右臂基座 (0.233,-0.30) 为圆心、半径约 0.57 m 的扇形：

```
x:      .40 .43 .46 .49 .52 .55 .58 .61 .64 .67 .70 .73 .76 .79 .82
y=+0.30  .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
y=+0.24  #   #   .   .   .   .   .   .   .   .   .   .   .   .   .
y=+0.18  #   #   #   #   #   #   .   .   .   .   .   .   .   .   .
y=+0.12  #   #   #   #   #   #   #   #   .   .   .   .   .   .   .
y=+0.06  #   #   #   #   #   #   #   #   #   #   .   .   .   .   .
y=+0.03  #   #   #   #   #   #   #   #   #   #   #   .   .   .   .
y= 0.00  #   #   #   #   #   #   #   #   #   #   #   .   .   .   .
y=-0.06  #   #   #   #   #   #   #   #   #   #   #   #   .   .   .
y=-0.12  #   #   #   #   #   #   #   #   #   #   #   #   #   .   .
y=-0.15  #   #   #   #   #   #   #   #   #   #   #   #   #   #   .
y=-0.30  #   #   #   #   #   #   #   #   #   #   #   #   #   #   .
```

面积最大的全可行矩形是 x[0.40,0.70] y[-0.30,0.03]（保留 132/202 个可行点），远角再留 1 cm 余量：
**新范围 x[0.40,0.69] y[-0.30,0.03]**。

### drawer_open_place — `duck`（左臂抓取）

原 x[0.53,0.80] y[0.18,0.35]。x=0.53 整列与 x=0.56、y≥0.24 全失败（`left_arm_duck_pregrasp_qpos`：离左臂基座
不到 0.36 m，俯抓的预抓取位姿超关节限位），x≥0.59 全部可行（58/70 点全可行）。**新范围 x[0.59,0.80] y[0.18,0.35]**。
`drawer` 组随机（x 偏移 [-0.01,0.05]、y 偏移 [-0.18,0]）在 duck 扫描的 210 次试验里从未导致失败，不改。

### manipulate_pipette — `pipette`（右臂抓取）、`beaker1`

pipette 原 x[0.45,0.72] y[-0.28,-0.10]：x=0.72 整列全失败，x=0.69 在 y≥-0.16 失败；x≤0.66 除个别 `+` 外全可行。
**新范围 x[0.45,0.66] y[-0.28,-0.10]**。beaker1 63 点无一全失败（20 个 `+` 来自 pipette 随机），不改。

### items_handover — `pen`（右臂抓取后交接）、`holder`（左臂放置）

pen 原 x[0.52,0.675] y[-0.30,0.00]：全失败点集中在左上角（x≥0.61 且 y≥-0.06，x=0.675 且 y≥-0.15），
失败节点 `right_arm_pen_grasp_qpos`。不含失败点且保留最多可行点的矩形：**新范围 x[0.52,0.64] y[-0.30,-0.09]**（保留 23/25 个 `#`）。
holder 原 x[0.50,0.70] y[0.00,0.25]：全失败点在 x=0.50 列与 y≥0.22 的近侧（离左臂基座 <0.3 m，放笔的俯压位姿够不到）。
**新范围 x[0.53,0.70] y[0.00,0.18]**。这两张图里的 `+` 大多来自伙伴物体随机（pen 的失败 29 次是 holder 引起、
holder 的失败 35 次是 pen 引起）。

### water_pouring — `cup`（左臂侧抓）、`bottle`

cup 原 x[0.56,0.70] y[0.05,0.30]。侧抓要求杯子离左臂基座 ≥ 约 0.43 m（实测 (0.644,0.144)=0.44 m 可行、
(0.616,0.144)=0.41 m 不可行、(0.644,0.30)=0.41 m 不可行、(0.672,0.30)=0.44 m 可行）：

```
x:      .560 .588 .616 .644 .672
y=+0.30   .    .    .    .    #
y=+0.21   .    .    .    .    #
y=+0.14   .    .    .    #    #
y=+0.08   .    .    #    #    #
y=+0.05   .    +    #    #    #
```

x=0.672 整列全可行；官方成功专家数据的 cup_x 分布 0.592–0.755、均值 0.693，也集中在远端。
**新范围 x[0.675,0.70] y[0.05,0.30]**（保留整个 y 区间；官方范围的 x 上限 0.70 本身就切掉了最可行的 0.70–0.75 段，
本次只收不放）。bottle 扫到的 9 点全部至少一次可行（失败均由 cup 不可行连带），不改。

### sample_loading — `rack`（左臂放管）、`cube`

cube 扫描 99 点仅 1 点全可行、68 点全失败，但失败节点集中在 `left_arm_cube_place_qpos`（放进 rack）与
`right_arm_cube_up2_qpos`/`left_arm_takeover_*`（交接点由 cube 位置和朝向 rack 的偏航共同决定），是 rack 随机造成的，
cube 自身范围保持不变（官方成功数据 cube_x 0.451–0.68、cube_y -0.277–-0.002 均匀覆盖整个范围）。
rack 没来得及扫，用 2026-08-27 覆盖审计的证据：官方 1000 集成功数据里 rack_y 五个等宽 bin 的计数是 [552, 324, 111, 8, 1]
（y>0.09 几乎没有成功），rack_x 五个 bin 的严格保留计数是 [0, 27, 119, 228, 379]（x<0.66 几乎没有成功）；
当时把 rack 放到 x 0.686–0.70、y 0.06–0.09 做 20 次 reset 有 13 次可规划，原始配置只有 7 次。
**新范围 x[0.67,0.70] y[0.00,0.08]**（原 x[0.63,0.70] y[0.00,0.15]）。

### handle_basket — `milk`（左臂侧抓）、`basket`

**上游代码在这个任务上跑不了专家生成**：`HandleBasketEnv.__init__` 不保存 `kwargs["action_config"]`（其余 9 个任务都保存），
`create_demo_action_list` 还调用了类里不存在的 `_sync_carry_basket_runtime_attrs()`，`origin/main` 同样如此。
评测不走这条路径所以不受影响，但采集/分析都会在第一步抛 AttributeError；`tasks/` 不能动，这里没有修。
因此 milk/basket 只能按几何推：milk 的抓取位姿与 cup 同构（工具 z 沿物体 x、俯仰 -20° 的侧抓），套用 cup 实测的
≥0.43 m 门槛，原范围近侧角 (0.60,0.30) 离左臂基座只有 0.37 m、(0.65,0.20) 0.43 m，判为够不着；
**新范围 x[0.67,0.75] y[0.05,0.30]**（原 x[0.60,0.75]，这是推断值，不是实测）。basket（右臂抓提手，离右臂基座 0.40–0.51 m；
milk 放入点 = basket 位置 +0.18 y，离左臂基座 0.43–0.54 m）不改。

### 不改的任务

- table_rearrangement：fork / spoon 扫到的 42 点无一全失败；整个范围离对应臂基座 0.17–0.44 m，plate 固定在 (0.5, 0)。
- mixer_operating：官方成功数据 beaker、beaker_mixer 的位置在各自范围内均匀分布（均值都落在范围中心），远角 0.58 m 仍在可达半径内。
- item_assembly：两物体只随机 ±1 cm。

### 改动一览

| 任务 | 物体/事件 | 原 `position_range`（x,y） | 新 `position_range`（x,y） | 依据 |
|---|---|---|---|---|
| click_bell | button `init_button_pose` | [0.40,0.85]×[-0.30,0.30] | [0.40,0.69]×[-0.30,0.03] | 网格实测 |
| drawer_open_place | duck `random_duck_pose` | [0.53,0.80]×[0.18,0.35] | [0.59,0.80]×[0.18,0.35] | 网格实测 |
| manipulate_pipette | pipette `random_pipette_pose` | [0.45,0.72]×[-0.28,-0.10] | [0.45,0.66]×[-0.28,-0.10] | 网格实测 |
| items_handover | pen `random_pen_pose` | [0.52,0.675]×[-0.30,0.00] | [0.52,0.64]×[-0.30,-0.09] | 网格实测 |
| items_handover | holder `random_holder_pose` | [0.50,0.70]×[0.00,0.25] | [0.53,0.70]×[0.00,0.18] | 网格实测 |
| water_pouring | cup `init_cup_pose` | [0.56,0.70]×[0.05,0.30] | [0.675,0.70]×[0.05,0.30] | 网格实测（45/54 点）+ 数据分布 |
| sample_loading | rack `random_rack_pose` | [0.63,0.70]×[0.00,0.15] | [0.67,0.70]×[0.00,0.08] | 覆盖审计数据 |
| handle_basket | milk `init_milk_pose` | [0.60,0.75]×[0.05,0.30] | [0.67,0.75]×[0.05,0.30] | 几何推断 |

z 分量一律不变。`random`（random 已删除） 与 `random` 同步改。

## 4. 已知问题

- `HandleBasketEnv`（上游）专家生成路径坏了：缺 `action_config` 属性、缺 `_sync_carry_basket_runtime_attrs` 方法。
  要在这台机器上采 handle_basket 数据，得先等上游修或在本地打补丁（会让 `tasks/` 偏离官方，需单独决定）。
- 可达域本质上是以臂基座为圆心的扇形/环形，轴对齐矩形只能取子集：click_bell 只保留了 65% 的可行网格点，
  cup 只剩 2.5 cm 宽。要保留完整可行域，得给随机化事件加「距臂基座距离」约束（新 functor），本次按要求只改配置。

## 5. 还没做的验证

改后配置应跑两道验证再用于正式评测（都要占卡，本次未跑）：

```bash
# 1) 专家可规划率（每物体随机抽 60 个位置，其余物体按新范围随机）
../RoboSynChallenge/.venv/bin/python scripts/analyze_rigid_spawn_range.py \
    --gym_config configs/click_bell/random/gym_config.json --action_config configs/click_bell/action_config.json \
    --event init_button_pose --sample-mode random --samples 60 --no-plot
# 2) 完整专家 rollout 的官方判定成功率
python -m scripts.run_env --gym_config configs/click_bell/random/gym_config.json \
    --action_config configs/click_bell/action_config.json --num_envs 1 --headless \
    --filter_dataset_saving --filter_visual_rand --report_task_success --max_episodes 20 --seed 20260902
```

改了 `random` 就等于改了评测分布：本分支下的成功率与官方口径（`origin/main` 的 `random`）不可直接对表，报告里要写明。
