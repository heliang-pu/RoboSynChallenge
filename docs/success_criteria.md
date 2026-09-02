# RoboSynChallenge 任务成功判据（对齐官方 origin/main）

对齐核验日期：2026-08-27
基准：`origin/main` = `https://github.com/EDEM-AI/RoboSynChallenge.git`，HEAD `6f555fc`（Merge PR #41 feat/add-smolvla-policy）
本地分支：`feat/rtc-async-pi05`

## 0. 结论：判据与官方完全一致

```
git diff --stat origin/main -- robosynchallenge/tasks/     # 空
git status --porcelain -uall -- robosynchallenge/tasks/    # 空
```

`robosynchallenge/tasks/` 下 10 个任务的 `is_task_success` / `_evaluate_task_state` / `compute_task_state`
与官方 `origin/main` **逐字节相同**，工作区也无未提交改动、无未跟踪覆盖文件，因此不需要任何回退。

另外核验的几处可能影响判定口径的地方：

| 位置 | 状态 |
| --- | --- |
| `scripts/eval_policy_parallel.py` 评测主循环的成功判定 | 与官方一致：`if not is_truncated and env.get_wrapper_attr("is_task_success")(): episode_success = True`（本地 [eval_policy.py:812](scripts/eval_policy_parallel.py#L812)，官方同一行逻辑）。本地只新增了 `--rollout_save` 侧录与 sidecar 写出，不改判定 |
| `configs/*/clear/gym_config.json` | 工作区/暂存区已恢复成官方原样（关节名大写 `RIGHT_JOINT[1-6]` 等），`git diff origin/main -- configs/` 对官方配置零改动，只有新增的自制 coverage/syn 配置目录 |
| `scripts/run_env_seeded.py` | 新增 `--report_task_success` / `--save_only_success` / `--success_settle_steps`，都只是**调用**官方判据，未改判据本身；默认关闭时行为与官方一致 |
| 判定基类 `EmbodiedEnv` | 来自外部依赖 `embodichain` 包，本地未 vendor、未 patch；`third_party/` 只新增了 `evo_rl` |

> 历史：2026-08-25/26 已把各机器上自改的判定（尤其 sample_loading 的几何插入版）全部 `git checkout` 回官方版，
> 旧版备份见 [docs/notes/sample_loading_geometric_evaluator_backup_20260826.py](notes/sample_loading_geometric_evaluator_backup_20260826.py)。
> **重要口径提醒**：用几何版判定打过的标签（recap round1 rollout 标签、held-out 40 集、20-seed 诊断评估）与官方口径不可直接对表。

## 1. 判定调用链

- 评测口径只认 **`is_task_success()`**：`eval_policy.py` 每个 env step 后调用一次，返回 True 且 **未 truncated** 才算这一集成功；到 `max_episode_steps` 仍未 True 则判超时失败。
- `compute_task_state()` 返回的 `(success, fail, metrics)` 用于环境自身的 terminate/truncate 与调试，**多数任务里的 success 位被刻意置 0**，不要拿它当成功率。
- 每个任务注册了三个变体：
  - `XxxEnv`（如 `SampleLoading`）—— **真判据**，评测走这个；
  - `XxxTestEnv`（`SampleLoadingTest` 等）—— `is_task_success` 恒返回 `True`（专供人工目视/采集，**绝不能拿来算成功率**）；
  - `XxxAgentEnv` —— 继承真判据，加 agent 接口。
- `max_episode_steps`：`DrawerOpenPlace` 900，其余均 600；`TableRearrangement` 基类未显式设置（用注册默认值），其 Test 变体为 600。

## 2. 十个任务的判据

判定统一在 arena 局部坐标系下，用 `get_local_pose(to_matrix=True)` 的 4×4 位姿；
「倒下」类检查统一形如：取物体旋转矩阵某一列与世界 Z 轴夹角 ≥ 阈值。

### 2.1 click_bell（按铃）— [click_bell.py:131-147](robosynchallenge/tasks/click_bell/click_bell.py#L131-L147)

- 读 `button` articulation 的单个 prismatic 关节：`press_depth = -qpos[:, 0]`（关节行程 `[-0.005, 0]`）。
- **压深 ≥ 0.0048 m** 即置位。
- **latched（一次触发永久成立）**：`self._button_pressed |= success`，`is_task_success()` 直接返回 `_button_pressed`；`reset()` 清零。
- 注意 `compute_task_state` 里返回的 success 被强制置 False（第 143 行），只有 `is_task_success` 是真判据。
- `ClickBellTest` 阈值是 0.004，但它的 `is_task_success` 恒 True，与评测无关。

### 2.2 drawer_open_place（开抽屉放鸭子）— [drawer_open_place.py:141-165](robosynchallenge/tasks/drawer_open_place/drawer_open_place.py#L141-L165)

- `duck` 与 `drawer` 的 `outer_box` link 在 **XY 平面的距离 ≤ 0.10 m**。
- 仅此一条，无稳定帧、无姿态、无速度要求；瞬时成立即成功（评测循环一旦读到 True 就 break）。

### 2.3 handle_basket（提篮左移）— [handle_basket.py:201-305](robosynchallenge/tasks/handle_basket/handle_basket.py#L201-L305)

官方 2026-08-21 `0129f36` 重写版，四条同时成立并**连续保持 75 个 env step**：

| 条件 | 阈值 |
| --- | --- |
| 牛奶在篮内 `in_basket` | `‖milk_xy − basket_xy‖ < 0.10` 且 `milk_z > basket_z` |
| 篮被提起 `picked` | `basket_z − orig_basket_z > 0.01` |
| 向左位移 `moved_left` | `y − orig_y > 0.15`（**Y 轴**；旧版判 X 轴 0.05 的是自改版，已废弃），且 `picked or in_basket` |
| 稳定 | `moved_left & in_basket` 连续累计 ≥ **75 步** |

- 基线 `orig_basket_x/z` 在 `_hb_diag_step >= 1` 时采样，`orig_basket_y` 首次调用即采样。
- 稳定计数按 `self._elapsed_steps` 的真实增量累加（不再用调用次数当时钟），中断即清零。
- 调试：`ROBOSYN_DEBUG_SUCCESS_FLAGS=1` 打印 `_hb_debug_flags`。
- 相关配置：官方把 basket / milk 的 `mass` 都降到 0.025。

### 2.4 item_assembly（硅胶管对接）— [item_assembly.py:180-317](robosynchallenge/tasks/item_assembly/item_assembly.py#L180-L317)

五项 **与** 关系：

1. `angle_ok`：两根 guijiao 的局部 X 轴夹角（取 `acos|dot|`，反平行视同平行）**≤ 15°**；
2. `valid_pose_mask`：两个位姿矩阵全有限；
3. `step_mask`：`_elapsed_steps >= success_min_steps`（默认 **5**），防 t=0 假成功；
4. `contact_ok`：接触传感器 `guijiao_contact` 中，**仅 guijiao1↔guijiao2 之间**的有效接触最小距离 **≤ 0.003 m**（拿不到传感器/对象时退化为恒 True）；
5. `lateral_ok`：两轴的**径向偏移** `‖(c2−c1) − ((c2−c1)·â1)â1‖ ≤ success_lateral_tol`（默认 **0.02 m**）。

无稳定帧要求。该任务没有 Test 变体，只有基类与 Agent。

### 2.5 items_handover（笔递交入笔筒）— [items_handover.py:126-198](robosynchallenge/tasks/items_handover/items_handover.py#L126-L198)

`_evaluate_task_state`（基础层）：

- `pen_near_holder`：pen 与 holder **XY 距离 ≤ 0.03 m**；
- `~pen_ret`：pen 的局部 X 轴与世界 Z 夹角 **< 1.309 rad(≈75°)**（未倒）；
- `~holder_ret`：holder 的局部 Y 轴与世界 Z 夹角 **< 1.309 rad**（笔筒未倒）。

`is_task_success` 在此之上**再加 AABB 竖直重叠**：`min(pen_maxZ, holder_maxZ) − max(pen_minZ, holder_minZ) > 0.08 m`
（即笔确实插进筒里而不是搭在筒口）。拿不到 AABB 时回退到基础层判定。

### 2.6 manipulate_pipette（移液器）— [manipulate_pipette.py:137-184](robosynchallenge/tasks/manipulate_pipette/manipulate_pipette.py#L137-L184)

- **按压计数**：pipette 滑动关节 `qpos[:,0] <= 0.66 * slide_min + tolerance` 视为「到底」；由未到底→到底的上升沿计一次，累计计数 `_pipette_min_reach_count`。判定要求 **≥ 1 次**（变量名叫 `pipette_pressed_twice`，实际阈值是 1）。
- **未失败**：`beaker1` 的局部 Z 轴与世界 Z 夹角 **< π/9 (20°)**，且 pipette 的局部 X 轴与世界 Z 夹角 **< π/9**。
- 成功 = `(~failed) & (按压计数 ≥ 1)`；计数在 `_initialize_episode` 清零，**本集内 latched**。

### 2.7 mixer_operating（搅拌器）— [mixer_operating.py:258-292](robosynchallenge/tasks/mixer_operating/mixer_operating.py#L258-L292)

三项 **与** 关系：

1. `~beaker_ret`：beaker 局部 Z 轴与世界 Z 夹角 **< π/3 (60°)**；
2. `beaker_near_mixer`：beaker 与 `beaker_mixer` **XY 距离 ≤ 0.08 m**；
3. `_button_contact_happened`（**latched**）：接触报告中同时满足
   - 一侧是 mixer 的 user_id、另一侧是机械臂 link 的 user_id，
   - 接触点落在按钮区域内（到 `_get_button_position` 的距离 ≤ `_button_region_radius`），
   - **冲量 ≥ 0.01**。
   
   一旦发生即置位，`reset()` 清零。
- `compute_task_state` 的 success 位恒 0，真判据只在 `is_task_success`。

### 2.8 sample_loading（试管入架）— [sample_loading.py:127-314](robosynchallenge/tasks/sample_loading/sample_loading.py#L127-L314)

官方版判据（自 2026-08-04 `4d1d773` 未变）。阈值均可用 env 属性覆盖，默认值：

| 名称 | env 属性 | 默认 |
| --- | --- | --- |
| XY 距离 | `success_pos_thresh` | 0.035 m |
| Z 上限（相对 rack） | `success_z_thresh` | 0.07 m |
| 线速度 | `success_vel_thresh` | 0.05 m/s |
| 脱手距离 | `success_eef_release_dist` | 0.04 m |
| 稳定帧 | `success_stable_steps` | **8** |
| 竖直角 | `success_vertical_angle_thresh` | 0.087266 rad (≈5°) |
| 底面对齐 | `success_bottom_align_thresh` | 0.005 m |

单帧 `placement_ok` 是三条支路的**并集**：

- 主支路：`not_fallen & pos_ok & z_ok & vel_ok & not_held`
  - `not_fallen`：cube 和 rack 的局部 Z 轴与世界 Z 夹角均 **< 0.1745 rad (10°)**；
  - `pos_ok`：`‖cube_xy − rack_xy‖ < 0.035`；
  - `z_ok`：`cube_z <= rack_z + 0.07`；
  - `vel_ok`：cube 线速度模 `< 0.05`（取不到时视为通过）；
  - `not_held`：cube 到左右 EEF（FK 求得）距离**均 > 0.04**；FK 不可用时退化为「速度 < 0.05」。
- 竖直支路 `alt_condition`：`vertical_ok & pos_ok & vel_ok & (z_ok | bottom_ok)`；
- 底面支路 `bottom_condition`：`not_fallen & pos_ok & vel_ok & not_held & bottom_ok`
  （`bottom_ok`：cube 与 rack 的世界系最低顶点 z 之差 `< 0.005`）。

**成功 = `placement_ok` 连续成立 ≥ 8 帧**（`_place_stable_count`，中断清零）。
返回的 metrics 含 `cube_xy_dist / place_stable_count / cube_vertical_angle / cube_lin_vel_norm / bottom_z_diff` 等，便于排查。

> 采集侧注意：专家脚本在松手瞬间就结束，稳定计数来不及攒到 8。`scripts/run_env_seeded.py --success_settle_steps N` 就是为此加的
> 保持位姿空步（默认 0 = 官方行为），仅影响采集，不影响评测口径。

### 2.9 table_rearrangement（摆餐具）— [table_rearrangement.py:155-190](robosynchallenge/tasks/table_rearrangement/table_rearrangement.py#L155-L190)

- 目标位由 plate 位姿动态给出：`spoon_target_y = plate_y − 0.16`，`fork_target_y = plate_y + 0.16`。
- `y_ok`：`|spoon_y − spoon_target_y| ≤ tolerance` 且 `|fork_y − fork_target_y| ≤ tolerance`；
- `z_ok`：`spoon_z − plate_z ≤ height_tolerance` 且 `fork_z − plate_z ≤ height_tolerance`（不许悬空）。
- `tolerance` 默认 **0.02**、`height_tolerance` 默认 **0.05**，可由 `metadata["success_params"]` 覆盖。
- 只看 Y 与相对高度，**不看 X、不看姿态**。

### 2.10 water_pouring（倒水）— [water_pouring.py:143-190](robosynchallenge/tasks/water_pouring/water_pouring.py#L143-L190)

一个「抓起 → 倾倒 → 扶正」的时序判据，靠三个 latch 位实现：

- `held`（当前帧持握）：bottle 到 `right_link6` 距离 `< 0.30`，且（`bottle_z > 初始 z + 0.03` 或 `_grasp_started` 已置位）；
- `relative_pose_valid`：瓶口 `bottle_mouth = bottle_pos + 0.236 · bottle_Y轴`
  - 瓶口与 cup 的 XY 距离 `< 0.08`，
  - `cup_z + 0.04 < mouth_z < cup_z + 0.30`，
  - `bottle_axis_xy · mouth_to_cup_xy > −0.02`（倾倒方向大致朝杯子）；
- `pouring_now`：`held & (π/4 < bottle_angle < 2π/3) & relative_pose_valid & ~cup_fall`，其中 `bottle_angle = acos(bottle_Y轴的 z 分量)`；
- `cup_fall`：cup 局部 Z 轴与世界 Z 夹角 ≥ **π/4 (45°)**。

latch：`_grasp_started |= held`；`_grasp_lost |= _grasp_started & ~held`；`_pouring_started |= pouring_now`。

**成功 = `_pouring_started & returned_upright & ~_grasp_lost & ~cup_fall`**，
其中 `returned_upright = held & (bottle_angle < π/4)`——即倒完必须**还握着瓶子并把瓶身扶正**，中途脱手（`_grasp_lost`）直接判负。

## 3. 使用与复核

```bash
# 复核判据是否仍与官方一致（应无输出）
git fetch origin
git diff --stat origin/main -- robosynchallenge/tasks/
git status --porcelain -uall -- robosynchallenge/tasks/

# 采集时打印官方判据结果与 metrics
python scripts/run_env_seeded.py ... --report_task_success

# handle_basket 判定细节
ROBOSYN_DEBUG_SUCCESS_FLAGS=1 python scripts/eval_policy_parallel.py ...
```

要改判据请到上游提 PR，不要在本地分叉——本地自改判定与官方评测口径对不上，会污染所有基于成功标签的下游产物
（RECAP 优势值、价值函数训练集、成功率对表）。
