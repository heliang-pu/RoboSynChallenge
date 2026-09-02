# CLAUDE.md

本 worktree = 分支 `fix/random-spawn-reachability`，从 `main`（`8fc3081`）切出。
只做一件事：**修正 10 个评测任务 `random` / `random_3p` 配置里物体的随机生成范围**，把机械臂够不到、
专家 IK 必定失败的区域去掉。仓库总说明见 `main` 的 CLAUDE.md，这里只写本分支特有的东西。

## 本分支改了什么

- `configs/<task>/random/gym_config.json` 与 `configs/<task>/random_3p/gym_config.json`：8 个物体事件的 `position_range`
  （click_bell button、drawer duck、pipette、items_handover pen/holder、water_pouring cup、sample_loading rack、handle_basket milk）。
  逐项依据见 `docs/random_spawn_reachability.md`；旋转范围、`clear` / `coverage_*` / `aug_*` 配置没动。
- `scripts/analyze_rigid_spawn_range.py`：跟上 EmbodiChain 启动器的参数变化（`--renderer`、`max_episodes`，`make_env`
  改为与 `run_env.py` 同一条 `config_to_cfg` 路径），否则一启动就 `AttributeError: renderer` / `KeyError: enable_rt`；
  env 缺 `action_config` 属性时补挂。
- `robosynchallenge/tasks/` **一个字节都没改**：判定仍是官方版。

## 范围是怎么定的

用官方 `random` 配置逐物体扫网格（`analyze_rigid_spawn_range.py --event <事件> --grid-size gx gy --trials-per-point 3`，
判据 = 专家动作图能否生成），取不含「三次全失败」点的最大轴对齐矩形。扫描只完成了一部分就按要求停了（不占卡），
没扫到的物体用官方成功专家数据的位置分布（2026-08-27 覆盖审计）和实测的臂可达半径旁证；
handle_basket 的专家代码在上游就是坏的（见下），只能按抓取几何推断。**改后配置没有再跑仿真验证**，
验证命令写在 docs 的 §5，跑之前先确认可以占卡。

## 注意

- **上游 bug**：`HandleBasketEnv.__init__` 不保存 `kwargs["action_config"]`，`create_demo_action_list` 还调用不存在的
  `_sync_carry_basket_runtime_attrs()`；`origin/main` 同样。评测不受影响，采集/分析在第一步就抛 AttributeError。
- 这条分支不含 `.venv` / `tests/`，仿真要用 `main` worktree 的 `.venv`：
  `cd` 到本目录后 `../RoboSynChallenge/.venv/bin/python scripts/analyze_rigid_spawn_range.py ...`。
  资源路径由 `robosynchallenge/data/asset_resolver.py` 按**包根目录**（editable 安装指向 main）解析，gym_config 放哪都行。
- 改了 `random` 就等于改了评测分布：本分支下的成功率与官方口径（origin/main 的 random）**不可直接对表**，报告里要写明。
- 可达域本质是以臂基座为圆心的扇形，矩形只能取子集；要保留完整可行域得加「距臂基座距离」约束的随机化 functor，本分支没做。
