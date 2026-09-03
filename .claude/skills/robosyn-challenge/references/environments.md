# 环境分层、常用命令、测试

## 环境分层（最容易踩的坑）

**仓库刻意不提供统一环境**，仿真侧和训练侧的 torch/JAX 版本直接冲突，不要试图合并：

| 环境 | 位置 | 用途 |
|---|---|---|
| 仿真/采集/评估 | 仓库根 `.venv`（uv，Python 3.11 钉死，torch 2.10+cu128） | `scripts/run_env_seeded.py`、`scripts/eval_policy_parallel.py`、`launch/*.sh` |
| 策略训练 | `policy/{act,dp,pi0,pi05}/`（各自 pyproject + uv.lock，已提交） | `finetune.sh` 内部走 `uv run --frozen`，首次运行自动建环境 |
| LeRobot pi0.5 | `policy/pi05_lerobot/`（uv，**Python 3.12**，lerobot 锁到含 MEM 的 git rev） | 上游要 3.12 而仿真侧钉 3.11，装不进同一解释器 |
| 重型策略 | `policy/{dm05,g05,motus,xr1,lila_wam}/` 的 `setup_env.sh` / README | 含 flash-attn、DeepSpeed 等自编译组件，不进 uv |

`policy/{g05,motus,xr1,lila_wam}` 的上游是 git submodule，clone 后需
`git submodule update --init`。

## 常用命令

```bash
# 任务清单
bash launch/_print_available_tasks.sh

# 采集：<task> <setting(clear|random)> <format(3_0|2_1)>
bash launch/run_task_seeded.sh click_bell clear 2_1 --max_episodes 100 --headless
bash launch/collect_validated_batch.sh ...     # 带校验门，产出即训练就绪
python scripts/validate_lerobot_dataset.py ... # 首个失败门即非零退出，可当硬门禁

# 评估（统一入口，各 policy 的 eval.sh 只是包装）
python scripts/eval_policy_parallel.py --config policy/<name>/deploy_policy.yml \
    --overrides --task_name click_bell --setting random --max_episodes 100

# 训练
cd policy/act  && bash finetune.sh <dataset_root> outputs/train/act_x 0
cd policy/pi05 && bash finetune.sh pi05_click_bell my_exp 0
cd policy/pi05/train_scripts && ./train_click_bell.sh 0   # 10 个任务各有一键脚本
```

各 `finetune.sh` 头部注释就是该策略的完整用法/超参手册，**改训练前先读它**。

`scripts/` 有 40 个脚本、`launch/` 有 30 多个：`scripts/README.md` 和 `launch/README.md`
是逐脚本的用途/参数手册，**先查表再读源码**。

## 测试

pytest 装在根 `.venv`：

```bash
.venv/bin/python -m pytest tests -q --ignore=tests/test_rtc_pi05.py
.venv/bin/python -m pytest tests/test_seeded_collection.py -q   # 单文件
```

历史上必须写成 `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ...`，因为 ROS2 jazzy 把
`/opt/ros/jazzy/lib/python3.12/site-packages` 泄漏进 `PYTHONPATH`，pytest 插件自动加载会
import `launch_testing` 并崩在 `No module named 'lark'`。本机 ROS2 已于 2026-09-01 卸载，
**但在还装着 ROS 的机器（4090 / pro6000 等）上仍要带这两个前缀**。

已知失败（不是你改坏的）：`test_rtc_pi05.py` 需要 JAX，根环境 collect 就报错（故 `--ignore`），
只能在 `policy/pi05` 环境里跑。

## 采集流水线

`launch/collect_validated_batch.sh` 在 env 循环外层：调 `run_task_seeded.sh` 采集 → 从日志
抓数据集路径 → 跑 5 道门禁（`validate_lerobot_dataset.py`）→ 转 v2.1 并二次验证。
`collect_parallel_validated.sh` 是**多进程分片 + 合并**（每 worker 独立进程与独立 dataset，
最后 `lerobot_edit_dataset --operation.type merge`）。

`feat/parallel-collect` 分支另有**单进程多环境并行采集**：批量环境负责摆场景/执行/渲染/录制，
专家规划放在单环境 worker 子进程里（两边同 seed 场景逐位一致），10 个任务的 action bank
零改动。要点见该分支的 `docs/parallel_collection.md`。

## 已知的上游/环境坑

- **task action bank 假设规划结果在 cpu**（`ret.positions[0].numpy()`），cuda 设备下需在
  `MotionGenerator` 出口把结果搬回 cpu（不要改 `robosynchallenge/tasks/`，那是红线）。
- **随机化绑定设备**：随机化走 env 设备的 RNG，cuda 与 cpu 同种子产出不同场景，
  两个体系各自内部可复现、不可互换。
- **多环境部分重置**：`current_rollout_step` 是全 env 共享标量，部分重置会清零；
  `recorder.finalize()` 会把 step>0 的 buffer 再存一遍。并行采集因此在 reset 前直接
  `dataset_manager.apply(mode="save", env_ids=...)`，并在 finalize 前清零写指针。
