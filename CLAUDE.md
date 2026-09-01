# CLAUDE.md

**当前 worktree = `feat/parallel-collect`，实验分支**：单进程多环境**并行专家数据采集**
（`--num_envs N` wave 批次 + 单环境规划 worker）。从 `feat/parallel-eval` 切出（复用其
种子协议与并行评估），做完连同 eval 分支一起合回 `main`。
设计/用法/口径见 [docs/parallel_collection.md](docs/parallel_collection.md)
与 [docs/parallel_eval.md](docs/parallel_eval.md)；本文件只记环境与坑。

仓库总体约定见 `main` worktree（`../RoboSynChallenge`）的 CLAUDE.md。
本 worktree **没有自己的 .venv**（软链到 main 的），也依赖 EmbodiChain 本地分支
`parallel-fixes-upstream`（= 官方 tag v0.2.4 + 并行修复）（partial-reset 修复 + 规划器 B=1 放行，未合上游）。

## 本分支改了什么

- `scripts/expert_plan_worker.py`（新）：单环境规划 worker。同 seed 复现场景 →
  原生 `create_demo_action_list` → 返回 `(T, dof)` 轨迹 + 官方初始 qpos。
  含 `MotionGenerator` 结果搬回 cpu 的 monkeypatch（tasks/ 红线不许改 bank）。
- `scripts/parallel_collection.py`（新）：wave 驱动层。per-slot 种子化部分重置 →
  init qpos 回写 → 尾帧 padding 整批执行 → 自适应静置 → per-env 官方判定 →
  `dataset_manager.apply(mode="save", env_ids=成功槽位)` 直接落盘 → 边车。
- `scripts/run_env.py`：`num_envs>1` 分支接线；同款 MotionGenerator cpu 补丁
  （串行 cuda 采集因此也可用）；并行模式不套 SeededCollection。
- `launch/run_task.sh` **零改动**：`--num_envs N` 作为附加参数即可覆盖写死的
  `--num_envs 1`（argparse 取末次出现）。

## 坑（都实测踩过）

- **落盘不能走 reset(save_data=True)**：`current_rollout_step` 是全 env 共享标量，
  部分重置清零后第二个槽位会存出空集 → 必须在 reset 前直接 `apply(mode="save")`。
- **`recorder.finalize()` 会把 step>0 的所有 env buffer 再存一遍**（给"最后一集没
  被 reset 冲洗"的流程兜底）→ 已显式落盘的流程要先把 `current_rollout_step` 清零
  再 finalize，否则重复集（validate 门禁 `expected 4 found 6` 就是它）。
- **种子体系绑定设备**：随机化走 env 设备的 RNG 生成器，cuda 与 cpu 同种子不同场景。
  并行采集（cuda）与日常串行采集（cpu）的 seed 不可互换复现；并行体系内部
  （worker 与批量环境同 cuda）逐位一致。
- **task bank 假设规划结果在 cpu**（`ret.positions[0].numpy()`），cuda 设备下必须
  经 MotionGenerator monkeypatch 搬运；不要去改 `robosynchallenge/tasks/`（红线）。
- worker 以脚本方式启动时 `sys.path[0]` 是 `scripts/`，editable 安装会把
  `robosynchallenge` 解析到 main worktree —— worker 头部已强制插入本 worktree 根。
- 部分重置下 robot 随机化与单环境路径不等价（详见 docs/parallel_eval.md 已知偏差），
  采集侧靠「worker 回写官方 init qpos」中和，**不要删这步**。
- 首 wave 的槽位重置必须从 slot 0 按序来（材质备份惰性初始化，乱序会 segfault）。
