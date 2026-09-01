# 并行评估（`num_envs > 1`）

`scripts/eval_policy.py` 支持在**单进程内同时跑 N 个环境**做评估（wave 批次模式），
利用 EmbodiChain/DexSim 的多 arena 批量仿真（PhysX GPU 物理 + camera group 批量渲染）。
与既有的 `num_shards`/`shard_index` 多进程分片正交，两者可叠加。

```bash
python scripts/eval_policy.py --config policy/act/deploy_policy.yml \
    --overrides --task_name click_bell --setting random \
    --max_episodes 100 --num_envs 8 --device cuda --headless True
```

要点：

- **必须 `--device cuda`**。DexSim 在 CPU 设备下对多环境退化成逐 env 的 Python
  循环（`articulation.py` 的 setter 分支），没有并行收益；脚本会给出警告但不阻止。
- `num_envs=1`（默认）时走原有串行循环，行为与改动前逐字节一致。
- **N 不是越大越好，先测再定**。本机 4090（click_bell clear + 轻量 ACT、361 步
  全跑满的最差情形）实测吞吐（env-steps/s，括号内相对串行）：

  | N=1 | N=2 | N=3 | N=4 | N=6 | N=8 |
  |---|---|---|---|---|---|
  | 50 (1.0×) | 66 (1.3×) | 77 (1.5×) | **86 (1.7×，甜点)** | 49 (1.0×) | 53 (1.1×) |

  瓶颈是渲染不是显存（N=8 时整卡也就 ~10GB）：每步渲 3 相机 × N 路 view，
  **4→6 之间有个断崖**（21.4 → 8.2 wave step/s，疑似 12 路 view 以上触发批渲
  资源阈值），所以这张卡就用 N=4。推理越重的策略批量摊薄越多，但断崖是渲染侧的，
  甜点大概率仍在 4；换卡（pro6000/A100）用 1 个 wave 试跑重新定位。
  偶发：上一个 sim 进程刚被杀掉后立即建环境可能在 URDF 装配段 segfault，
  重跑即可（瞬态，非 N 相关）。

## 种子协议：与串行同一批种子、同一批场景

并行模式**不改变种子口径**，可与串行/分片结果直接对表：

1. rng 仍按 episode 序号 0..M-1 逐个抽种子（与串行完全同序），分片过滤规则照旧，
   episode k 拿到的 seed_k 与「单进程串行跑满」时完全一致；
2. 每个 episode 用 `env.reset(seed=seed_k, options={"reset_ids": [slot]})` 单独播种
   重置自己的槽位。EmbodiChain 的 reset 会 `torch.manual_seed(seed_k)` 后只为这
   1 个 env 抽随机化样本，RNG 消费模式与单环境相同——**物体场景与单环境同种子
   逐位一致**（click_bell random 下实测：button 位姿差 ≤6e-8，cpu/cuda、跨进程
   均成立）。

已知偏差：

- **机器人初始 qpos 在 slot>0 与单环境同种子不逐位一致**（实测差可达 ~1 rad）。
  锚点（重置前 qpos）两边一致、随机偏移的种子一致，但 `randomize_robot_eef_pose`
  的 FK/IK 在部分 `env_ids` 路径下与单环境路径给出不同解（疑为 warp/IK 内核在
  部分索引下的分支差异，与此前修掉的高级索引 bug 同族，待上游根修）。分布同族、
  幅度同范围，成功率统计等价；但**逐 seed 复现单集轨迹仍需 `num_envs=1`**。
  slot 0（每 wave 首个重置的槽位）不受影响。另注意：**首个 wave 必须从 slot 0
  开始按序重置**——先重置非 0 槽位会触发 `restore_visual_material` 的惰性初始化
  越界（上游 bug，本实现天然按序，不会踩到）。

- `mode: interval` 的随机化事件（如 `randomize_light` 每 10 步一次）在批量下从
  同一条全局 RNG 流抽样，数值与串行不同但同分布；
- 批量物理与单环境物理不保证逐位一致（broadphase 顺序等），与「换台机器跑」同
  量级的数值差异。

**结论**：并行模式的成功率与串行口径统计等价、种子集相同，但不是逐 episode 复
读机。要复现单个 episode 的轨迹（调试、录像）请用 `num_envs=1`。

## Wave 批次语义

`max_episodes` 个 episode 按 `num_envs` 分成若干 wave。每个 wave：

1. 逐槽位带种子部分重置（末尾不满的 wave 用首个种子填充空槽，保证整批
   `elapsed_steps` 同步、统一在 `max_env_steps` 截断，空槽不计入统计）；
2. `eval_reset_sync_steps` 的物理沉降在整个 wave 重置完后执行一次（串行是每
   episode 一次，语义等价：每个槽位都是「设完状态 → k 步沉降 → 首帧观测」）；
3. `reset_model` 每 wave 调一次（等价于 N 个 episode 同时开始）；
4. 反复调用策略适配器的 `eval()`，直到所有在评槽位「成功已锁存或已截断」。

**成功判定与官方口径逐条对应**（由 `ParallelEvalProxy` 裁判）：

- 每次 `env.step()` 后读 `is_task_success()`（per-env 张量），成功且该 env 当步
  与此前均未截断 → 锁存成功、记录步数；
- 截断步不计成功：底层 `env.step()` 会在截断步对 done env 自动部分重置并清掉任
  务内部锁存，两边行为一致（串行模式同样如此）；
- 成功一经锁存不回退——等价于官方串行在成功当步立即 break 结束 episode。

## 策略适配器兼容性

适配器契约**不需要改**。代理把 `env.get_wrapper_attr("is_task_success")` 拦截成
「所有在评 env 都完成才 True」的标量，按官方模板写的 `if ...: break` 原样工作；
适配器照常整 chunk 驱动 `env.step()`。

真正的要求只有一条：**推理和动作要带 batch 维**——obs 本来就是 `[N, ...]`，适配
器必须把它整批喂给模型并 `env.step([N, dof])`。代理会在收到第一维 ≠ N 的动作时
直接报错（防止「静默只评了 env 0」）。

| 策略 | 状态 |
|---|---|
| `act`（进程内路径） | ✅ 张量流天然带 batch（LeRobot select_action/action queue 支持批量），实测通过 |
| `act`（`act_python` worker 路径） | ❌ worker 协议单观测，`--num_envs 1` |
| `smolvla` / `pi05_lerobot`（跨进程 worker） | ❌ 同上，需扩 worker 协议 |
| `pi05`（openpi/JAX + RTC 调度器） | ❌ 适配器显式 `qpos[0]`、调度器按单环境写，批量化是后续工作 |
| 其他 | 未验证，代理的 batch 检查会兜底报错 |

## 与其他功能的关系

- `rollout_save`：**互斥**（直接报错退出）。EmbodiChain 的 LeRobot 录制 buffer 用
  全局 `current_rollout_step`，N 个 env 异步结束会互相截断数据。
- `eval_video_log`：并行下自动关闭（告警）。录像请用 `num_envs=1`。
- 推理计时：并行下每次计时是**一批 N 个 episode 共享的调用**，
  `evaluation_metrics.json` 的 `inference_timing_scope` 会注明；延迟数字与单环境
  口径不可比，要测部署延迟用 `num_envs=1`。
- 结果文件新增 `config.num_envs` 字段。

## 实现位置

- `scripts/eval_policy.py`：`ParallelEvalProxy`（裁判代理）、
  `run_parallel_episodes`（wave 循环）、`settle_after_wave_reset`；
  `main()` 里 `num_envs > 1` 分支。串行循环体保持原样。
- 依赖的 EmbodiChain 机制：`EnvCfg.num_envs`（`make_env_from_configs` 已透传）、
  partial reset（`options["reset_ids"]`）、camera group 批量渲染、
  `reset(seed)` 的全局播种 + 按 `env_ids` 抽样。
