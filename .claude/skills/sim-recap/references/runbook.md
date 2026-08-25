# sim-RECAP runbook(脚本版)

所有命令在仓库根目录执行;脚本都在 `launch/recap/`,共同定义(路径、解释器、模型路径)在
`launch/recap/_common.sh`,换机器只改那一个文件。`<task>` 如 `sample_loading`,`<tag>` 如 `round1`。

| 阶段 | 命令 | 产物 / 说明 |
|---|---|---|
| 0 前置 | `nvidia-smi`;`launch/recap/stop.sh <cmdline子串>` 清场 | rollout 每片需 ~16-20GB;别误杀用户自己的 `/home/phl/Datacollect_T/covered_eval_run/` 或 `collect_until_valid` 采集 |
| 1 rollout | `bash launch/recap/01_rollout.sh <task> <tag> pi05_<task> <task> <ckpt_id> <episodes> [shards=2]` | 分片并行(各自 seed)→ 合并 → `lerobot_dataset/rollouts/<task>_<tag>`(v2.1+边车)+ 工作目录 `rollout_v30/rollout_merged`。GPU 独占 ~1 分钟/集 |
| 1b 复核 | 三视角看 `rollouts/<task>_<tag>/videos/`;`bash launch/recap/02_set_label.sh <task> <tag> <ep> success\|failure` | 评估器在稳定计数触发后就结束,看不到试管后来掉出;改标签三处同步 |
| 2-4 数据池 | `bash launch/recap/03_build_pool.sh <task> <tag> <专家目录> [expert_episodes]` | 专家 v2.1→v3.0(指纹缓存,边车随带)→ 可选前 N 集子集 → 合并 `[专家, rollout]` → 写 `episode_success`。round1 用 200 专家:150 rollout |
| 5 价值训练 | `bash launch/recap/04_value_train.sh <task> <tag> [steps=8000] [bs=64]` | 脱离会话;bs64 ≈ 27.5GB、5.4 s/步;1 epoch(350 集)≈2500 步;不必训满 |
| 6a 质检选档 | `bash launch/recap/05_value_qc.sh <task> <tag> 002000 003000 004000` | 60 集子集,每档 ~10 分钟(GPU 独占);规则:成功−失败首帧 value ≥ 0.3 的档里取 advantage std 最大者 |
| 停训 | `bash launch/recap/stop.sh lerobot_value_train` | 发布前必须停:阶段 6 会原地改写 merged_v30 |
| 6b-7 发布 | `setsid nohup bash launch/recap/06_publish.sh <task> <tag> <step> > /tmp/pub.log 2>&1 < /dev/null & disown` | no_reward 与 reward 两版 v2.1 → NAS `recap_{no_,}reward_dataset/simrecap_<task>_<tag>/` + 本地 `lerobot_dataset/simrecap_<task>_<tag>`(链进 pi05 训练目录);三列存活门禁;约 1.5h |
| 8 ACP 微调 | `bash launch/recap/07_acp_finetune.sh <task> <tag> <exp> [gpu] [权重 params 目录]` | 配置 `pi05_sim_recap` 由环境变量注入 repo_id/indicator/权重;norm stats 按 repo_id 分目录自动算;对照组 `SIMRECAP_INDICATOR_KEY=none` |
| 9 评估 | `bash launch/recap/08_eval.sh <task> <tag> <exp> [episodes=100]` | 官方 `random`;自动取最新 checkpoint(20k 步只存 10000/19999);推理自动挂 `Advantage: positive` |
| 下一轮 | `01_rollout.sh <task> round2 pi05_sim_recap <exp> 19999 …` 然后 `03_build_pool.sh <task> round2 lerobot_dataset/simrecap_<task>_round1`(自带边车) | `07` 的权重指向上轮 `checkpoints/pi05_sim_recap/<exp>/19999/params` |

## 参考数字(sample_loading round1)

- rollout 150 集:成功 11/150(评估器)→ 人工复核 10/150;`random_rollout` 下真实成功率 ≈7%(官方 random 3%)。
- 价值 loss:5.3 → 2.0(500)→ 1.4(2000)→ 1.2(3000)→ 0.83(5000)→ 0.72(6000+ 平台)。
- 质检:1500 步分离 0.11(欠训练);3000 步分离 0.43、A.std 0.030、近零 70%(**选它**);6500 步分离 0.69 但 A.std 0.005、近零 96%(记忆化)。
- 全量推理 350 集:GPU 共享时 ~1.5-2h,独占 ~1h。

## 打标语义(按 Evo-RL 原实现,不做后处理)

任务内**全帧混算** `np.quantile(adv, 1−0.3)`,`>=` 为 1;不区分专家/rollout;`force_intervention_positive`
读 `complementary_info.is_intervention`,仿真数据没有该列 → no-op。专家帧只有 ~10% 为正是正常结果
(专家可预测 → A≈0)。论文(π\*0.6)用同样的分位规则,迭代阶段比例 ~40%,Evo-RL 默认 0.3。
