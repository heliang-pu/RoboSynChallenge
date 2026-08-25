---
name: sim-recap
description: Run a sim-RECAP round in RoboSynChallenge — policy rollouts with automatic success labels, value-function (pistar06) training and checkpoint selection, advantage/indicator write-back, and publishing LeRobot v2.1 datasets (reward / no-reward) for pi0.5 ACP fine-tuning. Use for "跑一轮 recap", "价值函数训练/质检", "advantage 写回", "发布 recap 数据集", "ACP 微调 pi0.5". Not for plain expert data collection or ordinary SFT.
---

# sim-RECAP(仿真里的优势条件化迭代训练)

思想:π\*0.6 RECAP 的三步——成败标签 → 价值函数 → advantage 二值化后以文本
`Advantage: positive/negative` 拼进 prompt 微调 VLA,部署时永远挂 positive。
仿真评估器替代人工标注,整个闭环无人值守。实现取自 Evo-RL(收编在
`third_party/evo_rl/`),**打标规则按 Evo-RL 原实现**:任务内全帧混算
top-30%(`positive_ratio=0.3`),不区分专家/rollout,不做后处理。

## 一轮的形状

```text
π_k rollout(评估器自动判成败)──┐
                                  ├→ 合并池(专家 + rollout,v3.0)→ 打 episode_success
专家演示(天然全 success)────────┘          │
                              价值训练(全量微调 SigLIP+Gemma,~12h/8000 步)
                                           │  ← 质检:选 advantage 信号最强的 checkpoint(≈1 epoch 处)
                              推理写回 value/advantage/indicator 三列(后缀 _<round>)
                                           │
                     发布 v2.1:reward 版(三列)+ no_reward 版(对照)→ NAS + 本地
                                           │
                              ACP 微调 pi0.5(pi05_sim_recap)→ π_{k+1} → 仿真评估
```

所有阶段都有现成脚本 `launch/recap/01..08_*.sh`(人手可跑,不依赖 AI),用法与判定规则见
[references/runbook.md](references/runbook.md);人手版手册在 `docs/tutorials/sim_recap.md`。

## 不可违反的约定

- **对外一律 v2.1,v3.0 只活在 `lerobot_dataset/.simrecap_work/`**。价值栈(Evo-RL)只认
  v3.0;交付前转 v2.1 并做列存活门禁。目录/列名/边车约定见
  [references/data-layout.md](references/data-layout.md)。
- **专家数据用清洗版**(NAS `Sim_clean_filtered`,sample_loading 为 756 集),不要用
  本地 1000 集原始版:RECAP 把专家整体标 success,原始版里被剔除的脏轨迹会污染价值函数。
- **rollout 采集用任务专属 `random_rollout` 设置**(排除物理无解场景),**评测仍用官方
  `random`**。新任务照 `configs/sample_loading/random_rollout/README.md` 的几何推导复制。
- **价值函数不要训满**:小数据池(几百集)上 0.7B 参数会记忆化,advantage 按 Bellman
  恒等式塌向 0。经验:sample_loading 350 集,3000 步(≈1.2 epoch)可用,6500 步 advantage std 塌 5-6 倍(0.030 → 0.005)。
  必须用 `launch/recap/05_value_qc.sh` 质检后再选 checkpoint。
- **长任务用 `setsid nohup … < /dev/null & disown` 启动**,否则会话重启会连带杀掉训练。
- **停进程按 PID 杀,`pgrep -f` 模式用 `[x]` 括号技巧**(`'lerobot_value_trai[n]'`),
  否则会匹配并杀掉自己的 shell;杀 eval/采集主进程后必须清理 image-writer 孤儿子进程,
  否则 20+GB 显存不释放。
- 人工复核过的成败标签(`episode_success.json`)优先于评估器判定;用 `launch/recap/02_set_label.sh`
  改,它同步三处(v2.1 交付版、v3.0 工作版、分片)。边车是跨轮唯一的标签载体:v2.1→v3.0 上转器会
  丢掉非标准文件,03_build_pool.sh 会把边车带进缓存;上轮池没有边车时脚本会拒绝(否则失败集会被标成功)。

## 环境(三套,不要混)

| 阶段 | 解释器 |
|---|---|
| rollout / 评估 / v2.1 训练读取校验 | `policy/pi05/.venv/bin/python`(eval.sh 内部处理) |
| 转换、打标、剥元数据、边车 | `~/miniconda3/envs/robosyn/bin/python`(lerobot 0.4.4 + pyarrow,pandas 3) |
| 合并、价值训练、价值推理 | `~/miniconda3/envs/evo-rl/bin/python` + `PYTHONPATH=third_party/evo_rl/src`(pandas 2) |

踩过的坑与修法见 [references/pitfalls.md](references/pitfalls.md)——动手前先读,每条都是真实事故。
