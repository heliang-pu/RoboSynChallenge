# 目录、命名与格式约定

## 目录

```text
lerobot_dataset/
├── rollouts/<task>_<tag>/                 rollout 交付版(v2.1 + episode_success.json)
├── simrecap_<task>_<tag>/                 本轮发布的 reward 版合并池(v2.1,三列)→ 链进 policy/pi05/training_data/RoboSynChallenge/
├── sample_loading_syn/<变体>/              8 组分布变体合成数据(configs/sample_loading/syn_*,v2.1)
└── .simrecap_work/                        隐藏工作目录,全部 v3.0
    ├── _cache/expert_v30_<指纹>/          专家上转缓存(跨轮复用,.fingerprint 标记完成;边车随带)
    └── <task>_<tag>/
        ├── shards/s0,s1/<robot>_<scene>_<task>_NNN/   rollout 分片(记录器自建子目录)
        ├── rollout_v30/rollout_merged/    分片合并后的 rollout(+边车)
        ├── expert_v30 -> _cache/...       软链
        ├── merged_v30/                    价值训练/推理用的合并池(阶段 6 在此原地写三列)
        ├── qc_v30/                        质检副本(qc<step> 后缀列,可删)
        ├── no_reward_v30/ reward_v30/     发布中间态(转完即 v2.1)
outputs/value_train/value_<task>_<tag>/checkpoints/000500 … last/
NAS: /home/phl/FermiBotNas/dataset/RoboSynChallenge/{recap_reward_dataset,recap_no_reward_dataset}/simrecap_<task>_<tag>/
```

## 标签与列

| 名称 | 粒度 | 位置 | 值 | 生产者 → 消费者 |
|---|---|---|---|---|
| `episode_success.json` 边车 | 集 | 数据集根目录 | `{"episode_index", "success": bool}`(rollout 边车另有 `seed`、`env_steps`),`saved_episode_count` | eval_policy / 人工复核 → label_rollout_dataset |
| `episode_success` 列 | 集 | v3.0 `meta/episodes/*.parquet` | `"success"/"failure"` | label_rollout_dataset → 价值训练与推理(奖励构造需要) |
| `complementary_info.value_<tag>` | 帧 | data parquet | [-1,0] 归一化 return-to-go | value_infer → 分析 |
| `complementary_info.advantage_<tag>` | 帧 | data parquet | Σ50步奖励 + V(t+50) − V(t) | value_infer → 二值化 |
| `complementary_info.acp_indicator_<tag>` | 帧 | data parquet | 0/1,任务内 top-30% | value_infer → openpi `ACPAdvantageTag`(训练拼 prompt,30% dropout;推理无该键 → 自动 positive) |

- v2.1 的 `episodes.jsonl` 会带上 `episode_success`,但上转器和打标脚本都不读它——**跨轮只认
  `episode_success.json` 边车**(两个方向的转换器都会丢边车,脚本负责带回);三列是普通数据列,v2.1 原样携带。
- 合并池布局固定 `[专家…, rollout…]`,rollout 全局索引 = 专家集数 + 边车索引。
- 价值目标:`g = −剩余步数 − c_fail·[失败]`,`c_fail = 任务内最长集 × 1.0`,归一化到 [-1,0];
  专家集固定 372 步、失败 rollout 600 步,同一归一化下自洽。

## 格式与工具

| 转换 | 工具 | 解释器 |
|---|---|---|
| v2.1 → v3.0 | `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id <name> --root <parent> --push-to-hub false`(原地,先复制;残留 `<name>_old/` 可删) | robosyn |
| v3.0 → v2.1 | `scripts/convert_lerobot3.0_to_2.1.py --repo-id <name> --root <parent>`(原地,先剥元数据) | robosyn |
| 合并 v3.0 | `lerobot_edit_dataset --operation.type merge`,`HF_LEROBOT_HOME=<parent>` | evo-rl + PYTHONPATH |
| 校验 v2.1 | `scripts/validate_lerobot_dataset.py <dir> --expected-episodes N --producer-exit-code 0` | pi05 venv(训练读取门) |

schema(sample_loading):`action/observation.state/qvel/qf` float32[14],三路相机 480×640 video,
`cube_pose/rack_pose` float32[4,4],25 fps,`timestamp = frame_index/fps`(逻辑时基,严格 0.04s)。
rollout 与专家由同一记录器产出,schema 逐项一致。
