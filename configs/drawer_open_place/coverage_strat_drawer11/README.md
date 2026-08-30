# drawer_open_place / coverage_strat_drawer11

- 用途: 全范围分层补匀(kind=strat)
- 建议集数: 37
- 事件模式: keep-original(仅收窄官方事件范围)
- 依据: stratified block over the official range of unobserved object drawer (pose column always zero in the dataset): drawer_dx[0.020,0.050] drawer_dy[-0.090,0.000]

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| drawer | position_range | [[-0.01, -0.18, 0.0], [0.05, 0.0, 0.0]] | [[0.02, -0.09, 0.0], [0.05, 0.0, 0.0]] |
| drawer | rotation_range | - | - |

## 采集

```bash
python -m scripts.run_env \
    --gym_config configs/drawer_open_place/coverage_strat_drawer11/gym_config.json \
    --action_config configs/drawer_open_place/action_config.json \
    --num_envs 1 --max_episodes 37 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/drawer_open_place/coverage_strat_drawer11`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/drawer_open_place/coverage_summary.json。
