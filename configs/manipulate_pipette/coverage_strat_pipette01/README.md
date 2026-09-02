# manipulate_pipette / coverage_strat_pipette01

- 用途: 全范围分层补匀(kind=strat)
- 建议集数: 125
- 事件模式: keep-original(仅收窄官方事件范围)
- 依据: stratified block over the official range of unobserved object pipette (pose column always zero in the dataset): pipette_x[0.450,0.585] pipette_y[-0.190,-0.100]

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| pipette | position_range | [[0.45, -0.28, 0.86], [0.72, -0.1, 0.86]] | [[0.45, -0.19, 0.86], [0.585, -0.1, 0.86]] |
| pipette | rotation_range | [[0, 0, -30], [0, 0, 30]] | [[0.0, 0.0, -30.0], [0.0, 0.0, 30.0]] |

## 采集

```bash
python -m scripts.run_env_seeded \
    --gym_config configs/manipulate_pipette/coverage_strat_pipette01/gym_config.json \
    --action_config configs/manipulate_pipette/action_config.json \
    --num_envs 1 --max_episodes 125 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/manipulate_pipette/coverage_strat_pipette01`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/manipulate_pipette/coverage_summary.json。
