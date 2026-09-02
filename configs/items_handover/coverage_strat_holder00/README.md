# items_handover / coverage_strat_holder00

- 用途: 全范围分层补匀(kind=strat)
- 建议集数: 38
- 事件模式: pair-constrained(约束对采样)
- 依据: stratified xy quadrant of holder for uniform replenishment over the full official range: holder_x[0.500,0.600] holder_y[0.000,0.125]
- 几何依据: half_extents pen=[0.07561, 0.00805, 0.00805], holder=[0.0485, 0.10131, 0.0485](mesh AABB x body_scale x 1.05), min_xy_clearance=0.03, max_xy_center_distance=0.48(非约束上界)
- 离线 MC 可行率(2 万采样): 87.9%

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| pen | position_range | [[0.52, -0.3, 0.884], [0.675, 0.0, 0.884]] | [[0.52, -0.3, 0.884], [0.675, 0.0, 0.884]] |
| pen | rotation_range | [[0, 0, -30], [0, 0, 30]] | [[0, 0, -30], [0, 0, 30]] |
| holder | position_range | [[0.5, 0.0, 0.884], [0.7, 0.25, 0.884]] | [[0.5, 0.0, 0.884], [0.6, 0.125, 0.884]] |
| holder | rotation_range | - | [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] |

## 采集

```bash
python -m scripts.run_env_seeded \
    --gym_config configs/items_handover/coverage_strat_holder00/gym_config.json \
    --action_config configs/items_handover/action_config.json \
    --num_envs 1 --max_episodes 38 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/items_handover/coverage_strat_holder00`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/items_handover/coverage_summary.json。
