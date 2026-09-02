# item_assembly / coverage_strat_guijiao101

- 用途: 全范围分层补匀(kind=strat)
- 建议集数: 38
- 事件模式: pair-constrained(约束对采样)
- 依据: stratified xy quadrant of guijiao1 for uniform replenishment over the full official range: guijiao1_dx[-0.010,0.000] guijiao1_dy[0.000,0.010]
- 几何依据: half_extents guijiao1=[0.10632, 0.02049, 0.02049], guijiao2=[0.10586, 0.02416, 0.02416](mesh AABB x body_scale x 1.05), min_xy_clearance=0.03, max_xy_center_distance=0.37(非约束上界)
- 离线 MC 可行率(2 万采样): 100.0%

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| guijiao1 | position_range | [[-0.01, -0.01, 0.0], [0.01, 0.01, 0.0]] | [[-0.01, 0.0, 0.0], [0.0, 0.01, 0.0]] |
| guijiao1 | rotation_range | - | [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] |
| guijiao2 | position_range | [[-0.01, -0.01, 0.0], [0.01, 0.01, 0.0]] | [[-0.01, -0.01, 0.0], [0.01, 0.01, 0.0]] |
| guijiao2 | rotation_range | - | [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] |

## 采集

```bash
python -m scripts.run_env_seeded \
    --gym_config configs/item_assembly/coverage_strat_guijiao101/gym_config.json \
    --action_config configs/item_assembly/action_config.json \
    --num_envs 1 --max_episodes 38 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/item_assembly/coverage_strat_guijiao101`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/item_assembly/coverage_summary.json。
