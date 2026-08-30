# item_assembly / coverage_gap_guijiao1_dy_hi

- 用途: 密度缺口补采(kind=marginal)
- 建议集数: 55
- 事件模式: pair-constrained(约束对采样)
- 依据: guijiao1_dy in [0.007, 0.010] holds 46 kept episodes vs 91.6 expected (8-bin uniform)
- 几何依据: half_extents guijiao1=[0.10632, 0.02049, 0.02049], guijiao2=[0.10586, 0.02416, 0.02416](mesh AABB x body_scale x 1.05), min_xy_clearance=0.03, max_xy_center_distance=0.37(非约束上界)
- 离线 MC 可行率(2 万采样): 100.0%

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| guijiao1 | position_range | [[-0.01, -0.01, 0.0], [0.01, 0.01, 0.0]] | [[-0.01, 0.0075, 0.0], [0.01, 0.01, 0.0]] |
| guijiao1 | rotation_range | - | [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] |
| guijiao2 | position_range | [[-0.01, -0.01, 0.0], [0.01, 0.01, 0.0]] | [[-0.01, -0.01, 0.0], [0.01, 0.01, 0.0]] |
| guijiao2 | rotation_range | - | [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] |

## 采集

```bash
python -m scripts.run_env \
    --gym_config configs/item_assembly/coverage_gap_guijiao1_dy_hi/gym_config.json \
    --action_config configs/item_assembly/action_config.json \
    --num_envs 1 --max_episodes 55 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/item_assembly/coverage_gap_guijiao1_dy_hi`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/item_assembly/coverage_summary.json。
