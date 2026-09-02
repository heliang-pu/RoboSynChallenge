# sample_loading / coverage_gap_rack_x_lo

- 用途: 密度缺口补采(kind=marginal)
- 建议集数: 130
- 事件模式: pair-constrained(约束对采样)
- 依据: rack_x in [0.630, 0.656] holds 19 kept episodes vs 282.4 expected (8-bin uniform)
- 几何依据: half_extents cube=[0.00881, 0.00871, 0.10239], rack=[0.1445, 0.06098, 0.04163](mesh AABB x body_scale x 1.05), min_xy_clearance=0.005, max_xy_center_distance=0.5(非约束上界)
- 离线 MC 可行率(2 万采样): 79.6%

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| cube | position_range | [[0.45, -0.28, 0.86], [0.68, 0.0, 0.86]] | [[0.45, -0.28, 0.86], [0.68, 0.0, 0.86]] |
| cube | rotation_range | [[-20, 0, 0], [20, 0, 0]] | [[-20, 0, 0], [20, 0, 0]] |
| rack | position_range | [[0.63, 0, 0.865], [0.7, 0.15, 0.865]] | [[0.63, 0.0, 0.865], [0.65625, 0.15, 0.865]] |
| rack | rotation_range | [[0, 0, 0], [0, 0, 90]] | [[0.0, 0.0, 0.0], [0.0, 0.0, 90.0]] |

## 采集

```bash
python -m scripts.run_env_seeded \
    --gym_config configs/sample_loading/coverage_gap_rack_x_lo/gym_config.json \
    --action_config configs/sample_loading/action_config.json \
    --num_envs 1 --max_episodes 130 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/sample_loading/coverage_gap_rack_x_lo`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/sample_loading/coverage_summary.json。
