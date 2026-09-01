# water_pouring / coverage_gap_cup_x_lo

- 用途: 密度缺口补采(kind=marginal)
- 建议集数: 190
- 事件模式: pair-constrained(约束对采样)
- 依据: cup_x in [0.560, 0.613] holds 18 kept episodes vs 199.5 expected (8-bin uniform); note: cup_x of the expert data is shifted vs the current official range, so this region lacks on-support data
- 几何依据: half_extents bottle=[0.03557, 0.24752, 0.0351], cup=[0.03115, 0.0355, 0.04607](mesh AABB x body_scale x 1.05), min_xy_clearance=0.02, max_xy_center_distance=0.5(非约束上界)
- 离线 MC 可行率(2 万采样): 97.9%

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| bottle | position_range | [[0.55, -0.15, 0.83], [0.7, 0.0, 0.83]] | [[0.55, -0.15, 0.83], [0.7, 0.0, 0.83]] |
| bottle | rotation_range | [[0.0, -180, 0.0], [0.0, 180, 0.0]] | [[0.0, -180, 0.0], [0.0, 180, 0.0]] |
| cup | position_range | [[0.56, 0.05, 0.85], [0.7, 0.3, 0.85]] | [[0.56, 0.05, 0.85], [0.6125, 0.3, 0.85]] |
| cup | rotation_range | [[0.0, 0.0, -180], [0.1, 0.1, 180]] | [[0.0, 0.0, -180.0], [0.1, 0.1, 180.0]] |

## 采集

```bash
python -m scripts.run_env \
    --gym_config configs/water_pouring/coverage_gap_cup_x_lo/gym_config.json \
    --action_config configs/water_pouring/action_config.json \
    --num_envs 1 --max_episodes 190 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/water_pouring/coverage_gap_cup_x_lo`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/water_pouring/coverage_summary.json。
