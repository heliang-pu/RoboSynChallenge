# table_rearrangement / coverage_strat_fork10_spoon11

- 用途: 全范围分层补匀(kind=strat)
- 建议集数: 31
- 事件模式: pair-constrained(约束对采样)
- 依据: stratified block over the official random range (no pose columns in the dataset): fork_x[0.525,0.650] fork_y[0.150,0.225] spoon_x[0.525,0.650] spoon_y[-0.225,-0.150]
- 几何依据: half_extents fork=[0.10648, 0.01301, 0.00953], spoon=[0.07622, 0.01296, 0.0072](mesh AABB x body_scale x 1.05), min_xy_clearance=0.03, max_xy_center_distance=0.49(非约束上界)
- 离线 MC 可行率(2 万采样): 100.0%

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| fork | position_range | [[0.4, 0.15, 0.83], [0.65, 0.3, 0.83]] | [[0.525, 0.15, 0.83], [0.65, 0.225, 0.83]] |
| fork | rotation_range | [[0, 0, -45], [0, 0, 45]] | [[0.0, 0.0, -45.0], [0.0, 0.0, 45.0]] |
| spoon | position_range | [[0.4, -0.3, 0.83], [0.65, -0.15, 0.83]] | [[0.525, -0.225, 0.83], [0.65, -0.15, 0.83]] |
| spoon | rotation_range | [[0, 0, -45], [0, 0, 45]] | [[0.0, 0.0, -45.0], [0.0, 0.0, 45.0]] |

## 采集

```bash
python -m scripts.run_env \
    --gym_config configs/table_rearrangement/coverage_strat_fork10_spoon11/gym_config.json \
    --action_config configs/table_rearrangement/action_config.json \
    --num_envs 1 --max_episodes 31 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/table_rearrangement/coverage_strat_fork10_spoon11`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/table_rearrangement/coverage_summary.json。
