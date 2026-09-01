# mixer_operating / coverage_strat_beaker_mixer01

- 用途: 全范围分层补匀(kind=strat)
- 建议集数: 125
- 事件模式: pair-constrained(约束对采样)
- 依据: stratified xy quadrant of beaker_mixer for uniform replenishment over the full official range: beaker_mixer_x[0.540,0.595] beaker_mixer_y[0.050,0.100]
- 几何依据: half_extents beaker=[0.04773, 0.0443, 0.1139], beaker_mixer=[0.13489, 0.0452, 0.15778](mesh AABB x body_scale x 1.05), min_xy_clearance=0.02, max_xy_center_distance=0.43(非约束上界)
- 离线 MC 可行率(2 万采样): 100.0%

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| beaker | position_range | [[0.63, -0.25, 0.85], [0.75, -0.2, 0.85]] | [[0.63, -0.25, 0.85], [0.75, -0.2, 0.85]] |
| beaker | rotation_range | [[0, 0, -180], [0, 0, 180]] | [[0, 0, -180], [0, 0, 180]] |
| beaker_mixer | position_range | [[0.54, 0.0, 0.9], [0.65, 0.1, 0.9]] | [[0.54, 0.05, 0.9], [0.595, 0.1, 0.9]] |
| beaker_mixer | rotation_range | [[0, 0, 0], [0, 0, 0]] | [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] |

## 采集

```bash
python -m scripts.run_env \
    --gym_config configs/mixer_operating/coverage_strat_beaker_mixer01/gym_config.json \
    --action_config configs/mixer_operating/action_config.json \
    --num_envs 1 --max_episodes 125 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/mixer_operating/coverage_strat_beaker_mixer01`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/mixer_operating/coverage_summary.json。
