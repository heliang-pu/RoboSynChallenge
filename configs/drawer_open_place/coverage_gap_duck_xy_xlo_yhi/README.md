# drawer_open_place / coverage_gap_duck_xy_xlo_yhi

- 用途: 密度缺口补采(kind=xy)
- 建议集数: 40
- 事件模式: keep-original(仅收窄官方事件范围)
- 依据: duck xy block x[0.530,0.620] y[0.293,0.350] holds 54 kept episodes vs 108.8 expected (4 grid cells)

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| duck | position_range | [[0.53, 0.18, 0.87], [0.8, 0.35, 0.87]] | [[0.53, 0.29333, 0.87], [0.62, 0.35, 0.87]] |
| duck | rotation_range | [[0, 0, -30], [0, 0, 30]] | [[0.0, 0.0, -30.0], [0.0, 0.0, 30.0]] |

## 采集

```bash
python -m scripts.run_env_seeded \
    --gym_config configs/drawer_open_place/coverage_gap_duck_xy_xlo_yhi/gym_config.json \
    --action_config configs/drawer_open_place/action_config.json \
    --num_envs 1 --max_episodes 40 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/drawer_open_place/coverage_gap_duck_xy_xlo_yhi`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/drawer_open_place/coverage_summary.json。
