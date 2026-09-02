# drawer_open_place / coverage_gap_duck_x_lo

- 用途: 密度缺口补采(kind=marginal)
- 建议集数: 40
- 事件模式: keep-original(仅收窄官方事件范围)
- 依据: duck_x in [0.530, 0.564] holds 23 kept episodes vs 123.0 expected (8-bin uniform)

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| duck | position_range | [[0.53, 0.18, 0.87], [0.8, 0.35, 0.87]] | [[0.53, 0.18, 0.87], [0.56375, 0.35, 0.87]] |
| duck | rotation_range | [[0, 0, -30], [0, 0, 30]] | [[0.0, 0.0, -30.0], [0.0, 0.0, 30.0]] |

## 采集

```bash
python -m scripts.run_env_seeded \
    --gym_config configs/drawer_open_place/coverage_gap_duck_x_lo/gym_config.json \
    --action_config configs/drawer_open_place/action_config.json \
    --num_envs 1 --max_episodes 40 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/drawer_open_place/coverage_gap_duck_x_lo`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/drawer_open_place/coverage_summary.json。
