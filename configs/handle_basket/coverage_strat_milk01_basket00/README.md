# handle_basket / coverage_strat_milk01_basket00

- 用途: 全范围分层补匀(kind=strat)
- 建议集数: 31
- 事件模式: keep-original(仅收窄官方事件范围)
- 依据: stratified block over the official random range (no pose columns in the dataset): basket_x[0.600,0.650] basket_y[-0.150,-0.125] milk_x[0.600,0.675] milk_y[0.175,0.300]

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| milk | position_range | [[0.6, 0.05, 0.83], [0.75, 0.3, 0.83]] | [[0.6, 0.175, 0.83], [0.675, 0.3, 0.83]] |
| milk | rotation_range | [[0, 0, 0], [0, 0, 0]] | [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] |
| basket | position_range | [[0.6, -0.15, 0.83], [0.7, -0.1, 0.83]] | [[0.6, -0.15, 0.83], [0.65, -0.125, 0.83]] |
| basket | rotation_range | [[0, 0, 0], [0, 10, 0]] | [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]] |

## 采集

```bash
python -m scripts.run_env_seeded \
    --gym_config configs/handle_basket/coverage_strat_milk01_basket00/gym_config.json \
    --action_config configs/handle_basket/action_config.json \
    --num_envs 1 --max_episodes 31 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/handle_basket/coverage_strat_milk01_basket00`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/handle_basket/coverage_summary.json。
