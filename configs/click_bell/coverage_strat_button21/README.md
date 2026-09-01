# click_bell / coverage_strat_button21

- 用途: 全范围分层补匀(kind=strat)
- 建议集数: 55
- 事件模式: keep-original(仅收窄官方事件范围)
- 依据: stratified block over the official random range (no pose columns in the dataset): button_x[0.700,0.850] button_y[-0.100,0.100]

## 区间(config 单位,与官方 random 对比)

| 对象 | 参数 | 官方 random | 本配置 |
|---|---|---|---|
| button | position_range | [[0.4, -0.3, 0.83], [0.85, 0.3, 0.83]] | [[0.7, -0.1, 0.83], [0.85, 0.1, 0.83]] |
| button | rotation_range | - | - |

## 采集

```bash
python -m scripts.run_env \
    --gym_config configs/click_bell/coverage_strat_button21/gym_config.json \
    --action_config configs/click_bell/action_config.json \
    --num_envs 1 --max_episodes 55 --headless --report_task_success
```

输出数据集: `lerobot_dataset/coverage/click_bell/coverage_strat_button21`

由 scripts/build_coverage_configs.py 生成于 2026-08-27,分析来源 report/coverage/click_bell/coverage_summary.json。
