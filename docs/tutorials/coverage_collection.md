# 覆盖补采管线(带种子的官方口径合成,人手可复跑)

目标:对照官方 `random` 随机化范围审计现有专家数据的覆盖缺口,定向补采"官方判定成功"的净数据,
全程每集带种子(可复现场景、续采不重复),交付 NAS `Syn/<task>_coverage/<组名>/`(v2.1 + 边车)。

## 一轮流程

```bash
# 1 覆盖审计(读 NAS 专家数据的位姿列;无位姿列任务用 --stratify-only 全范围分层)
python scripts/analyze_random_coverage.py --task sample_loading --pose-cols cube_pose,rack_pose \
    --out report/coverage/sample_loading/
# 2 生成补采配置(configs/<task>/coverage_*/,含中文 README;汇总进 report/coverage/PLAN.json)
python scripts/build_coverage_configs.py --task sample_loading
# 3 并行采集(N worker;隔离-验证-晋升;断点续采;组级 SEED_MASTER=组名哈希)
bash launch/collect_coverage_queue.sh report/coverage/PLAN.json sample_loading 3
# 4 交付守护(转 v2.1 → pi05 训练环境读取门 → 推 NAS Syn/;可常驻)
bash launch/deliver_coverage.sh loop
```

## 关键语义

- **种子**:`run_env.py --seed <master>` 每集派生种子注入 `env.reset(seed=…)` 并同步 numpy/torch;
  边车 `episode_success.json` 记录 master_seed、每集 seed、官方判定成败、配置 sha1、commit。
  种子复现的是**初始场景**;GPU 物理接触结果不保证逐位一致。续采换 master(或 SEED_MASTER+尝试号)即不重复。
- **官方口径**:判定 = 任务类官方 `is_task_success()`(`tasks/` 与 origin/main 逐字节一致)。
  `--save_only_success` 只入库判定成功的集;`--success_settle_steps 75` 自适应静置(每 5 步查判定、成功即停)——
  官方判定要求连续 8 步稳定,专家脚本在松爪瞬间结束,不静置官方专家会被自己判 0%(实测 77 连败)。
- **产率预期**:官方口径下专家净成功率可能远低于直觉(sample_loading ≈10–25%,大头是管子斜靠孔沿 18°),
  ETA 以各任务实测为准。
- **落盘**:采集写本机 `lerobot_dataset/coverage/<task>/<组名>/`(不要把 save_path 指到 NAS,CIFS 直写录制不可靠),
  交付守护负责搬运。`.validated`(过四道门)与 `.delivered`(已上 NAS)两个标记控制断点续作。

## 跨机(pro6000 等)

仿真栈跑在 conda `robosyn` 环境(dexsim-engine 不在公网 PyPI,uv 装不出来):
`pack-conda-envs` 打包 → 目标机 `tar -xzf` 到 `~/miniconda3/envs/robosyn` → `conda-unpack`;
EmbodiChain 用 rsync 副本走 PYTHONPATH(本地对官方版有 4 处崩溃/纹理池修复,见 git diff,行为等价单环境)。
torch 需带目标卡架构(Blackwell = cu128 起)。NAS 按机器各自挂载(pro6000 用 /mnt/FermiBotNas)。
