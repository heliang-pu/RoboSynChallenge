# 第三方仓库补丁

这些改动落在本仓库之外的代码里，没法随本分支提交，所以以 patch 形式留档。
**没有它们，RLinf 的并行 rollout 会崩** —— 重新 checkout / 升级对应仓库后必须重新打。

## embodichain_parallel_envs.patch

目标仓库：`~/workspace/EmbodiChain`（DexForce 的 EmbodiChain 工作副本，editable 安装进两个 venv）。

```bash
cd ~/workspace/EmbodiChain
git apply --check /path/to/RoboSynChallenge/patches/embodichain_parallel_envs.patch && \
git apply         /path/to/RoboSynChallenge/patches/embodichain_parallel_envs.patch
```

两处修复，都只在 `num_envs > 1` 时暴露：

1. `managers/events.py` `get_pose()`：`qpos[env_ids, joint_ids]` 是高级索引（两个索引张量
   逐元素配对），`num_envs=1` 时 `[1]×[6]` 碰巧广播成功，`num_envs=4` 直接 `IndexError`。
   改成先取行再取列。
2. `managers/randomization/spatial.py` + `events.py`：`compute_fk` / `compute_ik` 没透传
   `env_ids`。episode 中途只有部分环境终止时 auto-reset 只带子集，qpos 也只切了这几行，
   而 FK/IK 按全部环境校验 batch → `Joint positions batch size mismatch`。全量 reset 时
   两者相等测不出来，所以表现为"短测通过、长测必崩"。

3. `managers/events.py` `register_entity_pose()`：`torch.bmm(entity_pose, entity_extra_attr_val)`
   里额外属性位姿是物体坐标系下的固定偏移，配置只给一份 → `(1,4,4)`，而 `entity_pose` 是
   `(num_envs,4,4)`，`bmm` 报 `batch2` 尺寸不匹配。只有配置里带 `*_pose_object` 额外属性的
   任务会走到（sample_loading 有，mixer_operating 没有）。修法是 batch 为 1 时 `expand` 到
   `num_envs`。

验证方式：`bash launch/rlinf_bench_envs.sh <task> 8 --steps 120`
（随机动作让环境频繁终止，部分重置持续发生），修复后全程无崩溃。
mixer_operating 和 sample_loading 都按此验证过。

两处都值得整理成最小复现提给 DexForce；`compute_fk` 的文档本身就要求 qpos 为
`(n_envs, num_joints)`，修复后的形状才是合契约的。

## RLinf 那边不需要 patch 文件

对 RLinf 的两处挂钩（env 类注册 + openpi 配置注册，共 20 行）由
`scripts/patch_rlinf_env.py` 幂等地打 / `--check` / `--revert`，不用手工 apply。
