# Vendored: Evo-RL

sim-RECAP 价值函数工具链的来源,收编进本仓库以便离线复现,不依赖外部 checkout。

| 项 | 值 |
|---|---|
| 上游仓库 | https://github.com/MINT-SJTU/Evo-RL (SJTU MINT & Evo-Tech) |
| 收编 commit | `3fcb87be97ab6c3e8e796d95c7f73ee3bc32bc49` |
| 收编日期 | 2026-08-24 |
| 许可证 | Apache-2.0(见 LICENSE) |

## 本仓库用到的部分

- `src/lerobot/values/pistar06/` —— π\*0.6 风格分布式价值函数(SigLIP + LLM 骨干,bin 分布输出)
- `src/lerobot/scripts/lerobot_value_train.py` / `lerobot_value_infer.py` —— 价值训练与 advantage 写回
- `src/lerobot/rl/` —— ACP 标签(`Advantage: positive/negative`)与训练 hook
- 其余 `src/lerobot/*` 是上游 LeRobot 0.4.4 fork 的支撑代码,原样保留

入口脚本见本仓库 `launch/run_value_train.sh` 与 `launch/run_value_infer.sh`。

## 相对上游的删减(体积原因,不影响价值工具链)

- `src/lerobot/assets/`(35M,PiPER URDF/网格,仅真机 teleop 需要;留了空目录占位,
  需要真机功能时从上游 `git lfs pull` 补回)
- `tests/`、`docs/`、`website/`、`examples/`、数据与输出目录

## 环境搭建

```bash
cd third_party/evo_rl
uv venv --python 3.10
uv pip install -e .
```

也兼容已有的 conda `evo-rl` 环境:launch 包装脚本会把本目录的 `src/`
顶到 `PYTHONPATH` 前面,保证跑的是这份收编代码。

## 更新方式

从上游拉新版本时,重复 rsync(维持上述删减),并更新本文件的 commit 记录。
不要在本目录内做本地修改;需要定制的逻辑放在本仓库的 scripts/ 与 launch/ 层。
