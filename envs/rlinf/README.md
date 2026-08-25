# RLinf 后训练环境（uv，锁定版本）

PPO/GRPO 后训练用的 Python 环境。**和 `policy/pi05/` 的 openpi(JAX) 环境是两套**，
互不兼容（这套是 torch 2.7.1+cu126 + rlinf-openpi，那套是 JAX 0.5.3 + openpi 原版）。

| 文件 | 用途 |
|---|---|
| `requirements.lock.txt` | 本机跑通冒烟 PPO 时的 `uv pip freeze`，444 个精确 pin，**全部来自 PyPI** |
| `requirements.dexforce.txt` | DexForce 私有源独有的包（只有 `dexsim-engine`），单独 `--no-deps` 装 |
| `install_from_lock.sh` | 一键复现：venv → lock → dexsim → EmbodiChain editable → 补丁 → RLinf 挂钩 → 校验 |

为什么拆两份：让 uv 对全部包都去查私有源（`--index-strategy unsafe-best-match`）时，
私有源一超时整个安装就卡死 —— 实测它经常超时。主 lock 只走 PyPI，私有源只碰它独有的那一个包。

## 复现

```bash
git clone https://github.com/RLinf/RLinf ~/workspace/RLinf
# EmbodiChain 工作副本放 ~/workspace/EmbodiChain(需 >=0.2.4 布局,含 embodichain_tasks/)
bash envs/rlinf/install_from_lock.sh
bash launch/rlinf_train.sh ppo --dry-run
```

需要能访问 DexForce 私有源 `http://pyp.open3dv.site:2345/simple/`（`dexsim-engine` 等只在那里）。

## 为什么不直接用 RLinf 的 install.sh

能用，本机就是那么装的——但它是"解析一次装什么算什么"：先按 torch 2.11 解析 env 侧，
再装 `rlinf-openpi` 时被降到 torch 2.7.1、lerobot 0.4.4→0.3.3，连带换掉 12 个包；
`--model openpi --env embodichain` 这个组合官方根本不支持，模型侧要手工补。
lock 记录的是这一串副作用之后**真正能跑**的终态，按它装省掉整个过程，也保证 H100 机器
和本机版本一致。

## 为什么 `--no-deps`

这个环境的包元数据自相矛盾：`cmeel-boost==1.90.0` 声明 `numpy>=2.0`，而 `rlinf-openpi`
把 numpy 压到 1.26.4。运行时没事（冒烟 PPO 就是这个状态跑通的），但 uv 一解析就
`No solution found`。freeze 是完整闭包，按 pin 直接装、跳过解析，才是"复现"的正确语义。
代价是 lock 里少一个包不会被自动补上 —— 所以别手工删行，要改就重新 freeze。

## lock 里刻意没有的

- `embodichain` / `embodichain-tasks`：本机是 editable 指向工作副本，脚本第 3 步单独装。
  不要装私有源的 wheel 版——任务代码是针对工作副本写的，且需要打并行补丁。
- `flash-attn`：源码编译要 nvcc 且很慢，本机跳过了。没有它只是注意力慢些；H100 上
  值得装（`uv pip install flash-attn --no-build-isolation`）。

## 版本要点

- `torch==2.7.1`（PyPI 默认 +cu126 轮子）、`ray==2.58.0`、`rlinf-openpi==0.1.1`、
  `rlinf-transformer-openpi==4.53.2`（RLinf 预打好 openpi 补丁的 transformers，
  所以**不需要**手工 `cp transformers_replace/*`）
- `warp-lang==1.14.0`：官方装的是 1.13.0，本机升到与 pi05 环境一致的 1.14.0
- `lerobot==0.3.3`：被 rlinf-openpi 降级的结果。EmbodiChain 的 LeRobot 录制器要 0.4.x，
  所以接入层在建环境时剥掉了 `dataset` 段（RL rollout 本来也不该录数据集）
- Python 3.11.14
