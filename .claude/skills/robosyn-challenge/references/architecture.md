# 架构：任务 → 配置 → 环境，以及策略适配契约

## 任务 → 配置 → 环境

`robosynchallenge/tasks/<task>/` 定义环境与专家策略；
`configs/<task>/<setting>/{gym_config.json,action_config.json}` 定义场景/相机/随机化与
专家轨迹参数。`action_config.json` 通常在任务根目录，找不到才回落到 setting 子目录
（`find_action_config`）。

`robosynchallenge/managers/{actions,datasets,events,observations}.py` 通过把模块名 append 进
`gym_utils.DEFAULT_MANAGER_MODULES` 注入 EmbodiChain 的 functor 注册表——这就是 gym_config
里自定义 functor 名字能被解析的原因。**新增 manager 必须同步这个列表**
（`scripts/eval_policy_parallel.py` 顶部有一份副本）。

语言指令不走 yml：`eval_policy.py` 从 gym_config 递归提取 instruction 写到
`env._current_instruction`，适配器首次调用时读它并 `model.set_language(...)`。

## 策略适配契约

`scripts/eval_policy_parallel.py` 用 `importlib.import_module(f"policy.{name}")` 动态加载，
要求包顶层导出：

```python
get_model(usr_args) -> model
eval(env, model, obs) -> (obs, info, truncated, inference_times)   # 旧的 3 元组也兼容
reset_model(model) -> None
```

通常靠 `policy/<name>/__init__.py` 里 `from .deploy_policy import *`。接新策略抄
`policy/Your_Policy/`。

**适配器自己驱动环境**，不是「返回动作让上层执行」：`eval()` 内部把整个 action chunk 逐个
`env.step()`，每步检查 `is_task_success()` 与 `truncated.any()` 提前 break，返回最后一次的
四元组。这决定了「一次推理执行几步」是策略侧的自由度，**也是接新策略最容易写错的地方**。

## 配置与 overrides

`deploy_policy.yml` 是唯一配置源。命令行 `--overrides` 后面是 `--key value` 成对的
REMAINDER，值会被 `eval()` 尝试解析（所以 `true` / `10` / `[1,2]` 都能写，字符串保持原样）。
`policy_name` / `task_name` / `setting` 必填。

## 官方入口 vs 功能版

**官方入口一律原样**（2026-09-02 起）：`scripts/eval_policy.py`、`scripts/run_env.py`、
`launch/run_task.sh` 与 `origin/main` 逐字节一致。我们的功能版另起文件：

| 功能版 | 增加的能力 |
|---|---|
| `scripts/eval_policy_parallel.py` | `--num_shards/--shard_index`、`--num_envs`、`rollout_save`、指标写盘挪到 `env.close()` 之前、多卡 JAX 限卡 |
| `scripts/run_env_seeded.py` | `--seed` 走 SeededCollection |
| `launch/run_task_seeded.sh` | 上者的包装 |

指标定义（成功率、平均步数、推理耗时）两套文件完全相同，只是执行布局不同。

## `env.close()` 会终止进程

本机 DexSim 栈上 `env.close()` 直接把进程带走（exit 0），CPU/CUDA 都复现。
官方原版把 close 放在 `finally`，其后的 summary 打印与 `evaluation_metrics.json` 写盘
**从未执行**——这是「`eval_result/` 里从来没有指标文件」的根因。功能版已把 close 挪到
指标落盘之后。上游后来也加了 `finally` + `flush_cleanup_queue` 与
`EMBODICHAIN_SIM_EXIT_PROCESS` 开关。

## 跨环境调用策略的两种手法（都是有意设计，不要"顺手修好"成正常安装）

1. **把策略源码目录拼进 `PYTHONPATH` 而不是安装它**（见 `policy/pi05/eval.sh` 与
   `scripts/eval_policy_parallel.py:add_policy_dependency_paths`），在仿真环境里直接调
   训练侧代码。适用于纯 Python、能跑在仿真侧 Python 3.11 上的策略。
2. **跨进程**：Python 版本本身冲突时（`policy/pi05_lerobot` 要 3.12），策略跑在自己的
   解释器里，与评估脚本用 stdio JSON 通信（`smolvla_worker.py`、`pi05_worker.py`）。

## 路径基准

仓库里不写机器绝对路径：`REPO_ROOT` 由 `BASH_SOURCE`/`__file__` 上溯，
`WS_ROOT=dirname $REPO_ROOT`，`EMBODICHAIN_ROOT=$WS_ROOT/EmbodiChain`、
`MODELS_ROOT=$WS_ROOT/models`，均可用环境变量覆盖。YAML 里的路径一律**相对仓库根**，
所以对应命令必须从仓库根执行。EmbodiChain 必须与本仓库**同级 clone**。
