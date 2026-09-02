#!/usr/bin/env bash
# =============================================================================
# sim-RECAP Phase C —— 价值函数训练(包装 Evo-RL 的 lerobot-value-train)
# =============================================================================
#
# 在打好 episode_success 标签的 LeRobot v3.0 数据集上训练 pistar06 价值函数
# (SigLIP 视觉 + LLM 文本骨干,分布式 bin 输出,目标 = 归一化的负剩余步数,
# 失败额外罚 c_fail;详见 Evo-RL README 第 4 节)。
#
# 用法:
#   bash launch/run_value_train.sh <dataset_dir> <run_name> [gpu_id] [extra...]
#
#   dataset_dir : LeRobot v3.0 数据集目录。rollout 数据必须先跑
#                 scripts/label_rollout_dataset.py 写入 episode_success 列;
#                 纯专家数据可不打标(用下面的 default_success 兜底)。
#   run_name    : 输出目录名(outputs/value_train/<run_name>)
#   gpu_id      : 默认 0
#   extra       : 透传给 lerobot-value-train,常用:
#                   --steps 8000                     训练步数(默认 8000)
#                   --batch_size 32                  显存不够时调小
#                   --targets.default_success success  无标签数据集整体视为成功
#                   --wandb.enable true              开 wandb(默认关)
#
# 环境:
#   默认用 conda env `evo-rl` 的 python,并把 EVO_RL_ROOT/src 顶到 PYTHONPATH
#   前面(不改动环境里已有的可编辑安装)。可用环境变量覆盖:
#     EVO_RL_ROOT   Evo-RL 路径(默认用仓库内收编的 third_party/evo_rl)
#     EVO_RL_PYTHON 解释器路径
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


DATASET_DIR="${1:?用法: run_value_train.sh <dataset_dir> <run_name> [gpu_id] [extra...]}"
RUN_NAME="${2:?缺少 run_name}"
GPU_ID="${3:-0}"
shift 3 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

# 默认用仓库内收编的 Evo-RL(third_party/evo_rl),可用环境变量指向外部 checkout
EVO_RL_ROOT="${EVO_RL_ROOT:-$REPO_ROOT/third_party/evo_rl}"
# 解释器优先级: 收编目录的 .venv > conda evo-rl 环境
if [[ -z "${EVO_RL_PYTHON:-}" ]]; then
    if [[ -x "$EVO_RL_ROOT/.venv/bin/python" ]]; then
        EVO_RL_PYTHON="$EVO_RL_ROOT/.venv/bin/python"
    else
        EVO_RL_PYTHON="$HOME/miniconda3/envs/evo-rl/bin/python"
    fi
fi

[[ -d "$EVO_RL_ROOT/src/lerobot" ]] || {
    echo "错误: 找不到 Evo-RL 源码 $EVO_RL_ROOT(可用 EVO_RL_ROOT 覆盖)" >&2; exit 1; }
[[ -x "$EVO_RL_PYTHON" ]] || {
    echo "错误: 找不到解释器 $EVO_RL_PYTHON(可用 EVO_RL_PYTHON 覆盖," >&2
    echo "      环境搭建: conda create -n evo-rl python=3.10 && pip install -e $EVO_RL_ROOT)" >&2; exit 1; }

DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
REPO_ID="local/$(basename "$DATASET_DIR")"
OUTPUT_DIR="$REPO_ROOT/outputs/value_train/$RUN_NAME"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Evo-RL 源码优先于环境内的可编辑安装
export PYTHONPATH="$EVO_RL_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "==============================================="
echo " 价值函数训练 (pistar06)"
echo " 数据集 : $DATASET_DIR"
echo " 输出   : $OUTPUT_DIR"
echo " GPU    : $GPU_ID"
echo " Evo-RL : $EVO_RL_ROOT"
echo "==============================================="

exec "$EVO_RL_PYTHON" -m lerobot.scripts.lerobot_value_train \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_DIR" \
    --value.type=pistar06 \
    --value.dtype=bfloat16 \
    --batch_size=64 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$RUN_NAME" \
    --wandb.enable=false \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
