#!/usr/bin/env bash
# =============================================================================
# pi0.5 单任务微调 —— 从【官方 pi0.5 base】起训,数据集 Sim_official_plus_seeded_clean_v21
# 用法: ./train_v21.sh <task> <gpu_ids> [exp_name] [extra args...]
# 输出: /data/train_out_v21/pi05_<task>_v21/<exp_name>/
# =============================================================================
set -euo pipefail
TASK="${1:?用法: ./train_v21.sh <task> <gpu_ids> [exp] [extra...]}"
GPU_IDS="${2:?必须指定 GPU}"
EXP_NAME="${3:-v21}"
shift 3 2>/dev/null || shift $# 2>/dev/null || true
EXTRA=("$@")

CONFIG="pi05_${TASK}_v21"
REPO="Sim_official_plus_seeded_clean_v21/${TASK}"
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$SD")"; cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_LEROBOT_HOME="/data"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online
export WANDB_ENTITY=puheliang
export WANDB_DIR="/data/train_out_v21/wandb"
export PATH="/root/.local/bin:/usr/local/cuda/bin:$PATH"
mkdir -p /data/train_out_v21/wandb

echo "=============================================="
echo " 任务   : $TASK"
echo " 配置   : $CONFIG"
echo " 数据集 : $HF_LEROBOT_HOME/$REPO"
echo " 基座   : /data/workspace/base_models/pi05_base_jax/params (官方)"
echo " GPU    : $CUDA_VISIBLE_DEVICES"
echo " 输出   : /data/train_out_v21/$CONFIG/$EXP_NAME/"
echo "=============================================="
[[ -d "$HF_LEROBOT_HOME/$REPO" ]] || { echo "[错误] 数据集不存在: $HF_LEROBOT_HOME/$REPO"; exit 1; }
[[ -d /data/workspace/base_models/pi05_base_jax/params ]] || { echo "[错误] 官方基座不存在"; exit 1; }

MODE="--overwrite"
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do [[ "$a" == "--resume" ]] && MODE=""; done
exec uv run --frozen scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" $MODE ${EXTRA[@]+"${EXTRA[@]}"}
