#!/usr/bin/env bash
# =============================================================================
# pi0.5 单任务微调 —— 从 all10 co-train checkpoint 67500 起训
#
# 用法: ./train_ft67500.sh <task> <gpu_ids> [exp_name] [extra args...]
#   task     : click_bell / drawer_open_place / handle_basket / item_assembly /
#              items_handover / manipulate_pipette / mixer_operating /
#              sample_loading / table_rearrangement / water_pouring
#   gpu_ids  : 逗号分隔,如 "0" 或 "0,1"
#   exp_name : 实验名,默认 ft67500
#
# 基座   : /data/workspace/models/cotrain/pi05_all10_h64_67500/params  (H64)
# 归一化 : 复用基座 all10_expert_h64 norm_stats(不重算)
# 输出   : /data/train_out/pi05_<task>_ft67500/<exp_name>/
# =============================================================================
set -euo pipefail

TASK="${1:?用法: ./train_ft67500.sh <task> <gpu_ids> [exp_name] [extra...]}"
GPU_IDS="${2:?必须指定 GPU,如 0 或 0,1}"
EXP_NAME="${3:-ft67500}"
shift 3 2>/dev/null || shift $# 2>/dev/null || true
EXTRA_ARGS=("$@")

CONFIG_NAME="pi05_${TASK}_ft67500"
REPO_ID="RoboSynChallenge/cobotmagic_Sim_${TASK}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PI05_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_LEROBOT_HOME="/data/lerobot_home"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online
export WANDB_ENTITY=puheliang
export WANDB_DIR="/data/train_out/wandb"
mkdir -p /data/train_out/wandb
export PATH="/root/.local/bin:/usr/local/cuda/bin:$PATH"

NGPU=$(awk -F, '{print NF}' <<< "$GPU_IDS")
echo "==============================================="
echo " 任务     : $TASK"
echo " 配置     : $CONFIG_NAME"
echo " 数据集   : $HF_LEROBOT_HOME/$REPO_ID"
echo " GPU      : $CUDA_VISIBLE_DEVICES  ($NGPU 张)"
echo " 输出     : /data/train_out/$CONFIG_NAME/$EXP_NAME/"
echo "==============================================="

[[ -d "$HF_LEROBOT_HOME/$REPO_ID" ]] || { echo "[错误] 数据集不存在: $HF_LEROBOT_HOME/$REPO_ID"; exit 1; }
[[ -d /data/workspace/models/cotrain/pi05_all10_h64_67500/params ]] || { echo "[错误] 基座权重不存在"; exit 1; }

MODE_FLAG="--overwrite"
for a in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do [[ "$a" == "--resume" ]] && MODE_FLAG=""; done

exec uv run --frozen scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" $MODE_FLAG ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
