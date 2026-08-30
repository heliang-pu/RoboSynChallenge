#!/usr/bin/env bash
# =============================================================================
# DM0.5 (OpenDM) SFT —— 海光 DCU (腾讯云 TIONE, taco-train-dev dtk26.04 镜像) 版
# =============================================================================
#
# 与 finetune.sh 的差异(均为在 DCU 镜像上实测所需):
#   1. 不用 conda/uv:激活 /tmp/dm05/venv(--system-site-packages,只覆盖
#      transformers==5.3.0 + opendm 缺的纯 Python 包;torch 用镜像自带的
#      2.7.1+dtk ROCm 版,系统 python 不受影响)。
#   2. 三处 attention 全部改 sdpa:
#        llm    默认 flex_attention → DCU 上 triton flex 内核不可用
#        vision 默认 flash_attention_2 → 镜像的 flash_attn 是 das 魔改版,
#               transformers 5.x 的 FA2 接口对不上,sdpa 最稳
#        action 默认已是 sdpa
#   3. 关 liger_kernel(未装,且是 CUDA triton 内核)。
#   4. 关 TF32:transformers 的 NVIDIA Ampere 架构检查不识别海光 DCU;
#      训练仍使用 BF16。
#   5. FSDP 使用 shard_grad_op + BACKWARD_POST,并将 Gemma3 RMSNorm 按 batch
#      分块计算,降低 batch 64 的 backward 重计算峰值显存。
#   6. 路径全部落在 /tmp/dm05(12T 本地盘;/root 只有 200G)。
#
# 用法:
#   bash finetune_dcu.sh <dataset_name> [nproc_per_node] [output_dir] [extra_opts...]
#
#   示例(8 卡,每卡 batch 64):
#     bash finetune_dcu.sh sample_loading_relative 8 "" \
#          --trainer-config.per-device-train-batch-size 64
#
#   冒烟(必须换 norm_stats_root,否则 max-batches 截断的统计量会被正式训练复用!):
#     bash finetune_dcu.sh sample_loading_relative 8 /tmp/dm05/user_checkpoints/smoke \
#          --data-config.norm-stats-root /tmp/dm05/norm_stats_smoke \
#          --data-config.compute-norm-stats-max-batches 2 \
#          --trainer-config.num-train-steps 5 --trainer-config.save-steps 5
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENDM_ROOT="$SCRIPT_DIR/opendm"
DM05_WORK="${DM05_WORK:-/tmp/dm05}"
VENV="${DM05_VENV:-$DM05_WORK/venv}"

DATASET_NAME="${1:?用法: bash finetune_dcu.sh <dataset_name> [nproc_per_node] [output_dir] [extra_opts...]}"
NPROC="${2:-8}"
OUTPUT_DIR="${3:-}"
shift 3 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$DM05_WORK/user_checkpoints/dm05_sft_$DATASET_NAME"
BASE_MODEL="${DM05_BASE_MODEL:-$DM05_WORK/checkpoints/DM05-robotwin2}"
NORM_STATS_ROOT="${DM05_NORM_STATS_ROOT:-$DM05_WORK/norm_stats}"

# ---------------- 环境 ----------------
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "[错误] venv 不存在: $VENV (python3 -m venv --system-site-packages $VENV && pip install transformers==5.3.0 ...)" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c "import opendm" 2>/dev/null; then
    echo "[错误] venv 里没有 opendm,先: cd $OPENDM_ROOT && pip install --no-deps -e ." >&2
    exit 1
fi
if [[ ! -f "$BASE_MODEL/model.safetensors" ]]; then
    echo "[错误] 找不到 DM05 基座: $BASE_MODEL" >&2
    exit 1
fi
python - <<'EOF'
import torch, transformers
n = torch.cuda.device_count()
assert torch.cuda.is_available() and n > 0, "torch 看不到 DCU"
print(f"[OK] torch {torch.__version__} / transformers {transformers.__version__} / DCU x{n} ({torch.cuda.get_device_name(0)})")
EOF

# DCU 可见性:torch-ROCm 同时认 HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export DM05_DATA_ROOT="${DM05_DATA_ROOT:-$DM05_WORK/datasets}"
export DM05_RMSNORM_BATCH_CHUNK="${DM05_RMSNORM_BATCH_CHUNK:-4}"
export DM05_ATTN_BATCH_CHUNK="${DM05_ATTN_BATCH_CHUNK:-4}"
export DM05_MLP_BATCH_CHUNK="${DM05_MLP_BATCH_CHUNK:-4}"
export DM05_SAVED_TENSORS_CPU_OFFLOAD="${DM05_SAVED_TENSORS_CPU_OFFLOAD:-1}"

echo "==============================================="
echo " DM0.5 SFT (DCU)"
echo " 数据集     : $DATASET_NAME"
echo " 进程/DCU   : $NPROC  (HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<全部>})"
echo " 基座权重   : $BASE_MODEL"
echo " 数据根目录 : $DM05_DATA_ROOT"
echo " norm_stats : $NORM_STATS_ROOT"
echo " 输出目录   : $OUTPUT_DIR"
echo " 透传参数   : ${EXTRA_ARGS[*]:-<无>}"
echo "==============================================="

cd "$OPENDM_ROOT"
exec bash script/dm05_launcher.sh \
    --exp playground/dm05_sft_demo.py \
    --nproc_per_node "$NPROC" \
    --master_port "${MASTER_PORT:-29500}" \
    --task train \
    --model-config.model-name-or-path "$BASE_MODEL" \
    --model-config.llm-attn-implementation sdpa \
    --model-config.vision-attn-implementation sdpa \
    --model-config.action-attn-implementation sdpa \
    --model-config.no-liger-kernel \
    --trainer-config.no-tf32 \
    --trainer-config.fsdp-sharding-strategy shard_grad_op \
    --trainer-config.fsdp-backward-prefetch BACKWARD_POST \
    --data-config.dataset-name "$DATASET_NAME" \
    --data-config.norm-stats-root "$NORM_STATS_ROOT" \
    --trainer-config.output-dir "$OUTPUT_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
