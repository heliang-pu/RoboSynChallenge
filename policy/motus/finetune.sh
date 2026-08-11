#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Motus Stage-3 SFT on RoboSynChallenge data.
#
#   bash policy/motus/finetune.sh [nproc_per_node] [config] [zero_stage]
#   bash policy/motus/finetune.sh 8 configs/robosyn_finetune.yaml zero1
#
# !! HARDWARE: Motus training needs >80GB VRAM per GPU (A100-80G / H100 / B200).
# !! The RTX 4090 48GB in this workspace CANNOT train this model — it can only
# !! run inference. This script is written for the training box.
#
# Prerequisites:
#   1. bash policy/motus/setup_env.sh --with-train      (adds deepspeed)
#   2. python policy/motus/prepare_data.py --lerobot-root ... --output-root ...
#   3. python policy/motus/prepare_data.py --t5-only --wan-path ...  (fills umt5_wan/)
#   4. Weights present: Motus (stage 2), Wan2.2-TI2V-5B, Qwen3-VL-2B-Instruct
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTUS_ROOT="$SCRIPT_DIR/Motus"
VENV_DIR="$SCRIPT_DIR/.venv"

NPROC="${1:-8}"
CONFIG="${2:-$SCRIPT_DIR/configs/robosyn_finetune.yaml}"
ZERO_STAGE="${3:-zero1}"
RUN_NAME="${RUN_NAME:-motus-robosyn}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -d "$MOTUS_ROOT" ]]; then
    echo "Error: Motus source tree not found at $MOTUS_ROOT" >&2
    exit 1
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
if ! "$PYTHON_BIN" -c "import deepspeed" >/dev/null 2>&1; then
    echo "Error: deepspeed missing. Run: bash policy/motus/setup_env.sh --with-train" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-/home/phl/workspace/outputs/${RUN_NAME}}"
mkdir -p "$OUTPUT_DIR"

# train.py resolves `data.*`, `models.*` and `utils.*` relative to the repo root.
cd "$MOTUS_ROOT"
export PYTHONPATH="$MOTUS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

echo "========================================="
echo "  Motus SFT"
echo "  Config:  $CONFIG"
echo "  GPUs:    $NPROC"
echo "  ZeRO:    $ZERO_STAGE"
echo "  Output:  $OUTPUT_DIR"
echo "========================================="

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    train/train.py \
    --deepspeed "configs/${ZERO_STAGE}.json" \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --report_to tensorboard \
    2>&1 | tee "$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"
