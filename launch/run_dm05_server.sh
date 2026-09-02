#!/bin/bash

# Start the OpenDM DM0.5 inference HTTP service used by policy/dm05.
#
# Usage: ./run_dm05_server.sh <checkpoint_dir> [extra_args...]
# Examples:
#   ./run_dm05_server.sh policy/dm05/opendm/checkpoints/DM05
#   ./run_dm05_server.sh /path/to/my_sft_checkpoint --exp playground/dm05_sft_demo.py
#   ./run_dm05_server.sh policy/dm05/opendm/checkpoints/DM05 --port 7892 --backend fast
#
# 推理服务需要在装有 opendm 依赖的环境里运行(conda activate <opendm_env>),
# 与仿真评估环境相互独立,见 policy/dm05/README.md。

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
OPENDM_ROOT="$REPO_ROOT/policy/dm05/opendm"

if [[ "$#" -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo -e "\n\033[1;33mUsage:\033[0m"
    echo -e "  $0 \033[1;32m<checkpoint_dir>\033[0m \033[1;35m[extra_args...]\033[0m\n"
    echo -e "\033[1;33mAvailable Extra Arguments:\033[0m"
    echo -e "  \033[1;35m--exp <entry.py>\033[0m       : Playground entry point (default: opendm/exp/dm05_exp.py)"
    echo -e "  \033[1;35m--chunk-size <n>\033[0m       : Action horizon, must match training (default: 50)"
    echo -e "  \033[1;35m--action-dim <n>\033[0m       : Output action dimension (default: 14)"
    echo -e "  \033[1;35m--port <n>\033[0m             : HTTP port (default: 7891)"
    echo -e "  \033[1;35m--backend <default|fast>\033[0m : Inference backend (fast requires TensorRT/Triton)"
    echo -e "  \033[1;35m--dataset-name <name>\033[0m  : data-config.dataset-name for SFT entry points\n"
    echo "Download the base checkpoint first, e.g.:"
    echo "  hf download Dexmal/DM05 --local-dir $OPENDM_ROOT/checkpoints/DM05"
    exit 0
fi

CHECKPOINT=$1
shift

EXP="opendm/exp/dm05_exp.py"
CHUNK_SIZE=50
ACTION_DIM=14
PORT=7891
BACKEND="default"
DATASET_NAME=""
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp) EXP="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --action-dim) ACTION_DIM="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --dataset-name) DATASET_NAME="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Resolve checkpoint relative to the challenge repo root if not absolute.
if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="$REPO_ROOT/$CHECKPOINT"
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo -e "\033[1;31mError: checkpoint directory not found: $CHECKPOINT\033[0m"
    echo "Download it first, e.g.:"
    echo "  hf download Dexmal/DM05 --local-dir $OPENDM_ROOT/checkpoints/DM05"
    exit 1
fi

if [ ! -d "$OPENDM_ROOT" ]; then
    echo -e "\033[1;31mError: opendm repo not found at $OPENDM_ROOT\033[0m"
    echo "Clone it with:"
    echo "  git clone https://github.com/dexmal/opendm.git $OPENDM_ROOT"
    exit 1
fi

CMD=(
    bash script/dm05_launcher.sh
    --exp "$EXP"
    --task inference
    --model-config.model-name-or-path "$CHECKPOINT"
    --model-config.chunk-size "$CHUNK_SIZE"
    --inference-config.output-action-dim "$ACTION_DIM"
    --inference-config.image-prompts "Head" "Left wrist" "Right wrist"
    --inference-config.port "$PORT"
)

if [ "$BACKEND" != "default" ]; then
    CMD+=(--inference-config.backend "$BACKEND")
fi
if [ -n "$DATASET_NAME" ]; then
    CMD+=(--data-config.dataset-name "$DATASET_NAME")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "========================================="
echo "Starting DM05 inference service"
echo "Checkpoint : $CHECKPOINT"
echo "Entry point: $EXP"
echo "Chunk size : $CHUNK_SIZE | Action dim: $ACTION_DIM | Port: $PORT | Backend: $BACKEND"
echo "========================================="
echo "Running command (cwd: $OPENDM_ROOT):"
echo "${CMD[@]}"
echo "========================================="

cd "$OPENDM_ROOT"
exec "${CMD[@]}"
