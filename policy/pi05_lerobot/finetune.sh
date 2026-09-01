#!/bin/bash
# ----------------------------------------------------------------------------
# bash finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]
# bash finetune.sh click_bell lerobot_dataset/cobotmagic_Sim_click_bell \
#   outputs/train/pi05_mem_click_bell 3 --steps=30000
#
# Visual MEM (short-horizon observation memory) is on by default — it is the
# reason this adapter tracks upstream main. Proprioceptive MEM is opt-in via
# PI05_USE_PROPRIOCEPTIVE_MEMORY=1. PI05_USE_VISUAL_MEMORY=0 gives a stock π₀.₅ run.
#
# Set PI05_NOHUP=1 to launch in the background and write a log under
# $LEROBOT_ROOT/outputs/train/logs by default.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TASK_NAME="${1:?Usage: bash policy/pi05_lerobot/finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]}"
DATASET_ROOT="${2:?Usage: bash policy/pi05_lerobot/finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]}"
OUTPUT_DIR="${3:?Usage: bash policy/pi05_lerobot/finetune.sh <task_name> <dataset_root> <output_dir> [gpu_id] [extra_opts...]}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

CONDA_ROOT="${CONDA_ROOT:-}"
PI05_CONDA_ENV="${PI05_CONDA_ENV:-lerobot}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${PI05_LEROBOT_ROOT:-}}"
DATASET_REPO_ID="${PI05_DATASET_REPO_ID:-RoboSynChallenge/cobotmagic_Sim_${TASK_NAME}}"
JOB_NAME="${PI05_JOB_NAME:-$(basename "$OUTPUT_DIR")}"
BASE_MODEL="${PI05_BASE_MODEL:-lerobot/pi05_base}"
BATCH_SIZE="${PI05_BATCH_SIZE:-32}"
TRAIN_STEPS="${PI05_STEPS:-30000}"
# Upstream's recommended single-GPU settings. MEM pushes 6 frames per camera
# through SigLIP, so the memory headroom matters more here than for stock π₀.₅.
DTYPE="${PI05_DTYPE:-bfloat16}"
GRADIENT_CHECKPOINTING="${PI05_GRADIENT_CHECKPOINTING:-true}"

USE_VISUAL_MEMORY="${PI05_USE_VISUAL_MEMORY:-true}"
# Off by default: visual MEM adds no parameters (it reuses SigLIP's pretrained
# q/k/v), while proprioceptive MEM both drops the discretized state out of the
# prompt and introduces `proprio_history_proj`, which lerobot/pi05_base has no
# weights for. On a single-task dataset that is a much bigger departure from the
# pretrained model, so opt in deliberately.
USE_PROPRIOCEPTIVE_MEMORY="${PI05_USE_PROPRIOCEPTIVE_MEMORY:-false}"
MEMORY_FRAMES="${PI05_MEMORY_FRAMES:-6}"

normalize_bool() {
    case "${1,,}" in
        1|true|yes|y|on) echo true ;;
        *) echo false ;;
    esac
}
USE_VISUAL_MEMORY="$(normalize_bool "$USE_VISUAL_MEMORY")"
USE_PROPRIOCEPTIVE_MEMORY="$(normalize_bool "$USE_PROPRIOCEPTIVE_MEMORY")"
GRADIENT_CHECKPOINTING="$(normalize_bool "$GRADIENT_CHECKPOINTING")"

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Error: dataset root does not exist: $DATASET_ROOT" >&2
    exit 1
fi

INFO_JSON="$DATASET_ROOT/meta/info.json"
STATS_JSON="$DATASET_ROOT/meta/stats.json"

# LeRobot >= 0.6 only reads v3.0 datasets. This repo keeps both: the collection
# pipeline writes v3.0 and converts down to v2.1 for openpi, so point this script
# at the v3.0 copy rather than the one policy/pi05 uses.
CODEBASE_VERSION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version',''))" "$INFO_JSON" 2>/dev/null || echo "")"
if [[ "$CODEBASE_VERSION" != "v3.0" ]]; then
    echo "Error: $DATASET_ROOT is codebase_version='${CODEBASE_VERSION:-unknown}'; LeRobot pi0.5 needs v3.0." >&2
    echo "Use the v3.0 copy of this dataset, or convert it:" >&2
    echo "  python -m lerobot.scripts.convert_dataset_v21_to_v30 \\" >&2
    echo "      --repo-id $DATASET_REPO_ID --root $DATASET_ROOT --push-to-hub false" >&2
    exit 1
fi

# MEM was pre-trained on observations one second apart, and `memory_stride` is
# counted in dataset frames, so the stride has to follow the dataset's fps —
# 25 here, not the upstream default of 30.
DATASET_FPS="$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1]))['fps']))" "$INFO_JSON" 2>/dev/null || echo "")"
if [[ -z "$DATASET_FPS" ]]; then
    echo "Error: cannot read fps from $INFO_JSON" >&2
    exit 1
fi
MEMORY_STRIDE="${PI05_MEMORY_STRIDE:-$DATASET_FPS}"

# π₀.₅ normalizes state and actions with QUANTILES, which needs q01/q99 in the
# dataset stats. Older datasets only carry min/max/mean/std.
HAS_QUANTILES="$(python3 -c "
import json, sys
try:
    stats = json.load(open(sys.argv[1]))
except Exception:
    print('no'); raise SystemExit
keys = stats.get('action', {})
print('yes' if 'q01' in keys and 'q99' in keys else 'no')
" "$STATS_JSON" 2>/dev/null || echo "no")"

RENAME_MAP="${PI05_RENAME_MAP:-}"
if [[ -z "$RENAME_MAP" ]]; then
    # Raw RoboSynChallenge exports name the state observation.qpos and the
    # cameras <cam>.color; converted v2.1 datasets already use LeRobot names.
    NEEDS_RENAME="$(python3 -c "
import json, sys
features = json.load(open(sys.argv[1])).get('features', {})
print('yes' if 'observation.qpos' in features else 'no')
" "$INFO_JSON" 2>/dev/null || echo "no")"
    if [[ "$NEEDS_RENAME" == "yes" ]]; then
        RENAME_MAP='{"observation.qpos":"observation.state","cam_high.color":"observation.images.cam_high","cam_left_wrist.color":"observation.images.cam_left_wrist","cam_right_wrist.color":"observation.images.cam_right_wrist"}'
    fi
fi

if [[ "${PI05_USE_CONDA:-auto}" != "0" && -n "$CONDA_ROOT" && -f "$CONDA_ROOT/bin/activate" ]]; then
    source "$CONDA_ROOT/bin/activate"
    if conda env list | awk '{print $1}' | grep -qx "$PI05_CONDA_ENV"; then
        conda activate "$PI05_CONDA_ENV"
    else
        echo "Warning: conda env '$PI05_CONDA_ENV' not found; using current Python environment." >&2
    fi
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [[ -z "$LEROBOT_ROOT" && -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    LEROBOT_ROOT="$SCRIPT_DIR/lerobot"
fi

if [[ -n "$LEROBOT_ROOT" && ! -d "$LEROBOT_ROOT" ]]; then
    echo "Error: configured LeRobot root does not exist: $LEROBOT_ROOT" >&2
    exit 1
fi

if [[ -z "$LEROBOT_ROOT" && ! "$(command -v lerobot-train || true)" ]]; then
    if [[ "${PI05_AUTO_SETUP:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        bash "$SCRIPT_DIR/setup_lerobot.sh" "$SCRIPT_DIR/lerobot"
        LEROBOT_ROOT="$SCRIPT_DIR/lerobot"
    else
        echo "Error: cannot find lerobot-train and no LeRobot source is configured." >&2
        echo "Set PI05_LEROBOT_ROOT=policy/pi05_lerobot/lerobot, install lerobot in this env, or run:" >&2
        echo "  bash policy/pi05_lerobot/setup_lerobot.sh" >&2
        exit 1
    fi
fi

if [[ "$HAS_QUANTILES" != "yes" ]]; then
    if [[ "${PI05_AUTO_QUANTILE:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        echo "Augmenting $DATASET_ROOT with quantile stats..."
        python -m lerobot.scripts.augment_dataset_quantile_stats \
            --repo-id "$DATASET_REPO_ID" --root "$DATASET_ROOT"
    else
        echo "Error: $STATS_JSON has no q01/q99, which π₀.₅ QUANTILES normalization requires." >&2
        echo "Run this once, or re-run with PI05_AUTO_QUANTILE=1:" >&2
        echo "  python -m lerobot.scripts.augment_dataset_quantile_stats \\" >&2
        echo "      --repo-id $DATASET_REPO_ID --root $DATASET_ROOT" >&2
        exit 1
    fi
fi

echo "========================================="
echo "  LeRobot PI0.5 Finetune"
echo "  Task:       $TASK_NAME"
echo "  Dataset:    $DATASET_ROOT (${DATASET_FPS} fps, $CODEBASE_VERSION)"
echo "  Repo ID:    $DATASET_REPO_ID"
echo "  Output:     $OUTPUT_DIR"
echo "  GPU:        $GPU_ID"
echo "  Base model: $BASE_MODEL"
echo "  dtype:      $DTYPE (gradient_checkpointing=$GRADIENT_CHECKPOINTING)"
echo "  MEM:        visual=$USE_VISUAL_MEMORY proprio=$USE_PROPRIOCEPTIVE_MEMORY"
echo "              frames=$MEMORY_FRAMES stride=$MEMORY_STRIDE"
echo "  LeRobot:    ${LEROBOT_ROOT:-installed package}"
echo "  Conda env:  ${CONDA_DEFAULT_ENV:-current environment}"
echo "========================================="

if [[ -n "$LEROBOT_ROOT" ]]; then
    cd "$LEROBOT_ROOT"
else
    cd "$REPO_ROOT"
fi

TRAIN_CMD=(
    lerobot-train
    --policy.type=pi05
    "--policy.pretrained_path=$BASE_MODEL"
    --policy.push_to_hub=false
    --policy.device=cuda
    "--policy.dtype=$DTYPE"
    "--policy.gradient_checkpointing=$GRADIENT_CHECKPOINTING"
    "--policy.use_visual_memory=$USE_VISUAL_MEMORY"
    "--policy.use_proprioceptive_memory=$USE_PROPRIOCEPTIVE_MEMORY"
    "--policy.memory_frames=$MEMORY_FRAMES"
    "--policy.memory_stride=$MEMORY_STRIDE"
    "--batch_size=$BATCH_SIZE"
    "--steps=$TRAIN_STEPS"
    --num_workers=8
    --persistent_workers=true
    --wandb.enable=false
    "--dataset.repo_id=$DATASET_REPO_ID"
    "--dataset.root=$DATASET_ROOT"
    "--output_dir=$OUTPUT_DIR"
    "--job_name=$JOB_NAME"
)
if [[ -n "$RENAME_MAP" ]]; then
    TRAIN_CMD+=("--rename_map=$RENAME_MAP")
fi
if [[ -n "${PI05_EMA_DECAY:-}" ]]; then
    # openpi keeps an EMA copy of the weights for inference (ema_decay=0.99).
    TRAIN_CMD+=(--ema.enable=true "--ema.decay=$PI05_EMA_DECAY")
fi
TRAIN_CMD+=("${EXTRA_ARGS[@]}")

case "${PI05_NOHUP:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        LOG_DIR="${PI05_LOG_DIR:-${LEROBOT_ROOT:-$REPO_ROOT}/outputs/train/logs}"
        mkdir -p "$LOG_DIR"
        LOG_PATH="$LOG_DIR/$JOB_NAME.log"
        nohup "${TRAIN_CMD[@]}" > "$LOG_PATH" 2>&1 &
        echo "Started LeRobot PI0.5 finetune in background."
        echo "  PID: $!"
        echo "  Log: $LOG_PATH"
        ;;
    *)
        "${TRAIN_CMD[@]}"
        ;;
esac
