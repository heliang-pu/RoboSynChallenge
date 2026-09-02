#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random outputs/train/pi05_mem_click_bell/checkpoints/last 5 \
#   --pytorch_device cuda --headless true --eval_video_log true
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="${ROBOSYN_VENV_DIR:-$EMBODICHAIN_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

POLICY_NAME=pi05_lerobot

TASK_NAME="${1:?Usage: bash policy/pi05_lerobot/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
SETTING="${2:?Usage: bash policy/pi05_lerobot/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
CHECKPOINT_PATH="${3:?Usage: bash policy/pi05_lerobot/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

# eval_policy.py runs each override value through eval(), so shell-style
# booleans have to be spelled the Python way.
for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

if [[ -n "${PI05_LEROBOT_PYTHON:-}" ]]; then
    EXTRA_ARGS+=(--pi05_lerobot_python "$PI05_LEROBOT_PYTHON")
fi
if [[ -n "${PI05_LEROBOT_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--lerobot_root "$PI05_LEROBOT_ROOT")
elif [[ -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    EXTRA_ARGS+=(--lerobot_root "$SCRIPT_DIR/lerobot")
fi

if [[ -d "$CHECKPOINT_PATH/pretrained_model" ]]; then
    CHECKPOINT_DIR="$CHECKPOINT_PATH/pretrained_model"
else
    CHECKPOINT_DIR="$CHECKPOINT_PATH"
fi

# DexSim uses physical GPU ordinals from --gpu_id. Keep the simulator parent
# unmasked, then mask only the PI0.5 worker via --pi05_lerobot_cuda_visible_devices.
unset CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  LeRobot PI0.5 Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  GPU:        $GPU_ID"
echo "  Python:     $PYTHON_BIN"
if [[ -n "${PI05_LEROBOT_PYTHON:-}" ]]; then
    echo "  Worker Py:  $PI05_LEROBOT_PYTHON"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    echo "  Worker Py:  $SCRIPT_DIR/.venv/bin/python (uv)"
else
    echo "  Worker Py:  current eval Python"
fi
if [[ -n "${PI05_LEROBOT_ROOT:-}" ]]; then
    echo "  LeRobot:    $PI05_LEROBOT_ROOT"
elif [[ -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    echo "  LeRobot:    $SCRIPT_DIR/lerobot"
else
    echo "  LeRobot:    installed package in worker Python"
fi
echo "========================================="

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: cannot find executable Python: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_name "$(basename "$(dirname "$CHECKPOINT_DIR")")" \
    --gpu_id "$GPU_ID" \
    --pi05_lerobot_cuda_visible_devices "$GPU_ID" \
    "${EXTRA_ARGS[@]}"
