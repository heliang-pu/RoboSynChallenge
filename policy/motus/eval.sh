#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <ckpt_path> <model_name> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random "$MODELS_ROOT"/motus/Motus_robotwin2 motus 0 --max_episodes 20
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

POLICY_NAME=motus

TASK_NAME="${1}"
SETTING="${2}"
CKPT_PATH="${3}"
MODEL_NAME="${4}"
GPU_ID="${5}"

shift 5 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Motus loads ~16GB of weights plus a transient ~9.4GB T5 encoder; keep the
# allocator from fragmenting across those two phases.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Motus' utils/common.py imports deepspeed, whose import runs a
# `$CUDA_HOME/bin/nvcc -V` compatibility probe. setup_env.sh drops a
# version-only shim in the venv when the box has no CUDA toolkit.
# (motus_model._ensure_cuda_home() does the same thing defensively, so this is
# belt-and-braces for anything that bypasses the adapter.)
if [[ -z "${CUDA_HOME:-}" && -x "$VENV_DIR/.cuda-shim/bin/nvcc" ]]; then
    export CUDA_HOME="$VENV_DIR/.cuda-shim"
fi

echo "========================================="
echo "  Motus Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CKPT_PATH"
echo "  Model:      $MODEL_NAME"
echo "  GPU:        $GPU_ID"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    echo "Run: bash policy/motus/setup_env.sh" >&2
    exit 1
fi

export MOTUS_VENV_DIR="$VENV_DIR"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
cd "$REPO_ROOT" # move to RoboSynChallenge root

# Unbuffered stdout is not cosmetic here: EmbodiChain's teardown (env.close(),
# Vulkan/OptiX renderer release) can take the process down hard, and anything
# still sitting in the stdio buffer — including a traceback and the per-episode
# SUCCESS/FAIL line — is lost with it. That is what made two earlier smoke runs
# look like silent 0-step exits.
PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --model_name "$MODEL_NAME" \
    --ckpt_path "$CKPT_PATH" \
    "${EXTRA_ARGS[@]}"
