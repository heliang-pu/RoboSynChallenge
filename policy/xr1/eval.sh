#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <train_config> <model_name> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random xr1_click_bell posttrain 0 --max_episodes 50
#
# 只想跑零样本预训练权重（语义不对，仅验证链路）:
#   bash eval.sh click_bell random none none 0 --max_joint_delta 0.2
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

POLICY_NAME=xr1

TASK_NAME="${1}"
SETTING="${2}"
TRAIN_CONFIG="${3}"
MODEL_NAME="${4}"
GPU_ID="${5}"

shift 5 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"
# XR-1 走 HF 本地目录，禁止联网探测，避免离线机上卡住
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

echo "========================================="
echo "  XR-1 Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Config:     $TRAIN_CONFIG / $MODEL_NAME"
echo "  GPU:        $GPU_ID"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    echo "       先跑 bash policy/xr1/setup_env.sh" >&2
    exit 1
fi

export XR1_VENV_DIR="$VENV_DIR"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$SCRIPT_DIR:$SCRIPT_DIR/Xiaomi-Robotics-1/xr1${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
cd "$REPO_ROOT" # move to RoboSynChallenge root

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --model_name "$MODEL_NAME" \
    --train_config_name "$TRAIN_CONFIG" \
    "${EXTRA_ARGS[@]}"
