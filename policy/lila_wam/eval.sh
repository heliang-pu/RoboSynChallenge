#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <checkpoint> <model_name> [gpu_id] [extra_opts...]
#
#   checkpoint: 训练产出的 run 目录(自动挑最大 epoch),或某个 checkpoint_epoch_N.pt
#               config.yaml / norm_stats.json 从同目录读
#
#   bash policy/lila_wam/eval.sh click_bell random \
#        policy/lila_wam/checkpoints/sft_2026-08-29_12-00-00 lila_wam 0 --max_episodes 20
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
# 评测跑在**仓库根 venv**里,因为 EmbodiChain / dexsim 装在那儿(见 README
# 的"仿真/采集/评估环境")。policy/lila_wam/.venv 只是训练环境,没有仿真栈。
# 根 venv 只需额外补两个包:
#     uv pip install --python .venv/bin/python "transformers>=4.56,<5" omegaconf
if [[ -z "${PYTHON_BIN:-}" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif [[ -z "${PYTHON_BIN:-}" && -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

POLICY_NAME=lila_wam

TASK_NAME="${1}"
SETTING="${2}"
CHECKPOINT="${3}"
MODEL_NAME="${4}"
GPU_ID="${5:-0}"

shift 5 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "========================================="
echo "  LiLa-WAM Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Model:      $MODEL_NAME"
echo "  GPU:        $GPU_ID"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    echo "Run: bash policy/lila_wam/setup_env.sh --with-sim" >&2
    exit 1
fi

# EmbodiChain 放在**最后**:根 venv 里已经装了 embodichain,把源码 checkout 排在
# 仓库前面会让 import 解析到源码而不是装好的包,与 dexsim 二进制对不上,
# 在建环境阶段直接段错误(exit 139)。
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$PYTHONPATH:$EMBODICHAIN_ROOT"
fi
cd "$REPO_ROOT" # move to RoboSynChallenge root

PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --model_name "$MODEL_NAME" \
    --checkpoint_path "$CHECKPOINT" \
    "${EXTRA_ARGS[@]}"
