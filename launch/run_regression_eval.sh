#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# 本机稳定版评测:临时换用 merge 前的 eval 代码跑,退出时自动还原。
#
#   bash launch/run_regression_eval.sh <task> <train_config> <model> <ckpt_id> [episodes]
#   bash launch/run_regression_eval.sh mixer_operating pi05_mixer_operating mixer_operating 28001 20
#
# 为什么存在:上游 d3161d4..84b6c0e 合并带来的 eval 改动,在"策略+仿真同一进程"
# (torch CUDA + JAX + Vulkan 混跑)的形态下,会让 dexsim 引擎在累计 ~1000 步后
# 确定性崩溃(无 traceback、退出码 0,总死在第 3 集附近)。二分结论:
#   - 任务代码、EmbodiChain 修复、每步 is_task_success、cuda.synchronize 逐一排除
#   - merge 前 eval 代码 + 相同其他条件 = 6/6 稳定
#   - 纯仿真进程同等调用强度 1500 步无恙 => 是混合运行时互操作问题,不是泄漏
# 触发点藏在上游 eval_policy 的 +224 行里,未继续细分(legacy 单进程路径,
# 不在 PPO 关键路径上;PPO 的 env worker 是纯仿真进程,已单独验证稳定)。
#
# 上游文件保持原样不动;本脚本用 git show 取 merge 前版本临时覆盖,trap 保证还原。
# ----------------------------------------------------------------------------
set -euo pipefail

TASK="${1:?用法: $0 <task> <train_config> <model> <ckpt_id> [episodes]}"
TRAIN_CONFIG="${2:?}"
MODEL="${3:?}"
CKPT="${4:?}"
EPISODES="${5:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

# merge 前基线 = 合并提交 cc9aef5 的第一父(本地 main 分叉点)
PRE_MERGE_REF="${PRE_MERGE_REF:-cc9aef5^}"

git diff --quiet policy/pi05/deploy_policy.py scripts/eval_policy.py \
    || { echo "错误: deploy_policy.py / eval_policy.py 有未提交改动,先处理掉再跑(脚本要临时覆盖它们)。" >&2; exit 1; }

restore() { git checkout -- policy/pi05/deploy_policy.py scripts/eval_policy.py 2>/dev/null || true; }
trap restore EXIT

git show "$PRE_MERGE_REF:policy/pi05/deploy_policy.py" > policy/pi05/deploy_policy.py
git show "$PRE_MERGE_REF:scripts/eval_policy.py" > scripts/eval_policy.py
echo "已临时切换到 merge 前 eval 代码($PRE_MERGE_REF),结束后自动还原"

WORKSPACE_ROOT="$(cd "$REPO/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-}"
if [[ -z "$EMBODICHAIN_ROOT" ]]; then
    for cand in "$WORKSPACE_ROOT/EmbodiChain" "$HOME/workspace/EmbodiChain"; do
        [[ -d "$cand/embodichain" ]] && { EMBODICHAIN_ROOT="$cand"; break; }
    done
fi
PI05_PY="${PI05_PY:-$HOME/workspace/RoboSynChallenge/policy/pi05/.venv/bin/python}"

PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.03 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$REPO:$EMBODICHAIN_ROOT:$REPO/policy" \
"$PI05_PY" scripts/eval_policy.py \
    --config policy/pi05/deploy_policy.yml --overrides \
    --task_name "$TASK" --setting random --model_name "$MODEL" \
    --train_config_name "$TRAIN_CONFIG" --checkpoint_id "$CKPT" \
    --pytorch_device cuda --max_episodes "$EPISODES" --headless True --eval_video_log False
