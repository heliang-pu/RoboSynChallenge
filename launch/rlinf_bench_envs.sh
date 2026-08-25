#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# 测环境并行吞吐,用来定 examples/rlinf/*.yaml 里的 total_num_envs。
#
#   bash launch/rlinf_bench_envs.sh                          # mixer_operating,默认档位
#   bash launch/rlinf_bench_envs.sh click_bell 1,8,16,32,64
#
# 必须在 RLinf 的 venv 里跑(要 import rlinf.envs)。环境变量与 rlinf_train.sh 同源。
# ----------------------------------------------------------------------------
set -euo pipefail

TASK="${1:-mixer_operating}"
ENVS="${2:-1,4,8,16,32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBOSYN_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROBOSYN_PATH/.." && pwd)"

RLINF_ROOT="${RLINF_ROOT:-$HOME/workspace/RLinf}"

if [[ -z "${EMBODICHAIN_ROOT:-}" ]]; then
    for cand in "$WORKSPACE_ROOT/EmbodiChain" "$HOME/workspace/EmbodiChain"; do
        [[ -d "$cand/embodichain" ]] && { EMBODICHAIN_ROOT="$cand"; break; }
    done
fi

RLINF_VENV_PYTHON="${RLINF_VENV_PYTHON:-$RLINF_ROOT/.venv/bin/python}"
[[ -x "$RLINF_VENV_PYTHON" ]] || {
    echo "错误: 找不到 RLinf 的解释器: $RLINF_VENV_PYTHON" >&2
    echo "先装 RLinf,或用 RLINF_VENV_PYTHON= 指定。" >&2
    exit 1
}

export PYTHONPATH="$ROBOSYN_PATH:$RLINF_ROOT:$EMBODICHAIN_ROOT:${PYTHONPATH:-}"
export EMBODICHAIN_PATH="$EMBODICHAIN_ROOT"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$RLINF_VENV_PYTHON" "$ROBOSYN_PATH/scripts/bench_env_throughput.py" \
    --task "$TASK" --envs "$ENVS" "${@:3}"
