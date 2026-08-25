#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# 测环境并行吞吐,用来定 examples/rlinf/*.yaml 里的 total_num_envs。
#
#   bash launch/rlinf_bench_envs.sh                          # mixer_operating,默认档位
#   bash launch/rlinf_bench_envs.sh click_bell 1,8,16,32,64
#
# 每个档位起一个独立进程 —— dexsim 引擎在 env.close() 时会终止整个进程(没有
# traceback、退出码 0),同一进程里连测多档,第一次关闭之后就什么都不会发生了。
#
# 必须在 RLinf 的 venv 里跑(要 import rlinf.envs)。环境变量与 rlinf_train.sh 同源。
# ----------------------------------------------------------------------------
set -uo pipefail

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
    echo "先跑 launch/rlinf_setup_env.sh,或用 RLINF_VENV_PYTHON= 指定。" >&2
    exit 1
}

export PYTHONPATH="$ROBOSYN_PATH:$RLINF_ROOT:${EMBODICHAIN_ROOT:-}:${PYTHONPATH:-}"
export EMBODICHAIN_PATH="${EMBODICHAIN_ROOT:-}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# 失败日志必须保留:引擎崩溃不产生 Python traceback,现场一丢就只能靠猜。
FAIL_DIR="${RSC_BENCH_FAIL_DIR:-/tmp/rsc_bench_failures}"
mkdir -p "$FAIL_DIR"

echo "task=$TASK  档位=$ENVS  (每档一个独立进程)"
echo
printf "%9s %14s %10s %10s %12s\n" "num_envs" "env-steps/s" "ms/step" "peak GB" "相对 1 env"
printf -- "-------------------------------------------------------\n"

BASELINE=""
IFS=',' read -ra LEVELS <<< "$ENVS"
for n in "${LEVELS[@]}"; do
    n="$(echo "$n" | tr -d '[:space:]')"
    [[ -z "$n" ]] && continue

    TMP_LOG="$(mktemp -t rsc_bench_XXXXXX.log)"
    "$RLINF_VENV_PYTHON" "$ROBOSYN_PATH/scripts/bench_env_throughput.py" \
        --task "$TASK" --num-envs "$n" "${@:3}" > "$TMP_LOG" 2>&1

    line="$(grep -a -m1 '^BENCH_RESULT' "$TMP_LOG" || true)"
    if [[ -z "$line" ]]; then
        # 没拿到结果:要么 OOM,要么引擎在建环境阶段就退了。把最后一条像样的错误摘出来。
        reason="$(grep -a -oiE '(CUDA out of memory|RuntimeError:.*|AttributeError:.*|ValueError:.*|KeyError:.*)' "$TMP_LOG" | tail -1)"
        kept="$FAIL_DIR/${TASK}_n${n}_$(date +%H%M%S).log"
        mv "$TMP_LOG" "$kept"
        printf "%9s  失败: %s\n" "$n" "${reason:-无结果;完整日志已保留: $kept}"
        continue
    fi

    fps="$(sed -E 's/.*fps=([0-9.]+).*/\1/' <<< "$line")"
    ms="$(sed -E 's/.*ms_per_step=([0-9.]+).*/\1/' <<< "$line")"
    gb="$(sed -E 's/.*peak_gb=([0-9.]+).*/\1/' <<< "$line")"
    rm -f "$TMP_LOG"
    [[ -z "$BASELINE" ]] && BASELINE="$fps"
    speedup="$(awk -v a="$fps" -v b="$BASELINE" 'BEGIN{printf "%.1f", (b>0? a/b : 0)}')"
    printf "%9s %14s %10s %10s %11sx\n" "$n" "$fps" "$ms" "$gb" "$speedup"
done

echo
echo "选档参考:取显存仍有余量(actor 和 rollout 也要占卡)且加速比还接近线性的那一档。"
echo "加速比明显走平说明渲染已经成为瓶颈,再加环境只会拖长每步耗时。"
