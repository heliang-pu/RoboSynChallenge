#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# 用 RLinf 对 pi0.5 做 PPO / GRPO 后训练。
#
#   bash launch/rlinf_train.sh ppo          # PPO
#   bash launch/rlinf_train.sh grpo         # GRPO
#   bash launch/rlinf_train.sh ppo --dry-run  # 只做检查和链接,不启动训练
#
# 这个脚本存在的理由:配置文件版本化在本仓库(examples/rlinf/),但 hydra 只从 RLinf
# 的 examples/embodiment/config/ 找配置。这里负责把它们链过去,并设好一串环境变量——
# 少设任何一个都会在跑起来几分钟后才以难懂的方式失败。
#
# 覆盖点(都可以从外面传进来):
#   RLINF_ROOT              RLinf 仓库,默认 ~/workspace/RLinf
#   EMBODICHAIN_ROOT        EmbodiChain 仓库,默认 ../EmbodiChain
#   ROBOSYN_PI05_TORCH_CKPT 转换后的 PyTorch checkpoint 目录(必须)
#   RLINF_PYTHON            解释器,默认 RLinf venv 里的 python
# ----------------------------------------------------------------------------
set -euo pipefail

MODE="${1:-ppo}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBOSYN_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROBOSYN_PATH/.." && pwd)"

RLINF_ROOT="${RLINF_ROOT:-$HOME/workspace/RLinf}"

# 很多机器上只有 python3 没有 python;前置检查用得到解释器,先解析出来。
# 训练本身用的是 RLinf venv 里的 python(由 run_embodiment.sh 决定),与此无关。
if [[ -z "${RLINF_PYTHON:-}" ]]; then
    if command -v python >/dev/null 2>&1; then RLINF_PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then RLINF_PYTHON=python3
    else echo "错误: 找不到 python 或 python3" >&2; exit 1
    fi
fi

# EmbodiChain 的位置不能假定是本仓库的同级目录 —— 在 git worktree 里
# ($ROBOSYN_PATH/.. 是 .claude/worktrees/)这个假设就不成立。按可靠性排序回退。
if [[ -z "${EMBODICHAIN_ROOT:-}" ]]; then
    for cand in \
        "$WORKSPACE_ROOT/EmbodiChain" \
        "$HOME/workspace/EmbodiChain" \
        "$($RLINF_PYTHON -c 'import embodichain,pathlib;print(pathlib.Path(embodichain.__file__).resolve().parent.parent)' 2>/dev/null || true)"
    do
        if [[ -n "$cand" && -d "$cand/embodichain" ]]; then
            EMBODICHAIN_ROOT="$cand"
            break
        fi
    done
fi
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-}"

case "$MODE" in
    ppo)   CONFIG_NAME="robosynchallenge_ppo_pi05" ;;
    grpo)  CONFIG_NAME="robosynchallenge_grpo_pi05" ;;
    smoke) CONFIG_NAME="robosynchallenge_ppo_pi05_smoke" ;;   # 单卡冒烟,验管线不训模型
    probe) CONFIG_NAME="robosynchallenge_ppo_pi05_probe"      # 单卡真训练(慢),任务由 ROBOSYN_TASK 选
           [[ -n "${ROBOSYN_TASK:-}" ]] || { echo "probe 模式需要 ROBOSYN_TASK=<task>,如 ROBOSYN_TASK=sample_loading" >&2; exit 1; }
           export ROBOSYN_TASK ;;
    *)     echo "用法: $0 {ppo|grpo|smoke|probe} [额外参数...]" >&2; exit 1 ;;
esac

DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then DRY_RUN=1; else EXTRA_ARGS+=("$arg"); fi
done

# --- 前置检查:每一项失败都会让训练在几分钟后才炸,不如现在就说清楚 -------------

die() { echo "错误: $*" >&2; exit 1; }

[[ -d "$RLINF_ROOT" ]] || die "找不到 RLinf: $RLINF_ROOT(用 RLINF_ROOT= 指定)"
[[ -d "$EMBODICHAIN_ROOT" ]] || die "找不到 EmbodiChain: $EMBODICHAIN_ROOT(用 EMBODICHAIN_ROOT= 指定)"

if [[ -z "${ROBOSYN_PI05_TORCH_CKPT:-}" ]]; then
    die "必须设 ROBOSYN_PI05_TORCH_CKPT —— 转换后的 PyTorch checkpoint 目录。
先跑: python scripts/convert_pi05_jax_to_torch.py --checkpoint-dir <JAX 的 <step> 目录> \\
        --config-name pi05_base_robosynchallenge_full --output-path <输出目录>"
fi
[[ -f "$ROBOSYN_PI05_TORCH_CKPT/model.safetensors" ]] \
    || die "$ROBOSYN_PI05_TORCH_CKPT 下没有 model.safetensors —— 这不是转换产物目录"
[[ -d "$ROBOSYN_PI05_TORCH_CKPT/assets" ]] \
    || die "$ROBOSYN_PI05_TORCH_CKPT 下没有 assets/ —— 缺 norm_stats,策略输出会完全错。
RLinf 的转换器从 checkpoint_dir.parent/assets 找,对不上 openpi 的 <step>/assets 布局;
scripts/convert_pi05_jax_to_torch.py 会补拷,直接用 RLinf 的脚本则不会。"

# RLinf 必须打过补丁,否则 env_type: robosynchallenge 解析不出来
"$RLINF_PYTHON" "$ROBOSYN_PATH/scripts/patch_rlinf_env.py" --rlinf-root "$RLINF_ROOT" --check >/dev/null 2>&1 \
    || die "RLinf 还没打补丁。先跑:
    python scripts/patch_rlinf_env.py --rlinf-root $RLINF_ROOT"

# --- 把配置链进 RLinf 的 config 目录 ------------------------------------------

RLINF_CONFIG_DIR="$RLINF_ROOT/examples/embodiment/config"
[[ -d "$RLINF_CONFIG_DIR" ]] || die "找不到 RLinf 的配置目录: $RLINF_CONFIG_DIR"

for src in "$ROBOSYN_PATH"/examples/rlinf/*.yaml; do
    ln -sfn "$src" "$RLINF_CONFIG_DIR/$(basename "$src")"
done
echo "已链接配置 -> $RLINF_CONFIG_DIR"

# --- 环境变量 ----------------------------------------------------------------

# robosynchallenge 要能在 Ray 起的每个 worker 进程里 import(task 靠 @register_env
# 在 import 时注册,而本仓库没声明 embodichain.tasks entry point)。
export PYTHONPATH="$ROBOSYN_PATH:$EMBODICHAIN_ROOT:${PYTHONPATH:-}"
export EMBODICHAIN_PATH="$EMBODICHAIN_ROOT"

# run_embodiment.sh 要求给这个位置参数,但对 openpi 路径它其实不起作用:
# 全仓只有 rlinf/models/embodiment/starvla/utils/action_space.py 读它
# (RLinf 文档说它决定动作维度和归一化,代码里不成立)。填 ALOHA 是因为
# CobotMagic 与 ALOHA 同为双臂 14 维,将来换 StarVLA 时这个值才会真正生效。
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}"

# 离屏渲染
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "======================================================"
echo "  模式        : $MODE ($CONFIG_NAME)"
echo "  RoboSyn     : $ROBOSYN_PATH"
echo "  RLinf       : $RLINF_ROOT"
echo "  EmbodiChain : $EMBODICHAIN_ROOT"
echo "  checkpoint  : $ROBOSYN_PI05_TORCH_CKPT"
echo "======================================================"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "--dry-run:检查全部通过,未启动训练。"
    exit 0
fi

cd "$RLINF_ROOT"
# run_embodiment.sh 用裸 `python` 起训练,依赖 venv 已激活;不激活会拿到系统 python(没有 ray)
if [[ -f "$RLINF_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$RLINF_ROOT/.venv/bin/activate"
fi
exec bash examples/embodiment/run_embodiment.sh "$CONFIG_NAME" "$ROBOT_PLATFORM" "${EXTRA_ARGS[@]}"
