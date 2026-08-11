#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> [ckpt_path] [gpu_id] [extra_opts...]
# bash eval.sh click_bell random policy/g05/GalaxeaVLA/checkpoints/g05-base/checkpoints/model_state_dict.pt 0 --max_episodes 50
#
# ckpt_path 留空则用 deploy_policy.yml 里的默认值。
# 环境由 policy/g05/setup_env.sh 建在 policy/g05/GalaxeaVLA/.venv。
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
G05_ROOT="$SCRIPT_DIR/GalaxeaVLA"
VENV_DIR="$G05_ROOT/.venv"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

POLICY_NAME=g05

TASK_NAME="${1}"
SETTING="${2}"
CKPT_PATH="${3}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"
# GalaxeaVLA 训练/推理脚本都靠这两个变量给出可读的报错
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export TOKENIZERS_PARALLELISM=false
# 权重和 processor 都在本地，别让 HF 联网卡住启动
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 不加这个的话，仿真器在进程退出阶段的清理会把 stdout 缓冲区里的
# "Episode ... SUCCESS/FAIL / success rate" 汇总行整段吞掉，
# 日志只停在 tqdm 进度条上，看起来像跑了一半就没了。
export PYTHONUNBUFFERED=1

echo "========================================="
echo "  G0.5 Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: ${CKPT_PATH:-<deploy_policy.yml default>}"
echo "  GPU:        $GPU_ID"
echo "========================================="

if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    echo "Hint: run 'bash policy/g05/setup_env.sh' first." >&2
    exit 1
fi

if [[ ! -d "$G05_ROOT" ]]; then
    echo "Error: GalaxeaVLA source not found at $G05_ROOT" >&2
    exit 1
fi

# 把我们自己写的 embodiment 配置装进 GalaxeaVLA 的 configs 树。
# hydra 只认 configs/{data,task}/*.yaml，但自研文件放在 policy/g05/configs 下不污染
# 第三方源码，运行前软链过去。拒绝覆盖非软链的官方文件。
for group in data task; do
    src_dir="$SCRIPT_DIR/configs/$group"
    [[ -d "$src_dir" ]] || continue
    for src in "$src_dir"/*.yaml; do
        [[ -e "$src" ]] || continue
        dst="$G05_ROOT/configs/$group/$(basename "$src")"
        if [[ -e "$dst" && ! -L "$dst" ]]; then
            echo "[warn] $dst 已存在且不是软链，跳过（不覆盖官方文件）" >&2
            continue
        fi
        ln -sfn "$src" "$dst"
    done
done

export G05_ROOT
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$G05_ROOT:$G05_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# 评估本身不解码视频（观测直接来自仿真器），但 dataset 相关代码可能被间接 import，
# 带上这个路径没有副作用。
NPP_LIB_DIR="$VENV_DIR/lib/python3.10/site-packages/nvidia/npp/lib"
if [[ -d "$NPP_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$NPP_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
cd "$REPO_ROOT" # move to RoboSynChallenge root

OVERRIDES=(--task_name "$TASK_NAME" --setting "$SETTING")
if [[ -n "$CKPT_PATH" ]]; then
    OVERRIDES+=(--ckpt_path "$CKPT_PATH")
fi

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    "${OVERRIDES[@]}" \
    "${EXTRA_ARGS[@]}"
