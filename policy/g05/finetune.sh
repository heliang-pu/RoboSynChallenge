#!/bin/bash
# ----------------------------------------------------------------------------
# G0.5 在 RoboSynChallenge 数据上微调
#
#   bash policy/g05/finetune.sh <num_gpus> [task_name] [hydra_overrides...]
#   bash policy/g05/finetune.sh 8 cobotmagic
#   bash policy/g05/finetune.sh 8 cobotmagic model.batch_size=8 model.max_epochs=5
#
# 前置:
#   1. bash policy/g05/setup_env.sh                        建环境
#   2. 权重放到 GalaxeaVLA/checkpoints（见 README_INTEGRATION.md）
#   3. python policy/g05/convert_lerobot_to_g05.py scan lerobot_dataset --emit-config
#      生成 policy/g05/configs/{data,task}/cobotmagic.yaml
#
# !! 显存要求 !!
#   G0.5 是 2B VLM + action expert，官方微调按 8 卡跑，单卡需要 >70GB。
#   本机 RTX 4090 48GB 单卡跑不动这个配置，必须换 A100/H100 80G 或 96G 机器。
#   想在小卡上冒烟验证流程，用 --test（截断数据 + 离线日志）并把 batch_size 调到 1。
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
G05_ROOT="$SCRIPT_DIR/GalaxeaVLA"
VENV_DIR="$G05_ROOT/.venv"

if [[ $# -lt 1 ]]; then
    echo "用法: bash policy/g05/finetune.sh <num_gpus> [task_name] [hydra_overrides...]" >&2
    exit 1
fi

NUM_GPUS="$1"
shift
TASK_NAME="${1:-cobotmagic}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

if [[ ! -d "$G05_ROOT" ]]; then
    echo "Error: 找不到 GalaxeaVLA: $G05_ROOT" >&2
    exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Error: 环境未就绪，先跑 bash policy/g05/setup_env.sh" >&2
    exit 1
fi

# --- 把我们自己写的 task/data 配置装进 GalaxeaVLA 的 configs 树 -----------------
# hydra 只认 configs/task/*.yaml 和 configs/data/*.yaml，但我们不想把自研文件
# 混进第三方源码目录，所以放在 policy/g05/configs 下，运行前软链过去。
OUR_CONFIGS="$SCRIPT_DIR/configs"
for group in data task; do
    src="$OUR_CONFIGS/$group/$TASK_NAME.yaml"
    dst="$G05_ROOT/configs/$group/$TASK_NAME.yaml"
    if [[ ! -f "$src" ]]; then
        echo "Error: 缺少配置 $src" >&2
        echo "先跑: python policy/g05/convert_lerobot_to_g05.py scan lerobot_dataset --emit-config" >&2
        exit 1
    fi
    if [[ -e "$dst" && ! -L "$dst" ]]; then
        echo "Error: $dst 已存在且不是软链，拒绝覆盖官方文件" >&2
        exit 1
    fi
    ln -sfn "$src" "$dst"
    echo "[config] $dst -> $src"
done

# --- 代理会拖慢/卡住 HF 与 wandb，与 setup_env.sh 保持一致 -----------------------
unset http_proxy https_proxy all_proxy ftp_proxy 2>/dev/null || true
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY 2>/dev/null || true

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$G05_ROOT:$G05_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 训练要用 torchcodec 解 mp4，它依赖 libnppicc.so.12（nvidia-npp-cu12 提供）。
# 该目录不在动态库搜索路径上，不加这行读视频会 RuntimeError。
NPP_LIB_DIR="$VENV_DIR/lib/python3.10/site-packages/nvidia/npp/lib"
if [[ -d "$NPP_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$NPP_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    echo "[warn] 找不到 $NPP_LIB_DIR，torchcodec 可能无法解码视频；" >&2
    echo "       重跑 bash policy/g05/setup_env.sh 补装 nvidia-npp-cu12。" >&2
fi

# configs/train.yaml 的 hydra.run.dir 是
#   ${oc.env:G05_OUTPUT_DIR}/${task}/${exp_name}
# G05_OUTPUT_DIR **没有默认值**，不导出会在 hydra 解析阶段直接 KeyError。
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-$SCRIPT_DIR/outputs}"
export EXP_NAME="${EXP_NAME:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$G05_OUTPUT_DIR"
RUN_DIR="$G05_OUTPUT_DIR/$TASK_NAME/$EXP_NAME"

echo "========================================="
echo "  G0.5 Finetune"
echo "  Task:    $TASK_NAME"
echo "  GPUs:    $NUM_GPUS"
echo "  Python:  $VENV_DIR/bin/python"
echo "  产物目录: $RUN_DIR"
echo "========================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo "提示: 官方 README 要求全量微调 >70GB 显存（A100 80G / H20 96G）。"
echo "      48GB 的 4090 跑不动全量微调；推理/评估则官方推荐 4090（>8GB）。"
echo
echo "训练完成后按下面这样部署（注意 checkpoint 文件名是 step_<N>.pt）:"
echo "  bash policy/g05/eval.sh <task> <setting> $RUN_DIR/checkpoints/step_<N>.pt 0"
echo "  同时把 deploy_policy.yml 的 sim_task / embodiment 都改成 $TASK_NAME"
echo

# logger 默认走离线：本机没登录 wandb，online 模式会卡在联网重试上。
# 放在 "$@" 前面，用户显式传 logger.mode=online 时以用户的为准（hydra 取最后一个）。
cd "$G05_ROOT"
exec "$VENV_DIR/bin/python" -m torch.distributed.run \
    --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" \
    scripts/finetune.py "task=$TASK_NAME" "logger.mode=offline" "${EXTRA_ARGS[@]}"
