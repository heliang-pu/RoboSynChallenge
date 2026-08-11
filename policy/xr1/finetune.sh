#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# XR-1 后训练
#
#   bash policy/xr1/finetune.sh <training_data_dir> [exp_name] [gpu_ids] [hydra overrides...]
#   bash policy/xr1/finetune.sh policy/xr1/training_data/click_bell click_bell 0 \
#        trainer.max_steps=20000
#
# <training_data_dir> 是 convert_lerobot_to_xr1.py 的输出目录（含 xr1_data.yaml）。
#
# 注意:
#   * 上游 README 让你跑 scripts/train.sh，但发布的仓库里没有这个文件，
#     所以这里直接调 tools/train.py（hydra 入口）。
#   * 训练必须有 flash-attn：CustomCollate 会把多条样本拼进一条序列，
#     靠 cu_seq_lens_* 做变长注意力隔离，只有 flash_attention_2 认这套参数。
#     用 sdpa 不会报错但样本之间会互相看见，等于训了个错的东西。
#   * XR-1 源码里把 "Qwen/Qwen3-VL-4B-Instruct" 写死成 HF hub id，
#     本脚本会在 HF 缓存里造一份指向本地目录的软链，让离线机器也能解析。
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
XR1_PKG_DIR="$SCRIPT_DIR/Xiaomi-Robotics-1/xr1"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

DATA_DIR="${1:?用法: finetune.sh <training_data_dir> [exp_name] [gpu_ids] [hydra overrides...]}"
EXP_NAME="${2:-posttrain}"
GPU_IDS="${3:-0}"
shift 3 2>/dev/null || shift $#
EXTRA_OVERRIDES=("$@")

DATA_DIR="$(cd "$DATA_DIR" && pwd)"

# 可通过环境变量覆盖
PRETRAINED="${PRETRAINED:-/home/phl/workspace/models/xr1/Xiaomi-Robotics-1-5B/model_states.pt}"
BACKBONE="${BACKBONE:-/home/phl/workspace/models/backbones/Qwen3-VL-4B-Instruct}"
PROJECT="${PROJECT:-robosynchallenge-xr1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/checkpoints}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:-20000}"
# 每 2000 步存一次，防中断（上游默认 10000，崩一次要丢 5+ 小时）
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
# FrozenVLMRunner = 冻结 VLM 只训 DiT+projector（单卡唯一可行方案）
# BaseRunner      = 上游默认的全参微调，需要多卡/多机
RUNNER="${RUNNER:-FrozenVLMRunner}"
# FusedAdam 要 JIT 编译 CUDA 算子；编译环境有问题时可换 torch.optim.AdamW（不用编译）
OPTIMIZER="${OPTIMIZER:-deepspeed.ops.adam.FusedAdam}"

# ---------------------------------------------------------------- 前置检查

[[ -x "$VENV_PYTHON" ]] || { echo "Error: 没有 venv，先跑 bash policy/xr1/setup_env.sh" >&2; exit 1; }
[[ -f "$DATA_DIR/xr1_data.yaml" ]] || {
    echo "Error: $DATA_DIR 下没有 xr1_data.yaml" >&2
    echo "       先跑 convert_lerobot_to_xr1.py 生成训练数据" >&2
    exit 1
}
[[ -f "$PRETRAINED" ]] || { echo "Error: 找不到预训练权重 $PRETRAINED" >&2; exit 1; }
[[ -d "$BACKBONE" ]] || { echo "Error: 找不到 backbone 目录 $BACKBONE" >&2; exit 1; }

if ! "$VENV_PYTHON" -c "import flash_attn" 2>/dev/null; then
    echo "Error: 没装 flash-attn，无法训练。" >&2
    echo "       XR-1 的 collate 把多条样本打包进一条序列，靠 flash-attn 的变长" >&2
    echo "       注意力(cu_seq_lens)做隔离；换 sdpa 会静默串味。" >&2
    echo "       请先跑: bash policy/xr1/setup_env.sh --with-flash-attn" >&2
    exit 1
fi

# deepspeed 在 import 期就要探测 CUDA op 兼容性，没有 CUDA_HOME 会直接抛
# MissingCUDAException（不是缺包）。这台机器没装系统级 CUDA，去 conda env 里找。
detect_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then echo "$CUDA_HOME"; return; fi
    local candidate
    for candidate in /usr/local/cuda /usr/local/cuda-*; do
        [[ -x "$candidate/bin/nvcc" ]] && { echo "$candidate"; return; }
    done
    for candidate in "$HOME"/miniconda3/envs/*/ "$HOME"/anaconda3/envs/*/; do
        [[ -x "${candidate}bin/nvcc" ]] && { echo "${candidate%/}"; return; }
    done
}
DETECTED_CUDA_HOME="$(detect_cuda_home)"
if [[ -n "$DETECTED_CUDA_HOME" ]]; then
    export CUDA_HOME="$DETECTED_CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
else
    echo "Error: 找不到 CUDA toolkit (nvcc)，deepspeed 无法 import" >&2
    exit 1
fi

# CUDA 12.1 的 nvcc 只认 gcc<=12，而 Ubuntu 24.04 默认 gcc 13，
# DeepSpeed JIT 编译 fused_adam 会报 "unsupported GNU version"。
# 指定 host 编译器即可，不必降级系统 gcc。
if [[ -z "${CUDAHOSTCXX:-}" && -x /usr/bin/g++-12 ]]; then
    export CUDAHOSTCXX=/usr/bin/g++-12
    export CC=/usr/bin/gcc-12
    export CXX=/usr/bin/g++-12
fi

if ! "$VENV_PYTHON" -c "import deepspeed" 2>/dev/null; then
    echo "Error: 没装 deepspeed（trainer 用的是 deepspeed 策略）。" >&2
    echo "       需要 CUDA toolkit，确认 nvcc 可用后重跑 setup_env.sh" >&2
    exit 1
fi

# ------------------------------------------------- 让写死的 HF hub id 能离线解析

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_home}"
CACHE_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct"
SNAPSHOT_DIR="$CACHE_DIR/snapshots/local"
if [[ ! -d "$SNAPSHOT_DIR" ]]; then
    echo ">>> 建立 HF 缓存软链: $SNAPSHOT_DIR -> $BACKBONE"
    mkdir -p "$SNAPSHOT_DIR" "$CACHE_DIR/refs"
    echo -n "local" > "$CACHE_DIR/refs/main"
    for file in "$BACKBONE"/*; do
        [[ -f "$file" ]] && ln -sf "$file" "$SNAPSHOT_DIR/$(basename "$file")"
    done
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# --------------------------------------------------------- 装配 hydra data 配置

CONFIG_DIR="$SCRIPT_DIR/configs"
DATASET_NAME="$(basename "$DATA_DIR")"
mkdir -p "$CONFIG_DIR/data"
cp "$DATA_DIR/xr1_data.yaml" "$CONFIG_DIR/data/$DATASET_NAME.yaml"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export MAX_LENGTH
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT}"

mkdir -p "$OUTPUT_ROOT"

echo "=========================================="
echo "  XR-1 后训练"
echo "  数据集    : $DATA_DIR  ($DATASET_NAME)"
echo "  预训练权重: $PRETRAINED"
echo "  backbone  : $BACKBONE"
echo "  输出      : $OUTPUT_ROOT/project_$PROJECT/$EXP_NAME"
echo "  GPU       : $GPU_IDS"
echo "  MAX_LENGTH: $MAX_LENGTH   MAX_STEPS: $MAX_STEPS   WANDB_MODE: $WANDB_MODE"
echo "  runner    : $RUNNER"
echo "  CUDA_HOME : $CUDA_HOME   CUDAHOSTCXX: ${CUDAHOSTCXX:-<default>}"
echo "  optimizer : $OPTIMIZER   save_interval: $SAVE_INTERVAL"
echo "=========================================="

cd "$XR1_PKG_DIR"  # process_save_cfg 会往 ./assets 写一份 config.py，必须在包目录下跑
mkdir -p assets

# 用本适配层的入口而不是 tools/train.py：它额外注册了 FrozenVLMRunner。
# 单卡 4090 放不下 5.5B 全参微调（Adam 状态就要 66G），必须冻结 VLM
# 只训 DiT+projector；理由和实测数字见 train_xr1.py 顶部注释。
exec "$VENV_PYTHON" -u "$SCRIPT_DIR/train_xr1.py" \
    hydra.searchpath="[file://$CONFIG_DIR]" \
    data="$DATASET_NAME" \
    model=posttrain \
    model.type="$RUNNER" \
    model.params.pretrained="$PRETRAINED" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.save_interval="$SAVE_INTERVAL" \
    trainer.optimizer.type="$OPTIMIZER" \
    trainer.project="$PROJECT" \
    trainer.exp_name="$EXP_NAME" \
    trainer.default_root_dir="$OUTPUT_ROOT" \
    "${EXTRA_OVERRIDES[@]}"
