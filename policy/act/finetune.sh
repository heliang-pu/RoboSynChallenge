#!/bin/bash
# =============================================================================
# ACT (Action Chunking Transformer) 训练脚本
# =============================================================================
#
# 模型简介:
#   ACT 是轻量级模仿学习策略:ResNet 视觉编码器 + Transformer + CVAE,
#   从零开始训练(无预训练基座),单张消费级显卡即可,适合快速出 baseline。
#
# 用法:
#   bash finetune.sh <dataset_root> <output_dir> [gpu_id] [extra_opts...]
#
#   dataset_root : LeRobot v2.1 格式数据集根目录(含 meta/ data/ videos/)
#   output_dir   : 训练产物输出目录(checkpoint、日志)
#   gpu_id       : GPU 编号,默认 0
#   extra_opts   : 透传给 scripts/train.py 的参数,常用的有:
#                    --steps 100000          训练步数(默认 100k)
#                    --batch-size 8          批大小(默认 8)
#                    --chunk-size 16         动作块长度(默认 16)
#                    --n-action-steps 8      每次执行的动作步数(默认 8)
#                    --save-freq 20000       存档间隔(默认 20k 步)
#                    --use-amp               混合精度训练(省显存、提速)
#                    --resume                从 output_dir 断点续训
#                    --overwrite             覆盖已存在的 output_dir
#                    --wandb                 开启 wandb 记录(默认关闭)
#                    --seed 1000             随机种子
#
# 示例:
#   # 用 click_bell 数据训 10 万步
#   bash finetune.sh training_data/RoboSynChallenge/cobotmagic_Sim_click_bell \
#        outputs/train/act_click_bell 0
#   # 短跑冒烟测试
#   bash finetune.sh /path/to/dataset outputs/train/act_smoke 0 --steps 1000
#
# 输出:
#   <output_dir>/checkpoints/<step>/pretrained_model/   —— 可直接用于 eval.sh 部署
#
# 多卡:
#   本脚本是单进程入口;DDP 多卡请直接用 torchrun 调 scripts/train.py
#   并加 --distributed(详见 ACT 文档)。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------- 参数解析 ----------------
DATASET_ROOT="${1:?用法: bash finetune.sh <dataset_root> <output_dir> [gpu_id] [extra_opts...]}"
OUTPUT_DIR="${2:?缺少 output_dir 参数}"
GPU_ID="${3:-0}"
shift 3 2>/dev/null || shift $#          # 其余参数原样透传给 train.py
EXTRA_ARGS=("$@")

# ---------------- 环境变量 ----------------
# CUDA_VISIBLE_DEVICES 限定本进程可见的 GPU(单进程训练只用一张卡)
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# HF_LEROBOT_HOME 是 LeRobot 解析本地数据集与缓存元数据的根目录
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$SCRIPT_DIR/training_data}"

# ---------------- 前置检查 ----------------
if [[ ! -d "$DATASET_ROOT/meta" ]]; then
    echo "[错误] $DATASET_ROOT 不是 LeRobot 数据集(缺少 meta/ 目录)" >&2
    exit 1
fi

echo "========================================="
echo "  ACT Policy 训练"
echo "  数据集 : $DATASET_ROOT"
echo "  输出   : $OUTPUT_DIR"
echo "  GPU    : $GPU_ID"
echo "  额外参数: ${EXTRA_ARGS[*]:-<无>}"
echo "========================================="

cd "$SCRIPT_DIR"

# train.py 会:
#   1. 加载数据集元信息,自动推断 obs/action 维度和相机键
#   2. 构建 ACT 模型(CVAE + Transformer)从零训练
#   3. 按 --save-freq 间隔把 checkpoint 存到 <output_dir>/checkpoints/
uv run --frozen python scripts/train.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
