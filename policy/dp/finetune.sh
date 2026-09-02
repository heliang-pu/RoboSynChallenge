#!/bin/bash
# =============================================================================
# Diffusion Policy (DP) 训练脚本
# =============================================================================
#
# 模型简介:
#   Diffusion Policy 用条件扩散模型建模动作分布:视觉编码器提取观测特征,
#   1D U-Net 通过迭代去噪生成动作序列。从零开始训练(无预训练基座),
#   对多模态动作分布(同一场景多种合理做法)比 ACT 更鲁棒,但推理更慢
#   (每次预测要跑多步去噪)。
#
# 用法:
#   bash finetune.sh <dataset_root> <output_dir> [gpu_id] [extra_opts...]
#
#   dataset_root : LeRobot v2.1 格式数据集根目录(含 meta/ data/ videos/)
#   output_dir   : 训练产物输出目录(checkpoint、日志)
#   gpu_id       : GPU 编号,默认 0
#   extra_opts   : 透传给 scripts/train.py 的参数,常用的有:
#                    --steps 100000             训练步数(默认 100k)
#                    --batch-size 8             批大小(默认 8)
#                    --horizon 16               扩散预测的动作序列长度(默认 16)
#                    --n-obs-steps 2            输入观测历史帧数(默认 2)
#                    --n-action-steps 8         每次执行的动作步数(默认 8)
#                    --num-inference-steps N    推理去噪步数(默认随训练配置)
#                    --save-freq 20000          存档间隔(默认 20k 步)
#                    --use-amp                  混合精度训练(省显存、提速)
#                    --resume                   从 output_dir 断点续训
#                    --overwrite                覆盖已存在的 output_dir
#                    --wandb                    开启 wandb 记录(默认关闭)
#
# 示例:
#   # 用 click_bell 数据训 10 万步
#   bash finetune.sh training_data/RoboSynChallenge/cobotmagic_Sim_click_bell \
#        outputs/train/dp_click_bell 0
#   # 短跑冒烟测试
#   bash finetune.sh /path/to/dataset outputs/train/dp_smoke 0 --steps 1000
#
# 输出:
#   <output_dir>/checkpoints/<step>/pretrained_model/   —— 可直接用于 eval.sh 部署
#
# 多卡:
#   本脚本是单进程入口;DDP 多卡请直接用 torchrun 调 scripts/train.py
#   并加 --distributed(详见 DP 文档)。
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
echo "  Diffusion Policy 训练"
echo "  数据集 : $DATASET_ROOT"
echo "  输出   : $OUTPUT_DIR"
echo "  GPU    : $GPU_ID"
echo "  额外参数: ${EXTRA_ARGS[*]:-<无>}"
echo "========================================="

cd "$SCRIPT_DIR"

# train.py 会:
#   1. 加载数据集元信息,自动推断 obs/action 维度和相机键
#   2. 构建 Diffusion Policy(视觉编码器 + 条件 U-Net)从零训练
#   3. 按 --save-freq 间隔把 checkpoint 存到 <output_dir>/checkpoints/
uv run --frozen python scripts/train.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
