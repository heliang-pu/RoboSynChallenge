#!/bin/bash
# =============================================================================
# pi0 (openpi) 微调脚本
# =============================================================================
#
# 模型简介:
#   pi0 是 Physical Intelligence 的 VLA 基座模型(PaliGemma 3B VLM +
#   300M action expert),用 flow matching 生成连续动作块。
#   本脚本在官方 pi0_base 权重上做全量微调(JAX 训练栈)。
#
# 用法:
#   bash finetune.sh <train_config_name> <exp_name> <gpu_id> [extra_opts...]
#
#   train_config_name : src/openpi/training/config.py 中注册的配置名,
#                       本仓库主要用:
#                         pi0_base_robosynchallenge_full
#                       其配置为: action_horizon=50, batch_size=32,
#                       num_train_steps=30000, extra_delta_transform=True
#                       (绝对关节角 → 相对动作块首帧的 delta,夹爪保持绝对值),
#                       cosine 学习率 (warmup 1k, peak 2.5e-5 → 2.5e-6),
#                       AdamW + EMA 0.99,每 10k 步存档。
#   exp_name          : 实验名,决定 checkpoint 子目录
#   gpu_id            : GPU 编号;多卡用逗号分隔如 "0,1"(配 --fsdp-devices)
#   extra_opts        : 透传给 train.py,常用:
#                         --resume                  断点续训(本脚本自动去掉 --overwrite)
#                         --num-train-steps=50000   覆盖训练步数
#                         --batch-size=16           显存不够时调小
#                         --no-wandb-enabled        关闭 wandb
#
# 示例:
#   bash finetune.sh pi0_base_robosynchallenge_full click_bell_v1 0
#   bash finetune.sh pi0_base_robosynchallenge_full click_bell_v1 0 --resume
#
# 数据准备:
#   LeRobot 数据集需放在 training_data/<repo_id>/,repo_id 与所选 config 的
#   data.repo_id 一致(默认 RoboSynChallenge/cobotmagic_Sim_click_bell,
#   换任务需修改 config.py 中的 repo_id 或新增配置)。
#
# 输出:
#   checkpoints/<train_config_name>/<exp_name>/<step>/
#
# 显存:
#   全量微调 batch=32 约需 80GB;单张 48G 卡可将 batch 调小或用 LoRA 配置。
# =============================================================================
set -euo pipefail

# ---------------- 参数解析 ----------------
TRAIN_CONFIG_NAME="${1:?用法: bash finetune.sh <train_config_name> <exp_name> <gpu_id> [extra_opts...]}"
EXP_NAME="${2:?缺少 exp_name 参数}"
GPU_ID="${3:?缺少 gpu_id 参数}"
shift 3 2>/dev/null || shift $#          # 其余参数透传给 train.py
EXTRA_ARGS=("$@")

# 默认 --overwrite(允许覆盖同名实验目录);传了 --resume 则去掉,
# 因为 train.py 不允许两者同时出现。
MODE_FLAG="--overwrite"
for arg in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
    [[ "$arg" == "--resume" ]] && MODE_FLAG=""
done

# ---------------- 定位根目录 ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI0_ROOT="$SCRIPT_DIR"
cd "$PI0_ROOT"

# ---------------- 环境变量 ----------------
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# LeRobot 本地数据集根目录
export HF_LEROBOT_HOME="$PI0_ROOT/training_data"
# JAX 预分配 90% 显存,避免碎片化 OOM
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "==============================================="
echo " 训练配置 : $TRAIN_CONFIG_NAME"
echo " 实验名   : $EXP_NAME"
echo " GPU      : $CUDA_VISIBLE_DEVICES"
echo " 数据目录 : $HF_LEROBOT_HOME"
echo " 输出目录 : $PI0_ROOT/checkpoints/$TRAIN_CONFIG_NAME/$EXP_NAME/"
echo "==============================================="

# ---------------- 第 1 步:归一化统计量(仅首次) ----------------
# 训练前必须有 norm_stats(state/action 的均值方差与分位数),
# 存在 assets/<config_name>/ 下;缺失时自动计算。
# 更换数据集后请删除对应 json 重算。
if [[ ! -d "$PI0_ROOT/assets/$TRAIN_CONFIG_NAME" ]]; then
    echo "[1/2] norm_stats 不存在,开始计算 ..."
    uv run --frozen scripts/compute_norm_stats.py --config-name "$TRAIN_CONFIG_NAME"
else
    echo "[1/2] norm_stats 已存在(assets/$TRAIN_CONFIG_NAME/),跳过"
fi

# ---------------- 第 2 步:启动训练 ----------------
# train.py 会:
#   1. 从 gs://openpi-assets 下载并加载 pi0_base 权重(有本地缓存)
#   2. 按 config 构建数据流水线(3 路相机 + 关节状态 + prompt)
#   3. 微调并按 save_interval 存档到 checkpoints/
echo "[2/2] 启动训练 ..."
uv run --frozen scripts/train.py "$TRAIN_CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    $MODE_FLAG \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
