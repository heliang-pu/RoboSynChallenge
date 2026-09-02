#!/bin/bash
# =============================================================================
# pi0.5 (openpi) 微调脚本 —— 通用入口
# =============================================================================
#
# 模型简介:
#   pi0.5 是 pi0 的升级版 VLA(PaliGemma 3B VLM + action expert,flow matching
#   生成连续动作块),开放世界泛化能力更强。本脚本在官方 pi05_base 权重上
#   做全量微调(JAX 训练栈)。
#
# 与 train_scripts/ 的关系:
#   本脚本是"传入任意 config 名"的通用入口;train_scripts/train_<task>.sh
#   是 10 个任务各自的一键脚本(已内置 config 名和数据检查),日常单任务
#   训练建议直接用后者。
#
# 用法:
#   bash finetune.sh <train_config_name> <exp_name> <gpu_id> [extra_opts...]
#
#   train_config_name : src/openpi/training/config.py 中注册的配置名,可选:
#                         pi05_base_robosynchallenge_full   (全任务通用配置)
#                         pi05_click_bell / pi05_sample_loading /
#                         pi05_water_pouring / pi05_mixer_operating /
#                         pi05_manipulate_pipette / pi05_handle_basket /
#                         pi05_items_handover / pi05_item_assembly /
#                         pi05_drawer_open_place / pi05_table_rearrangement
#                       各任务配置均为: pi05=True, action_horizon=50,
#                       batch_size=64, num_train_steps=20000,
#                       extra_delta_transform=True(绝对关节角 → delta 动作,
#                       夹爪保持绝对值),cosine 学习率(warmup 1k,
#                       peak 2.5e-5 → 2.5e-6),AdamW + EMA 0.99,每 10k 步存档。
#   exp_name          : 实验名,决定 checkpoint 子目录
#   gpu_id            : GPU 编号;多卡用逗号分隔如 "0,1"(配 --fsdp-devices)
#   extra_opts        : 透传给 train.py,常用:
#                         --resume                  断点续训(本脚本自动去掉 --overwrite)
#                         --num-train-steps=30000   覆盖训练步数
#                         --batch-size=32           显存不够时调小
#                         --no-wandb-enabled        关闭 wandb
#
# 示例:
#   bash finetune.sh pi05_click_bell click_bell_v1 0
#   bash finetune.sh pi05_click_bell click_bell_v1 0 --resume
#
# 数据准备:
#   LeRobot 数据集需放在 training_data/<repo_id>/,repo_id 与所选 config 的
#   data.repo_id 一致(RoboSynChallenge/cobotmagic_Sim_<task>)。
#
# 输出:
#   checkpoints/<train_config_name>/<exp_name>/<step>/
#   评估部署: bash eval.sh <task> <setting> <checkpoint_path> <gpu>
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
PI05_ROOT="$SCRIPT_DIR"
cd "$PI05_ROOT"

# ---------------- 环境变量 ----------------
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# LeRobot 本地数据集根目录
export HF_LEROBOT_HOME="$PI05_ROOT/training_data"
# JAX 预分配 90% 显存,避免碎片化 OOM
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "==============================================="
echo " 训练配置 : $TRAIN_CONFIG_NAME"
echo " 实验名   : $EXP_NAME"
echo " GPU      : $CUDA_VISIBLE_DEVICES"
echo " 数据目录 : $HF_LEROBOT_HOME"
echo " 输出目录 : $PI05_ROOT/checkpoints/$TRAIN_CONFIG_NAME/$EXP_NAME/"
echo "==============================================="

# ---------------- 第 1 步:归一化统计量(仅首次) ----------------
# 训练前必须有 norm_stats(state/action 的均值方差与分位数),
# 存在 assets/<config_name>/ 下;缺失时自动计算。
# 更换数据集后请删除对应 json 重算。
if [[ ! -d "$PI05_ROOT/assets/$TRAIN_CONFIG_NAME" ]]; then
    echo "[1/2] norm_stats 不存在,开始计算 ..."
    uv run --frozen scripts/compute_norm_stats.py --config-name "$TRAIN_CONFIG_NAME"
else
    echo "[1/2] norm_stats 已存在(assets/$TRAIN_CONFIG_NAME/),跳过"
fi

# ---------------- 第 2 步:启动训练 ----------------
# train.py 会:
#   1. 从 gs://openpi-assets 下载并加载 pi05_base 权重(有本地缓存)
#   2. 按 config 构建数据流水线(3 路相机 + 关节状态 + prompt)
#   3. 微调并按 save_interval 存档到 checkpoints/
echo "[2/2] 启动训练 ..."
uv run --frozen scripts/train.py "$TRAIN_CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    $MODE_FLAG \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
