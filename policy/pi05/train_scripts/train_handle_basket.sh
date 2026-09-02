#!/usr/bin/env bash
# =============================================================================
# pi0.5 单任务微调脚本 —— handle_basket
# =============================================================================
#
# 用途:
#   在 RoboSynChallenge 的 handle_basket 任务数据集上全量微调官方 pi05_base 模型。
#
# 训练配置(定义在 src/openpi/training/config.py 的 "pi05_handle_basket"):
#   - 基座模型   : gs://openpi-assets/checkpoints/pi05_base/params(首次运行自动下载)
#   - 模型结构   : Pi0Config(pi05=True, action_horizon=50) —— 每次预测 50 步动作块
#   - 数据集     : RoboSynChallenge/cobotmagic_Sim_handle_basket
#   - 动作变换   : extra_delta_transform=True —— 数据集为绝对关节角,
#                  训练时转成相对动作块首帧的 delta 动作(夹爪维度保持绝对值)
#   - batch_size : 64
#   - 训练步数   : 20,000(可用 --num-train-steps 覆盖)
#   - 学习率     : cosine decay,warmup 1000 步,peak 2.5e-5 → 2.5e-6
#   - 优化器     : AdamW(b1=0.9, b2=0.95, grad clip 1.0),EMA 0.99
#   - 存档间隔   : 每 10,000 步存一次 checkpoint
#
# 用法:
#   ./train_handle_basket.sh [GPU_ID] [EXP_NAME] [extra args...]
#
#   GPU_ID    : 使用的 GPU 编号,默认 0;多卡用逗号分隔,如 "0,1"
#   EXP_NAME  : 实验名(决定 checkpoint 子目录),默认 "handle_basket"
#   extra args: 透传给 train.py 的其他参数,例如:
#                 --resume                恢复中断的训练(与默认的 --overwrite 互斥,
#                                         传了 --resume 本脚本会自动去掉 --overwrite)
#                 --num-train-steps=30000 覆盖训练步数
#                 --batch-size=32         显存不够时调小 batch
#
# 输出位置:
#   checkpoints/pi05_handle_basket/<EXP_NAME>/<step>/
#
# 前置条件:
#   1. 数据集已放到 training_data/RoboSynChallenge/cobotmagic_Sim_handle_basket(LeRobot 格式)
#   2. 已安装 uv,且 policy/pi05 下能 `uv run` 起来
# =============================================================================
set -euo pipefail

# ---------------- 任务相关常量(各任务脚本唯一的差异) ----------------
CONFIG_NAME="pi05_handle_basket"                          # config.py 中注册的训练配置名
REPO_ID="RoboSynChallenge/cobotmagic_Sim_handle_basket"                             # LeRobot 数据集 repo id
DEFAULT_EXP_NAME="handle_basket"                       # 默认实验名

# ---------------- 解析命令行参数 ----------------
GPU_ID="${1:-0}"                                  # 第 1 个参数:GPU,默认 0
EXP_NAME="${2:-$DEFAULT_EXP_NAME}"                # 第 2 个参数:实验名
shift 2 2>/dev/null || shift $# 2>/dev/null || true  # 剩余参数原样透传给 train.py
EXTRA_ARGS=("$@")

# 默认使用 --overwrite(允许覆盖同名实验目录);
# 若用户显式传了 --resume,则改为断点续训,不能再带 --overwrite。
MODE_FLAG="--overwrite"
for arg in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
    if [[ "$arg" == "--resume" ]]; then
        MODE_FLAG=""                              # --resume 已在 EXTRA_ARGS 里,去掉 overwrite
    fi
done

# ---------------- 定位 pi05 根目录并进入 ----------------
# 脚本放在 policy/pi05/train_scripts/ 下,根目录是上一级
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PI05_ROOT"

# ---------------- 环境变量 ----------------
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# LeRobot 数据集根目录:数据需放在 training_data/<repo_id>/
export HF_LEROBOT_HOME="$PI05_ROOT/training_data"
# JAX 预分配 90% 显存,避免碎片化导致 OOM
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "==============================================="
echo " 任务       : handle_basket"
echo " 训练配置   : $CONFIG_NAME"
echo " 数据集     : $REPO_ID"
echo " GPU        : $CUDA_VISIBLE_DEVICES"
echo " 实验名     : $EXP_NAME"
echo " 数据根目录 : $HF_LEROBOT_HOME"
echo " 输出目录   : $PI05_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME/"
echo "==============================================="

# ---------------- 第 0 步:检查数据集是否存在 ----------------
if [[ ! -d "$HF_LEROBOT_HOME/$REPO_ID" ]]; then
    echo "[错误] 未找到数据集: $HF_LEROBOT_HOME/$REPO_ID"
    echo "       请先把 LeRobot 格式的数据集放到该目录,或建立软链接,例如:"
    echo "       mkdir -p $HF_LEROBOT_HOME/$(dirname "$REPO_ID")"
    echo "       ln -s /path/to/your/dataset $HF_LEROBOT_HOME/$REPO_ID"
    exit 1
fi

# ---------------- 第 1 步:计算归一化统计量(仅首次) ----------------
# 训练前必须先算 norm_stats(state / action 的均值方差与分位数),
# 结果写入 assets/<config_name>/<repo_id>/norm_stats.json。
# 已存在时跳过;更换/重新采集数据集后请手动删除该文件并重跑。
NORM_STATS="$PI05_ROOT/assets/$CONFIG_NAME/$REPO_ID/norm_stats.json"
if [[ ! -f "$NORM_STATS" ]]; then
    echo "[1/2] norm_stats 不存在,开始计算: $NORM_STATS"
    uv run --frozen scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
else
    echo "[1/2] norm_stats 已存在,跳过计算: $NORM_STATS"
fi

# ---------------- 第 2 步:启动训练 ----------------
# train.py 会:
#   1. 从 gs://openpi-assets 下载并加载 pi05_base 权重(有本地缓存)
#   2. 按 config 构建数据流水线(3 路相机 + 关节状态 + prompt)
#   3. 全量微调 20k 步,每 10k 步存档到 checkpoints/ 下
# wandb 默认开启,不需要时可加 --no-wandb-enabled。
echo "[2/2] 启动训练 ..."
uv run --frozen scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    $MODE_FLAG \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
