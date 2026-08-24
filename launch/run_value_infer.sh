#!/usr/bin/env bash
# =============================================================================
# sim-RECAP Phase C —— 价值推理 + advantage 写回(包装 lerobot-value-infer)
# =============================================================================
#
# 用训好的价值函数给数据集每一帧算 value / n-step advantage,并按任务内
# top-K 比例把 advantage 二值化成 0/1 indicator,三列一起写回数据集:
#   complementary_info.value_<tag>
#   complementary_info.advantage_<tag>
#   complementary_info.acp_indicator_<tag>   <- ACP 训练用这一列
#
# 用法:
#   bash launch/run_value_infer.sh <dataset_dir> <value_ckpt> <tag> [gpu_id] [extra...]
#
#   dataset_dir : 要打标的 LeRobot v3.0 数据集(原地写回!建议先备份或用副本)
#   value_ckpt  : run_value_train.sh 的输出目录(outputs/value_train/<run>)
#   tag         : 字段后缀,用轮次区分,如 round1
#   gpu_id      : 默认 0
#   extra       : 透传,常用:
#                   --acp.n_step 50           n-step advantage 视野
#                                             (默认 50,与 pi0.5 action_horizon 一致)
#                   --acp.positive_ratio 0.3  任务内 top 30% 记为 positive
#                   --runtime.batch_size 32   显存不够时调小
#
# 环境变量 EVO_RL_ROOT / EVO_RL_PYTHON 含义同 run_value_train.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


DATASET_DIR="${1:?用法: run_value_infer.sh <dataset_dir> <value_ckpt> <tag> [gpu_id] [extra...]}"
VALUE_CKPT="${2:?缺少 value_ckpt}"
TAG="${3:?缺少 tag(如 round1)}"
GPU_ID="${4:-0}"
shift 4 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

# 默认用仓库内收编的 Evo-RL(third_party/evo_rl),可用环境变量指向外部 checkout
EVO_RL_ROOT="${EVO_RL_ROOT:-$REPO_ROOT/third_party/evo_rl}"
# 解释器优先级: 收编目录的 .venv > conda evo-rl 环境
if [[ -z "${EVO_RL_PYTHON:-}" ]]; then
    if [[ -x "$EVO_RL_ROOT/.venv/bin/python" ]]; then
        EVO_RL_PYTHON="$EVO_RL_ROOT/.venv/bin/python"
    else
        EVO_RL_PYTHON="$HOME/miniconda3/envs/evo-rl/bin/python"
    fi
fi

[[ -d "$EVO_RL_ROOT/src/lerobot" ]] || {
    echo "错误: 找不到 Evo-RL 源码 $EVO_RL_ROOT(可用 EVO_RL_ROOT 覆盖)" >&2; exit 1; }
[[ -x "$EVO_RL_PYTHON" ]] || {
    echo "错误: 找不到解释器 $EVO_RL_PYTHON(可用 EVO_RL_PYTHON 覆盖)" >&2; exit 1; }

DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
REPO_ID="local/$(basename "$DATASET_DIR")"
OUTPUT_DIR="$REPO_ROOT/outputs/value_infer/${RUN_NAME:-$(basename "$DATASET_DIR")_$TAG}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$EVO_RL_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "==============================================="
echo " 价值推理 + advantage 写回"
echo " 数据集 : $DATASET_DIR (原地写回三列, 后缀 _$TAG)"
echo " 价值模型: $VALUE_CKPT"
echo " GPU    : $GPU_ID"
echo "==============================================="

exec "$EVO_RL_PYTHON" -m lerobot.scripts.lerobot_value_infer \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_DIR" \
    --inference.checkpoint_path="$VALUE_CKPT" \
    --runtime.device=cuda \
    --runtime.batch_size=64 \
    --acp.enable=true \
    --acp.n_step=50 \
    --acp.positive_ratio=0.3 \
    --acp.value_field="complementary_info.value_${TAG}" \
    --acp.advantage_field="complementary_info.advantage_${TAG}" \
    --acp.indicator_field="complementary_info.acp_indicator_${TAG}" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="value_infer_${TAG}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
