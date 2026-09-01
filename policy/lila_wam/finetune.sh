#!/bin/bash
# =============================================================================
# LiLa-WAM 训练脚本(LeRobot v2.1 数据)
# =============================================================================
#
# 模型简介:
#   LiLa-WAM 是轻量世界-动作模型:冻结的 DINOv3 ViT-L/16 视觉编码器 +
#   0.2B 可训练的 DiT 动作专家,流匹配(flow matching)出动作块,
#   同时用"未来帧特征预测"做辅助监督。任务用 VTT(视觉转移向量)条件化,
#   不吃语言输入。单张 24GB 消费级显卡即可训练。
#
# 用法:
#   bash finetune.sh <dataset_root|config.yaml> <task_name> [gpu_id] [extra_opts...]
#
#   dataset_root : LeRobot v2.1 数据集根目录(含 meta/ data/ videos/)
#                  也可以直接传一个写好的 config.yaml(多数据集/多任务时用这个)
#   task_name    : RoboSynChallenge 任务名,如 click_bell。既是 VTT 的索引键,
#                  也是评测时 deploy_policy 查条件向量用的 key
#   gpu_id       : GPU 编号,默认 0
#   extra_opts   : 透传给 train_lila_wam.py,常用:
#                    --epochs 12            训练轮数(上游建议 stage1 跑 11~12 轮)
#                    --batch_size 32        批大小(显存不够就调小)
#                    --learning_rate 4e-5   stage2 的低学习率
#                    --init_from <ckpt>     stage2:只load权重,重置优化器和调度器
#                    --resume <ckpt>        断点续训(恢复优化器+调度器+epoch)
#                    --max_steps 20         冒烟测试
#
# 示例:
#   # stage 1
#   bash policy/lila_wam/finetune.sh \
#        lerobot_dataset/click_bell/cobotmagic_Sim_click_bell click_bell 0 --epochs 12
#   # stage 2(低学习率再跑 3~4 轮)
#   bash policy/lila_wam/finetune.sh \
#        lerobot_dataset/click_bell/cobotmagic_Sim_click_bell click_bell 0 \
#        --epochs 4 --learning_rate 4e-5 \
#        --init_from policy/lila_wam/checkpoints/sft_.../checkpoint_epoch_12.pt
#
# 流水线(每步都是幂等的,重跑不会重复干活):
#   1) 生成本次训练的 config
#   2) 把视频解成 JPEG 帧缓存(训练是随机采帧,直接随机寻址 AV1 视频太慢)
#   3) 统计动作/状态的 min-max 归一化参数
#   4) 预计算 VTT 任务条件向量(要 DINOv3,吃 GPU)
#   5) 训练
#   跳过 2~4 步:LILA_SKIP_PREP=1 bash finetune.sh ...
#
# 输出:
#   policy/lila_wam/checkpoints/sft_<时间戳>/checkpoint_epoch_<N>.pt
#   同目录下带 config.yaml + norm_stats.json,eval.sh 只需要指到这个目录
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

DATASET_OR_CONFIG="${1:?用法: bash finetune.sh <dataset_root|config.yaml> <task_name> [gpu_id] [extra_opts...]}"
TASK_NAME="${2:?缺少 task_name 参数}"
GPU_ID="${3:-0}"
shift 3 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"

BASE_CONFIG="${LILA_WAM_BASE_CONFIG:-$SCRIPT_DIR/configs/robosyn_3cam.yaml}"
GENERATED_DIR="$SCRIPT_DIR/configs/generated"
CONFIG="$GENERATED_DIR/${TASK_NAME}.yaml"
NORM_STATS="$GENERATED_DIR/${TASK_NAME}_norm_stats.json"

cd "$REPO_ROOT"

if [[ "$DATASET_OR_CONFIG" == *.yaml || "$DATASET_OR_CONFIG" == *.yml ]]; then
    # 直接用现成的 config(多数据集/多任务共训场景)
    CONFIG="$DATASET_OR_CONFIG"
    NORM_STATS="$(dirname "$CONFIG")/$(basename "${CONFIG%.*}")_norm_stats.json"
    echo "使用现成 config: $CONFIG"
else
    if [[ ! -d "$DATASET_OR_CONFIG/meta" ]]; then
        echo "[错误] $DATASET_OR_CONFIG 不是 LeRobot 数据集(缺少 meta/ 目录)" >&2
        exit 1
    fi
    mkdir -p "$GENERATED_DIR"
    echo "=== [1/5] 由 $BASE_CONFIG 生成 $CONFIG ==="
    "$PYTHON_BIN" - "$BASE_CONFIG" "$CONFIG" "$DATASET_OR_CONFIG" "$TASK_NAME" <<'PYEOF'
import sys
from omegaconf import OmegaConf

base, out, dataset_dir, task_name = sys.argv[1:5]
config = OmegaConf.load(base)
config.dataset.dataset_dir = [dataset_dir]
config.dataset.task_names = [task_name]
OmegaConf.save(config, out)
print(f"wrote {out}: dataset_dir={dataset_dir} task={task_name}")
PYEOF
fi

echo "========================================="
echo "  LiLa-WAM 训练"
echo "  数据/配置 : $DATASET_OR_CONFIG"
echo "  任务      : $TASK_NAME"
echo "  GPU       : $GPU_ID"
echo "  额外参数  : ${EXTRA_ARGS[*]:-<无>}"
echo "========================================="

if [[ "${LILA_SKIP_PREP:-0}" != "1" ]]; then
    echo "=== [2/5] 构建 JPEG 帧缓存 ==="
    "$PYTHON_BIN" policy/lila_wam/build_frame_cache.py --config "$CONFIG"

    echo "=== [3/5] 统计归一化参数 -> $NORM_STATS ==="
    "$PYTHON_BIN" policy/lila_wam/compute_norm_stats.py --config "$CONFIG" --output "$NORM_STATS"

    echo "=== [4/5] 预计算 VTT 任务条件向量 ==="
    "$PYTHON_BIN" policy/lila_wam/precompute_task_cond.py --config "$CONFIG"
else
    echo "=== [2-4/5] LILA_SKIP_PREP=1,跳过数据准备 ==="
fi

echo "=== [5/5] 训练 ==="
"$PYTHON_BIN" policy/lila_wam/train_lila_wam.py \
    --config "$CONFIG" \
    --norm_stats_path "$NORM_STATS" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
