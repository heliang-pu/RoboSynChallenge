#!/usr/bin/env bash
# =============================================================================
# DM0.5 (OpenDM) SFT 训练脚本
# =============================================================================
#
# 模型简介:
#   DM0.5 是原力灵机(Dexmal)开源的 VLA 模型,以 HTTP 服务方式推理。
#   训练走 OpenDM 的 SFT 流程:torchrun 多进程 + tyro 配置,
#   默认配置为 chunk_size=50、base_lr=2.5e-5、per_device_batch=8、
#   50000 步、每 10000 步存档(见 playground/dm05_sft_demo.py)。
#
# !! 环境要求 !!
#   训练用独立的 opendm conda 环境(与仿真评估环境分离),先执行:
#     conda create -n opendm python=3.10 -y && conda activate opendm
#     cd policy/dm05/opendm && pip install -e .
#   然后在 opendm 环境里运行本脚本。
#
# !! 数据要求 !!
#   OpenDM 不直接读 LeRobot 数据集,而是读"每行一帧"的 JSONL + 图片目录,
#   且数据集必须先在 opendm/dataset/ 下注册进 CONVERSATION_DATA
#   (字段: jsonl_dir / image_dir / image_keys / state_desc 等)。
#   转换与注册方法见 opendm/docs/zh/dm05_finetuning.md 第 3、5 节。
#   验证流程时可先用内置的 demo 数据集(dataset_name=demo)。
#
# 用法:
#   bash finetune.sh <dataset_name> [nproc_per_node] [output_dir] [extra_opts...]
#
#   dataset_name   : CONVERSATION_DATA 中注册的数据集名(内置示例: demo)
#   nproc_per_node : 训练进程数 = GPU 数,默认 1;
#                    用 CUDA_VISIBLE_DEVICES 控制具体用哪几张卡
#   output_dir     : checkpoint 输出目录,
#                    默认 opendm/user_checkpoints/dm05_sft_<dataset_name>
#   extra_opts     : 透传给 SFT 入口的 tyro 参数,常用:
#                      --trainer-config.num-train-steps 20000    训练步数
#                      --trainer-config.per-device-train-batch-size 4
#                      --trainer-config.save-steps 5000          存档间隔
#                      --optimizer-config.base-lr 1e-5           学习率
#                      --model-config.chunk-size 50              动作块长度
#                                                (改了训练推理必须一致!)
#                      --data-config.compute-norm-stats-max-batches 1
#                                                (冒烟测试时加速统计)
#
# 示例:
#   # 冒烟测试:demo 数据 10 步,验证训练/保存链路
#   bash finetune.sh demo 1 "" \
#        --data-config.compute-norm-stats-max-batches 1 \
#        --trainer-config.num-train-steps 10 --trainer-config.save-steps 10
#   # 正式训练:自己注册的数据集,4 卡
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash finetune.sh my_robosyn_data 4
#
# 输出与部署:
#   <output_dir>/checkpoint-<step>/  内含权重和 norm_stats.json
#   (norm stats 训练时自动计算到 opendm/norm_stats/ 并随 checkpoint 保存,
#   推理必须用同一份做归一化/反归一化)。部署:
#     bash launch/run_dm05_server.sh <checkpoint_dir> \
#          --exp playground/dm05_sft_demo.py --dataset-name <dataset_name>
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENDM_ROOT="$SCRIPT_DIR/opendm"

# ---------------- 参数解析 ----------------
DATASET_NAME="${1:?用法: bash finetune.sh <dataset_name> [nproc_per_node] [output_dir] [extra_opts...]}"
NPROC="${2:-1}"
OUTPUT_DIR="${3:-}"
shift 3 2>/dev/null || shift $#          # 其余参数透传给 SFT 入口(tyro 格式)
EXTRA_ARGS=("$@")

# 输出目录默认按数据集名区分,放在 opendm/user_checkpoints/ 下
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="user_checkpoints/dm05_sft_$DATASET_NAME"

# 基座模型路径,可用环境变量覆盖;下载方法:
#   hf download Dexmal/DM05 --local-dir policy/dm05/opendm/checkpoints/DM05
BASE_MODEL="${DM05_BASE_MODEL:-./checkpoints/DM05}"

# ---------------- 前置检查 ----------------
if [[ ! -d "$OPENDM_ROOT" ]]; then
    echo "[错误] 找不到 opendm 源码: $OPENDM_ROOT" >&2
    exit 1
fi
# 训练依赖装在独立的 opendm 环境里,仿真环境里没有
if ! python -c "import opendm" 2>/dev/null; then
    echo "[错误] 当前 python 环境没有 opendm 包。" >&2
    echo "       请先: conda activate opendm" >&2
    echo "       (环境搭建见 policy/dm05/README.md 第 1 节)" >&2
    exit 1
fi
if [[ ! -d "$OPENDM_ROOT/$BASE_MODEL" && ! -d "$BASE_MODEL" ]]; then
    echo "[错误] 找不到 DM05 基座权重: $BASE_MODEL" >&2
    echo "       下载: hf download Dexmal/DM05 --local-dir $OPENDM_ROOT/checkpoints/DM05" >&2
    exit 1
fi

echo "==============================================="
echo " DM0.5 SFT"
echo " 数据集   : $DATASET_NAME"
echo " 进程/GPU : $NPROC  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<全部>})"
echo " 基座权重 : $BASE_MODEL"
echo " 输出目录 : $OPENDM_ROOT/$OUTPUT_DIR"
echo "==============================================="

# ---------------- 启动训练 ----------------
# dm05_launcher.sh 负责设置 PYTHONPATH / NCCL 环境并 torchrun 拉起
# playground/dm05_sft_demo.py(tyro 入口,--task train 走 exp.train())。
# norm stats 缺失时会先自动计算(存 ./norm_stats/,并随 checkpoint 复制)。
cd "$OPENDM_ROOT"
exec bash script/dm05_launcher.sh \
    --exp playground/dm05_sft_demo.py \
    --nproc_per_node "$NPROC" \
    --task train \
    --model-config.model-name-or-path "$BASE_MODEL" \
    --data-config.dataset-name "$DATASET_NAME" \
    --trainer-config.output-dir "$OUTPUT_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
