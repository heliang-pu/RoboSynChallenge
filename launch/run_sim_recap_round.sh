#!/usr/bin/env bash
# =============================================================================
# sim-RECAP 一轮闭环编排(无人值守,无需 Human-in-the-Loop)
# =============================================================================
#
# 思想(来自 π*0.6 RECAP / Evo-RL,仿真评估器替代人工标注):
#
#   rollout π_k(评估器自动判成败) ─┐
#                                    ├→ 专家+rollout 合并数据池
#   专家演示(天然全 success) ──────┘        │
#                                        训练价值函数(pistar06)
#                                              │
#                                    n-step advantage → 0/1 indicator 写回
#                                              │
#                              ACP 微调 pi0.5(prompt 注入 Advantage 标签)→ π_{k+1}
#
# 数据版本约定:对外统一 LeRobot v2.1。
#   专家数据集给 v2.1 即可(v3.0 也接受);最终产物是 v2.1;
#   价值函数栈(Evo-RL)只认 v3.0,所以中间产物放在隐藏工作目录
#   lerobot_dataset/.simrecap_work/<task>_<tag>/ 里,不污染对外目录。
#
# 用法:
#   bash launch/run_sim_recap_round.sh <task> <setting> <train_config> <model_name> \
#        <round_tag> [episodes] [gpu] [expert_dataset]
#
#   task           : 任务名,如 click_bell
#   setting        : 评估设置,建议 random(官方口径)
#   train_config   : 当前策略 π_k 的 openpi 训练配置名
#   model_name     : 当前策略 π_k 的实验名(checkpoints/<config>/<model_name>)
#   round_tag      : 本轮标签后缀,如 round1(决定 indicator 列名和数据集名)
#   episodes       : rollout 集数,默认 200
#   gpu            : 默认 0
#   expert_dataset : 专家数据集路径(v2.1 或 v3.0),
#                    默认 policy/pi05/training_data/RoboSynChallenge/cobotmagic_Sim_<task>
#
# 分阶段执行:环境变量 START_STAGE=N 可从第 N 阶段续跑(1-7)。
#   1 rollout 采集(v3.0,工作区)   2 专家数据准备(v2.1 则上转 v3.0 副本)
#   3 合并数据池                    4 打 episode_success 标签
#   5 价值函数训练                  6 advantage 写回
#   7 转 v2.1 并发布到 lerobot_dataset/ + 链接进 pi05 训练目录
#
# 各阶段用的环境:
#   1          仿真环境(policy/pi05/.venv,eval.sh 内部处理)
#   2,3,4,7    SIM_PYTHON(默认 conda robosyn,需 lerobot 0.4.4 + pyarrow)
#   5,6        Evo-RL 环境(见 run_value_train.sh 头部说明)
#
# 磁盘提示:阶段 2 对 v2.1 专家数据做整份复制后上转(不动原数据),
# 1000 集带视频约需同等体量的临时空间;round2 起复用上轮合并池可避免。
#
# 完成后启动 ACP 微调(阶段 8,手动):
#   1) 若 task/round 与 config 中 pi05_sim_recap 的默认值不同,编辑
#      policy/pi05/src/openpi/training/config.py 里 pi05_sim_recap 的
#      repo_id 与 acp_indicator_key(脚本结束时会打印应设的值);
#   2) bash policy/pi05/finetune.sh pi05_sim_recap <exp_name> <gpu>
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK="${1:?用法见文件头注释}"
SETTING="${2:?缺少 setting}"
TRAIN_CONFIG="${3:?缺少 train_config}"
MODEL_NAME="${4:?缺少 model_name}"
ROUND_TAG="${5:?缺少 round_tag(如 round1)}"
EPISODES="${6:-200}"
GPU_ID="${7:-0}"
EXPERT_DATASET="${8:-$REPO_ROOT/policy/pi05/training_data/RoboSynChallenge/cobotmagic_Sim_$TASK}"

START_STAGE="${START_STAGE:-1}"
SIM_PYTHON="${SIM_PYTHON:-$HOME/miniconda3/envs/robosyn/bin/python}"
[[ -x "$SIM_PYTHON" ]] || SIM_PYTHON=python

DATASET_ROOT="$REPO_ROOT/lerobot_dataset"
# v3.0 中间产物全部收在隐藏工作目录,对外目录只出现 v2.1
WORK_ROOT="$DATASET_ROOT/.simrecap_work/${TASK}_${ROUND_TAG}"
ROLLOUT_ID="rollout_v30"
ROLLOUT_DIR="$WORK_ROOT/$ROLLOUT_ID"
EXPERT_V30_ID="expert_v30"
EXPERT_V30_DIR="$WORK_ROOT/$EXPERT_V30_ID"
MERGED_ID="merged_v30"
MERGED_DIR="$WORK_ROOT/$MERGED_ID"
FINAL_ID="simrecap_${TASK}_${ROUND_TAG}"
FINAL_DIR="$DATASET_ROOT/$FINAL_ID"
INDICATOR_COL="complementary_info.acp_indicator_${ROUND_TAG}"
VALUE_RUN="value_${TASK}_${ROUND_TAG}"
TRAIN_DATA_DST="$REPO_ROOT/policy/pi05/training_data/RoboSynChallenge/$FINAL_ID"

stage() { echo; echo "############ [sim-RECAP $ROUND_TAG] 阶段 $1: $2 ############"; }

dataset_version() {
    "$SIM_PYTHON" -c "import json;print(json.load(open('$1/meta/info.json')).get('codebase_version',''))"
}

mkdir -p "$WORK_ROOT"

# ---------------- 阶段 1: rollout 采集(带自动成败标签,v3.0 落工作区) ----------------
if [[ "$START_STAGE" -le 1 ]]; then
    stage 1 "rollout 采集 $EPISODES 集 ($TASK/$SETTING, π=$TRAIN_CONFIG/$MODEL_NAME)"
    [[ -e "$ROLLOUT_DIR" ]] && { echo "错误: $ROLLOUT_DIR 已存在,换个 round_tag 或删除后重跑" >&2; exit 1; }
    (cd "$REPO_ROOT/policy/pi05" && bash eval.sh "$TASK" "$SETTING" "$TRAIN_CONFIG" "$MODEL_NAME" "$GPU_ID" \
        --max_episodes "$EPISODES" \
        --headless True \
        --rollout_save True \
        --rollout_save_path "lerobot_dataset/.simrecap_work/${TASK}_${ROUND_TAG}/$ROLLOUT_ID")
    [[ -f "$ROLLOUT_DIR/episode_success.json" ]] || { echo "错误: rollout 边车标签未生成" >&2; exit 1; }
fi

# ---------------- 阶段 2: 专家数据准备(v2.1 上转 v3.0 副本;v3.0 直接软链) ----------------
if [[ "$START_STAGE" -le 2 ]]; then
    stage 2 "专家数据准备: $EXPERT_DATASET"
    [[ -f "$EXPERT_DATASET/meta/info.json" ]] || { echo "错误: 专家数据集不存在 $EXPERT_DATASET" >&2; exit 1; }
    rm -rf "$EXPERT_V30_DIR"
    EXPERT_VERSION="$(dataset_version "$EXPERT_DATASET")"
    case "$EXPERT_VERSION" in
        v2.1)
            echo "专家数据是 v2.1,复制到工作区并上转 v3.0(原数据不动)..."
            cp -r "$EXPERT_DATASET" "$EXPERT_V30_DIR"
            "$SIM_PYTHON" -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
                --repo-id "$EXPERT_V30_ID" --root "$WORK_ROOT" --push-to-hub false
            ;;
        v3*)
            echo "专家数据已是 v3.0,直接软链进工作区"
            ln -sfn "$(realpath "$EXPERT_DATASET")" "$EXPERT_V30_DIR"
            ;;
        *)
            echo "错误: 无法识别专家数据版本 '$EXPERT_VERSION'" >&2; exit 1
            ;;
    esac
fi

# ---------------- 阶段 3: 合并专家 + rollout 成数据池(v3.0,工作区) ----------------
if [[ "$START_STAGE" -le 3 ]]; then
    stage 3 "合并 $EXPERT_V30_ID + $ROLLOUT_ID -> $MERGED_ID"
    [[ -e "$MERGED_DIR" ]] && { echo "错误: 合并输出已存在 $MERGED_DIR" >&2; exit 1; }
    # 顺序必须是 [专家, rollout]:阶段 4 依赖"前 N 集是专家"的布局
    "$SIM_PYTHON" -m lerobot.scripts.lerobot_edit_dataset \
        --root "$WORK_ROOT" \
        --repo_id "$MERGED_ID" \
        --push_to_hub false \
        --operation.type merge \
        --operation.repo_ids "['$EXPERT_V30_ID', '$ROLLOUT_ID']"
fi

# ---------------- 阶段 4: 打 episode_success 标签 ----------------
if [[ "$START_STAGE" -le 4 ]]; then
    EXPERT_EPISODES=$("$SIM_PYTHON" -c "import json;print(json.load(open('$EXPERT_V30_DIR/meta/info.json'))['total_episodes'])")
    # 前缀数据集若是上一轮的合并池(自带 episode_success.json,内含失败集),
    # 用它的标签;纯专家数据则整体视为 success。
    PREFIX_SIDECAR_ARGS=()
    if [[ -f "$EXPERT_V30_DIR/episode_success.json" ]]; then
        PREFIX_SIDECAR_ARGS=(--prefix-sidecar "$EXPERT_V30_DIR/episode_success.json")
        stage 4 "打标: 前 $EXPERT_EPISODES 集按前缀边车(上轮合并池),其余按 rollout 边车"
    else
        stage 4 "打标: 前 $EXPERT_EPISODES 集(专家)=success,其余按 rollout 边车"
    fi
    "$SIM_PYTHON" "$REPO_ROOT/scripts/label_rollout_dataset.py" \
        --dataset "$MERGED_DIR" \
        --sidecar "$ROLLOUT_DIR/episode_success.json" \
        --prefix-success "$EXPERT_EPISODES" \
        ${PREFIX_SIDECAR_ARGS[@]+"${PREFIX_SIDECAR_ARGS[@]}"}
fi

# ---------------- 阶段 5: 价值函数训练 ----------------
if [[ "$START_STAGE" -le 5 ]]; then
    stage 5 "训练 pistar06 价值函数 -> outputs/value_train/$VALUE_RUN"
    bash "$SCRIPT_DIR/run_value_train.sh" "$MERGED_DIR" "$VALUE_RUN" "$GPU_ID"
fi

# ---------------- 阶段 6: advantage 推理写回 ----------------
if [[ "$START_STAGE" -le 6 ]]; then
    stage 6 "价值推理 + advantage 二值化写回(列后缀 _$ROUND_TAG)"
    bash "$SCRIPT_DIR/run_value_infer.sh" "$MERGED_DIR" \
        "$REPO_ROOT/outputs/value_train/$VALUE_RUN" "$ROUND_TAG" "$GPU_ID"
fi

# ---------------- 阶段 7: 转 v2.1 并发布(对外只有 v2.1) ----------------
if [[ "$START_STAGE" -le 7 ]]; then
    stage 7 "v3.0 -> v2.1 并发布到 $FINAL_DIR"
    [[ -e "$FINAL_DIR" ]] && { echo "错误: 发布目标已存在 $FINAL_DIR" >&2; exit 1; }
    # 转换前把合并池的成败标签导出成边车(v2.1 的 meta 不保留自定义列;
    # 下一轮拿本数据池当前缀时,阶段 4 靠这个边车恢复标签)
    "$SIM_PYTHON" - <<PYEOF
import glob, json
import pyarrow.parquet as pq
records = []
for f in sorted(glob.glob("$MERGED_DIR/meta/episodes/**/*.parquet", recursive=True)):
    t = pq.read_table(f, columns=["episode_index", "episode_success"])
    for idx, label in zip(t.column("episode_index").to_pylist(), t.column("episode_success").to_pylist()):
        records.append({"episode_index": int(idx), "success": label == "success"})
records.sort(key=lambda r: r["episode_index"])
with open("$MERGED_DIR/episode_success.json", "w") as f:
    json.dump({"saved_episode_count": len(records), "episodes": records}, f, indent=2)
print(f"标签边车已导出: {len(records)} 集")
PYEOF
    "$SIM_PYTHON" "$REPO_ROOT/scripts/convert_lerobot3.0_to_2.1.py" \
        --repo-id "$MERGED_ID" --root "$WORK_ROOT"
    # 硬门禁:确认 indicator 列在转换后还活着
    "$SIM_PYTHON" - <<PYEOF
import glob, sys
import pyarrow.parquet as pq
files = sorted(glob.glob("$MERGED_DIR/data/**/*.parquet", recursive=True))
if not files:
    sys.exit("错误: 转换后找不到 data parquet")
cols = pq.read_schema(files[0]).names
if "$INDICATOR_COL" not in cols:
    sys.exit(f"错误: v2.1 转换丢失了 $INDICATOR_COL 列,可用列: {[c for c in cols if 'complementary' in c]}")
print(f"indicator 列存活确认: $INDICATOR_COL")
PYEOF
    mv "$MERGED_DIR" "$FINAL_DIR"
    mkdir -p "$(dirname "$TRAIN_DATA_DST")"
    ln -sfn "$FINAL_DIR" "$TRAIN_DATA_DST"
    echo "已发布 v2.1 数据池: $FINAL_DIR"
    echo "已链接: $TRAIN_DATA_DST -> $FINAL_DIR"
    echo "(工作区 $WORK_ROOT 保留 rollout 原始数据与价值中间产物,确认无误后可删)"
fi

# ---------------- 阶段 8 提示: ACP 微调 ----------------
cat <<EOF

############ [sim-RECAP $ROUND_TAG] 数据管线完成 ############

下一步 (ACP 微调 π_{k+1}):
  1. 确认 policy/pi05/src/openpi/training/config.py 中 pi05_sim_recap 配置:
       repo_id           = "RoboSynChallenge/$FINAL_ID"
       acp_indicator_key = "$INDICATOR_COL"
       weight_loader     指向 π_k 的 checkpoint(迭代式训练)
  2. 启动训练:
       bash policy/pi05/finetune.sh pi05_sim_recap ${TASK}_${ROUND_TAG} $GPU_ID
  3. 训完评估(推理时自动带 "Advantage: positive" 标签):
       cd policy/pi05 && bash eval.sh $TASK $SETTING pi05_sim_recap ${TASK}_${ROUND_TAG} $GPU_ID
  4. 下一轮: 用新 checkpoint 重跑本脚本,round_tag 递增(如 round2),
     expert_dataset 换成本轮的 $FINAL_DIR(v2.1,含全部标签列)以持续扩大数据池。
EOF
