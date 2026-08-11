#!/usr/bin/env bash
# 在 pro6000 上对指定 step 的 ckpt 跑冒烟：同步 -> 评测 -> 摘要
#
#   bash smoke_ckpt.sh 4000 [集数]
#
# 只从 4090 读该 step 的 mp_rank_00_model_states.pt 和 config.py，
# 不碰 last.ckpt（训练还在写它）。
set -uo pipefail

STEP="${1:?用法: smoke_ckpt.sh <step> [episodes]}"
EPISODES="${2:-5}"

REPO=/workspace/users/fmc3-8-workspace/Chen/robosynchallenge/RoboSynChallenge
SRC_HOST=phl@192.168.1.138
SRC=/home/phl/workspace/RoboSynChallenge/policy/xr1/checkpoints/project_robosynchallenge-xr1/sample_loading_eef
DST=/workspace/xr1_deploy/step${STEP}
STATS="$REPO/policy/xr1/training_data/sample_loading_eef/xr1_stats.json"
LOG=/workspace/xr1_smoke_step${STEP}.log

echo ">>> [1/3] 同步 ckpt-${STEP}"
if [[ ! -f "$DST/last.ckpt/checkpoint/mp_rank_00_model_states.pt" ]]; then
    ssh -o BatchMode=yes "$SRC_HOST" "test -d '$SRC/epoch=0-step=${STEP}.ckpt'" || {
        echo "!!! 4090 上还没有 epoch=0-step=${STEP}.ckpt，等它生成再跑" >&2; exit 2; }
    mkdir -p "$DST/last.ckpt/checkpoint"
    rsync -a "$SRC_HOST:$SRC/config.py" "$DST/" || exit 1
    rsync -a "$SRC_HOST:$SRC/epoch=0-step=${STEP}.ckpt/checkpoint/mp_rank_00_model_states.pt" \
        "$DST/last.ckpt/checkpoint/" || exit 1
else
    echo "    已存在，跳过同步"
fi
# config.py 里补齐 encoding/gripper_range（老配置没有；已有则幂等跳过）
python3 /tmp/patch_cfg.py "$DST/config.py" 2>/dev/null || true

echo ">>> [2/3] 冒烟 ${EPISODES} 集"
cd "$REPO" || exit 1
bash policy/xr1/eval.sh sample_loading clear "step${STEP}" posttrain 0 \
    --model_path "$DST" \
    --stats_path "$STATS" \
    --max_episodes "$EPISODES" \
    --headless True > "$LOG" 2>&1
echo "    退出码 $?  日志 $LOG"

echo ">>> [3/3] 摘要"
echo "--- 模型初始化（确认 stats / 解码 / 夹爪量程）---"
grep -a "\[XR1\] 就绪" "$LOG" | head -1
echo "--- 每集 IK 失败率 + 收缩救回 ---"
grep -a "IK 失败率" "$LOG"
echo "--- 成功率 ---"
grep -aE "SUCCESS|FAIL|success rate|Evaluation Results" "$LOG" | tail -8
echo "--- 推理延迟 ---"
grep -a "policy eval timing" "$LOG" | tail -5
echo "--- 视频 ---"
find "$REPO/eval_result" -name "*.mp4" -newermt "-30 minutes" 2>/dev/null | head -5
