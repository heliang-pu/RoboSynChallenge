#!/usr/bin/env bash
# 阶段 9:用官方 random 协议评估 ACP 微调后的策略(推理链路自动给 prompt 追加 "Advantage: positive")
# 用法: 08_eval.sh <task> <tag> <exp_name> [episodes=100] [gpu=0] [ckpt_id]
# 对照组(SIMRECAP_INDICATOR_KEY=none 训出的纯 SFT 模型)评估时也必须 SIMRECAP_INDICATOR_KEY=none,否则 prompt 会被追加 Advantage: positive
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}; EXP=${3:?exp_name}; N=${4:-100}; GPU=${5:-0}; CK=${6:-}
export SIMRECAP_REPO_ID="RoboSynChallenge/simrecap_${TASK}_${TAG}"
export SIMRECAP_INDICATOR_KEY="${SIMRECAP_INDICATOR_KEY:-complementary_info.acp_indicator_$TAG}"
cd "$REPO/policy/pi05"
# deploy_policy.yml 默认 checkpoint_id=30000,20k 步训练只存 10000/19999,必须显式指定;默认取最大步数目录
[ -n "$CK" ] || CK=$(find "checkpoints/pi05_sim_recap/$EXP" -maxdepth 1 -mindepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null | sort -n | tail -1)
[ -n "$CK" ] || die "checkpoints/pi05_sim_recap/$EXP 下没有 checkpoint"
log "使用 checkpoint $CK"; extra=(--checkpoint_id "$CK")
bash eval.sh "$TASK" random pi05_sim_recap "$EXP" "$GPU" --max_episodes "$N" --headless True "${extra[@]}"
