#!/usr/bin/env bash
# 把 acp_indicator 烤进 task 文本(per-frame task_index → "…\nAdvantage: positive/negative"),
# 供不走 openpi ACPAdvantageTag 的 VLA 框架直接按 task 训练。非破坏性:输出到新目录 + NAS。
# 用法:09_bake_prompt.sh <task> <tag> [source_name] [output_name]
# 例:09_bake_prompt.sh sample_loading round1 simrecap_sample_loading_round1_vlm3500 simrecap_sample_loading_round1_vlm3500_baked_prompt
# 依赖:LeRobot v3 数据集,带 acp_indicator_<tag> 列。输出为派生副本,不覆盖源数据。
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}
SRC_NAME=${3:-simrecap_${TASK}_${TAG}}
DST_NAME=${4:-${SRC_NAME}_baked_prompt}
SRC=$REPO/lerobot_dataset/$SRC_NAME
DST=$REPO/lerobot_dataset/$DST_NAME
NAS_DST=$NAS/recap_reward_dataset/$DST_NAME
need_file "$SRC/meta/info.json"
[ -e "$DST" ] && die "$DST 已存在;删除后重跑"
log "复制 $SRC -> $DST"; cp -a --reflink=auto "$SRC" "$DST"
"$PY_SIM" "$REPO/scripts/bake_acp_prompt_into_lerobot.py" "$DST" \
    --indicator-field "complementary_info.acp_indicator_$TAG" || die "烤 prompt 失败"
log "校验(pi05 环境训练读取门)"
"$REPO/policy/pi05/.venv/bin/python" "$REPO/scripts/validate_lerobot_dataset.py" "$DST" \
    --expected-episodes "$("$PY_SIM" -c "import json;print(json.load(open('$DST/meta/info.json'))['total_episodes'])")" \
    --producer-exit-code 0 >/dev/null 2>&1 && log "校验门全过" || log "警告: 校验未过,手动查"
log "同步到 NAS: $NAS_DST"; mkdir -p "$NAS_DST" && rsync -a --delete "$DST/" "$NAS_DST/"
log "完成: 本地 $DST  |  NAS $NAS_DST"
