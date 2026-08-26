#!/usr/bin/env bash
# 阶段 1:策略 rollout 分片并行采集(评估器自动打成败)→ 合并分片 → 交付 v2.1
# 用法: 01_rollout.sh <task> <tag> <train_config> <model_name> <ckpt_id> <episodes> [shards=2] [gpu=0] [setting=random_rollout]
# 环境: SEED_BASE=10001  分片 seed = SEED_BASE*(片号+1);同一策略换批次采集务必换 SEED_BASE(否则场景重复,不是新数据)
# 例:   01_rollout.sh sample_loading round1 pi05_sample_loading sample_loading 28000 150
# 产物: $WORK_ROOT/<task>_<tag>/rollout_v30/rollout_merged(v3.0 + 边车)
#       lerobot_dataset/rollouts/<task>_<tag>(v2.1 + 边车,交付/复核用)
# 前置: GPU 空闲显存 >= shards*0.32*总显存(+每片 ~3GB 渲染);建议先停掉其它占卡任务
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}; CFG=${3:?train_config}; MODEL=${4:?model_name}; CKPT=${5:?ckpt_id}; EPISODES=${6:?episodes}
SHARDS=${7:-2}; GPU=${8:-0}; SETTING=${9:-random_rollout}; SEED_BASE=${SEED_BASE:-10001}   # 同一策略再采一批(如 held-out)时换 SEED_BASE,否则场景与上次完全相同
WORK=$WORK_ROOT/${TASK}_${TAG}; SH=$WORK/shards; DELIVER=$REPO/lerobot_dataset/rollouts/${TASK}_${TAG}
[ -e "$WORK/rollout_v30" ] && die "$WORK/rollout_v30 已存在;换 tag 或删除后重跑"
[ -e "$SH" ] && die "$SH 已存在(上次半途产物);删除后重跑"
# 用 pi05_sim_recap 训出的策略做 rollout 时,配置由环境变量驱动:repo_id 必须与 checkpoint 内 assets 的 norm stats 一致
if [ "$CFG" = pi05_sim_recap ]; then
    A=$REPO/policy/pi05/checkpoints/$CFG/$MODEL/$CKPT/assets
    ns=$(cd "$A" 2>/dev/null && find . -name norm_stats.json | head -1); [ -n "$ns" ] || die "找不到 $A 下的 norm_stats.json"
    export SIMRECAP_REPO_ID=$(dirname "$ns" | sed 's#^\./##'); export SIMRECAP_INDICATOR_KEY=${SIMRECAP_INDICATOR_KEY:-complementary_info.acp_indicator_$TAG}
    log "pi05_sim_recap: SIMRECAP_REPO_ID=$SIMRECAP_REPO_ID"
fi
need_file "$REPO/configs/$TASK/$SETTING/gym_config.json"
mkdir -p "$SH"
FRAC=$(awk -v s="$SHARDS" 'BEGIN{printf "%.2f", (s>1)?0.32:0.4}')
per=$(( (EPISODES + SHARDS - 1) / SHARDS )); pids=()
for i in $(seq 0 $((SHARDS-1))); do
    n=$(( EPISODES - per*i )); [ $n -gt $per ] && n=$per; [ $n -le 0 ] && continue
    log "启动分片 s$i: $n 集, seed $((SEED_BASE*(i+1))), 显存池 $FRAC"
    ( cd "$REPO/policy/pi05" && XLA_PYTHON_CLIENT_MEM_FRACTION=$FRAC bash eval.sh "$TASK" "$SETTING" "$CFG" "$MODEL" "$GPU" \
        --checkpoint_id "$CKPT" --max_episodes "$n" --seed $((SEED_BASE*(i+1))) --headless True \
        --rollout_save True --rollout_save_path "lerobot_dataset/.simrecap_work/${TASK}_${TAG}/shards/s$i" --eval_video_log False \
        > "$SH/s$i.log" 2>&1 ) & pids+=($!); sleep 45
done
for p in "${pids[@]}"; do wait "$p" || log "警告: 某分片 exit!=0(看 $SH/s*.log)"; done

# 合并分片(顺序 s0,s1,...;边车索引按分片集数偏移)
ids=(); dirs=(); for i in $(seq 0 $((SHARDS-1))); do d=$(nested_dataset_dir "$SH/s$i") || true; [ -n "${d:-}" ] || continue
    need_file "$d/episode_success.json"; strip_meta "$d"; dirs+=("$d"); ids+=("$(realpath --relative-to="$SH" "$d")"); done
[ ${#dirs[@]} -gt 0 ] || die "没有任何分片产出数据集"
rm -rf "$SH/rollout_merged"; log "合并 ${#dirs[@]} 个分片"
merge_v30 "$SH" rollout_merged "${ids[@]}" || die "合并失败"
"$PY_SIM" - "$SH/rollout_merged/episode_success.json" "${dirs[@]}" <<'PYEOF'
import sys, json
out, dirs = sys.argv[1], sys.argv[2:]; eps = []; off = 0
for d in dirs:
    sc = json.load(open(f"{d}/episode_success.json"))
    for r in sc["episodes"]: r = dict(r); r["episode_index"] += off; eps.append(r)
    off += len(sc["episodes"])
json.dump({"labels_field": "episode_success", "saved_episode_count": len(eps), "episodes": eps}, open(out, "w"), indent=2)
print(f"  合并边车: {len(eps)} 集, 成功 {sum(e['success'] for e in eps)}")
PYEOF
mkdir -p "$WORK/rollout_v30"; mv "$SH/rollout_merged" "$WORK/rollout_v30/rollout_merged"
n=$("$PY_SIM" -c "import json;print(json.load(open('$WORK/rollout_v30/rollout_merged/meta/info.json'))['total_episodes'])")
log "rollout_v30/rollout_merged: $n 集"

# v2.1 交付副本
rm -rf "$DELIVER"; mkdir -p "$(dirname "$DELIVER")"; cp -r "$WORK/rollout_v30/rollout_merged" "$DELIVER"
to_v21 "$(dirname "$DELIVER")" "$(basename "$DELIVER")" "$WORK/rollout_v30/rollout_merged/episode_success.json" || die "v2.1 转换失败"
"$REPO/policy/pi05/.venv/bin/python" "$REPO/scripts/validate_lerobot_dataset.py" "$DELIVER" --expected-episodes "$n" --producer-exit-code 0 >/dev/null 2>&1 && log "v2.1 校验门全过" || log "警告: v2.1 校验未通过(手动运行 scripts/validate_lerobot_dataset.py 查看)"
log "完成: $DELIVER(v2.1)  |  下一步: 人工复核视频后用 02_set_label.sh 改标签,再 03_build_pool.sh"
