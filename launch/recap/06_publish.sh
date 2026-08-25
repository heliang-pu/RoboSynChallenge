#!/usr/bin/env bash
# 发布一轮:no_reward(未过价值模型)与 reward(指定 checkpoint 打标)两版 v2.1 → NAS + 本地。
# 阶段 6b-7:全量推理写回三列 → no_reward/reward 两版 v2.1 → NAS + 本地(建议 detach 运行,约 1.5h)
# 用法: 06_publish.sh <task> <tag> <checkpoint_step>      例: publish_round.sh sample_loading round1 003000
# 需要: .simrecap_work/<task>_<tag>/merged_v30 已打好 episode_success;outputs/value_train/value_<task>_<tag>/checkpoints/<step>
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}; STEP=${3:?checkpoint step, e.g. 003000}

WORK=$WORK_ROOT/${TASK}_${TAG}

CK=$REPO/outputs/value_train/value_${TASK}_${TAG}/checkpoints/$STEP
PY=$PY_SIM
NAME=simrecap_${TASK}_${TAG}
LOCAL_FINAL=$REPO/lerobot_dataset/$NAME
LOGDIR=${PUBLISH_LOGDIR:-$WORK}

[ -d "$CK" ] || { log "错误: checkpoint 不存在 $CK"; exit 1; }
[ -n "$(pids_by_cmd lerobot_value_train)" ] && { log "错误: 价值训练仍在运行(会读 merged_v30),先 stop.sh lerobot_value_train"; exit 1; }
"$PY_SIM" - "$WORK/merged_v30" "$TAG" <<'PYEOF' || exit 1
import sys, glob, pyarrow.parquet as pq
d, tag = sys.argv[1], sys.argv[2]
ep = sorted(glob.glob(f"{d}/meta/episodes/**/*.parquet", recursive=True)); da = sorted(glob.glob(f"{d}/data/**/*.parquet", recursive=True))
if not ep or not da: sys.exit("错误: merged_v30 不完整")
if "episode_success" not in pq.read_schema(ep[0]).names: sys.exit("错误: merged_v30 没有 episode_success 列,先跑 03_build_pool.sh")
if f"complementary_info.value_{tag}" in pq.read_schema(da[0]).names: sys.exit(f"错误: merged_v30 已含 _{tag} 列(上次发布残留),换 tag 或重建 merged_v30")
print("  发布前检查通过")
PYEOF

strip_meta(){
"$PY" - "$1" <<'PYEOF'
import sys, glob
import pyarrow as pa, pyarrow.parquet as pq
d = sys.argv[1]; n = 0
for f in sorted(glob.glob(f"{d}/meta/**/*.parquet", recursive=True)) + sorted(glob.glob(f"{d}/data/**/*.parquet", recursive=True)):
    t = pq.read_table(f)
    if not (t.schema.metadata or any(fl.metadata for fl in t.schema)): continue
    t2 = pa.Table.from_arrays([t.column(i) for i in range(t.num_columns)], schema=pa.schema([pa.field(fl.name, fl.type, fl.nullable) for fl in t.schema]))
    pq.write_table(t2, f, compression="snappy"); n += 1
print(f"  剥离元数据: {n} 个文件")
PYEOF
}
export_sidecar(){
"$PY" - "$1" "$2" <<'PYEOF'
import sys, glob, json
import pyarrow.parquet as pq
src, out = sys.argv[1], sys.argv[2]; recs = []
for f in sorted(glob.glob(f"{src}/meta/episodes/**/*.parquet", recursive=True)):
    t = pq.read_table(f, columns=["episode_index", "episode_success"])
    recs += [{"episode_index": int(i), "success": s == "success"} for i, s in zip(t.column(0).to_pylist(), t.column(1).to_pylist())]
recs.sort(key=lambda r: r["episode_index"])
json.dump({"labels_field": "episode_success", "saved_episode_count": len(recs), "episodes": recs}, open(out, "w"), indent=2)
print(f"  边车: {len(recs)} 集, 成功 {sum(r['success'] for r in recs)}")
PYEOF
}
publish(){  # $1=WORK 下的 v3.0 目录名  $2=NAS 容器  $3=是否要求三列(1/0)
    local src=$WORK/$1 dst=$2/$NAME
    log "转换 $1 -> v2.1"; strip_meta "$src" || return 1; export_sidecar "$src" "$WORK/$1.episode_success.json" || return 1
    "$PY" "$REPO/scripts/convert_lerobot3.0_to_2.1.py" --repo-id "$1" --root "$WORK" || { log "错误: $1 转换失败"; return 1; }
    rm -rf "$WORK/${1}_v3.0"; cp "$WORK/$1.episode_success.json" "$src/episode_success.json" || return 1
    if [ "$3" = 1 ]; then
        "$PY" - "$src" "$TAG" <<'PYEOF' || return 1
import sys, glob, pyarrow.parquet as pq
src, tag = sys.argv[1], sys.argv[2]
cols = pq.read_schema(sorted(glob.glob(f"{src}/data/**/*.parquet", recursive=True))[0]).names
miss = [c for c in (f"complementary_info.value_{tag}", f"complementary_info.advantage_{tag}", f"complementary_info.acp_indicator_{tag}") if c not in cols]
sys.exit(f"错误: v2.1 丢失列 {miss}") if miss else print("  三列存活确认")
PYEOF
    fi
    log "同步到 NAS: $dst"; mkdir -p "$dst" && rsync -a --delete "$src/" "$dst/" || { log "错误: rsync 失败"; return 1; }
    "$PY" -c "import json; i=json.load(open('$dst/meta/info.json')); print(f'  NAS 就绪: {i[\"codebase_version\"]} {i[\"total_episodes\"]} 集')"
}

log "1) no_reward"; rm -rf "$WORK/no_reward_v30"; cp -r "$WORK/merged_v30" "$WORK/no_reward_v30"
publish no_reward_v30 "$NAS/recap_no_reward_dataset" 0 || exit 1
log "2) 阶段 6: checkpoint $STEP 推理写回 merged_v30(列后缀 _$TAG)"
bash "$REPO/launch/run_value_infer.sh" "$WORK/merged_v30" "$CK" "$TAG" 0 --runtime.batch_size 32 > "$LOGDIR/value_infer_${TAG}.log" 2>&1 \
    || { log "错误: 阶段 6 失败,见 $LOGDIR/value_infer_${TAG}.log"; exit 1; }
grep -a "ACP stats" "$LOGDIR/value_infer_${TAG}.log" | tail -1
log "3) reward"; rm -rf "$WORK/reward_v30"; cp -r "$WORK/merged_v30" "$WORK/reward_v30"
publish reward_v30 "$NAS/recap_reward_dataset" 1 || exit 1
if [ -e "$LOCAL_FINAL" ]; then log "本地 $LOCAL_FINAL 已存在,跳过本地发布"; else
    cp -r "$WORK/reward_v30" "$LOCAL_FINAL"; mkdir -p "$REPO/policy/pi05/training_data/RoboSynChallenge"
    ln -sfn "$LOCAL_FINAL" "$REPO/policy/pi05/training_data/RoboSynChallenge/$NAME"; log "本地发布: $LOCAL_FINAL(已链进 pi05 训练目录)"
fi
log "全部完成"
