#!/usr/bin/env bash
# 阶段 2-4:专家数据准备(v2.1→v3.0,按指纹缓存)→ 可选取子集 → 与 rollout 合并 → 写 episode_success
# 用法: 03_build_pool.sh <task> <tag> <expert_dataset_dir> [expert_episodes=all]   expert_episodes: all | N(前 N 集) | A:B(第 A..B-1 集,如 200:260 取 held-out)
# 例:   03_build_pool.sh sample_loading round1 "$NAS_ROOT"/dataset/RoboSynChallenge/Sim_clean_filtered/cobotmagic_Sim_sample_loading 200
# 说明: 专家用清洗版(Sim_clean_filtered);expert_episodes 取前 N 集控制专家:rollout 比例(round1 用了 200:150)。
#       专家目录若自带 episode_success.json(上一轮发布的合并池),标签按边车恢复,不会把失败集错标成功。
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}; EXPERT=${3:?expert_dataset_dir}; NSUB=${4:-all}
[[ $NSUB == all || $NSUB =~ ^[0-9]+$ || $NSUB =~ ^[0-9]+:[0-9]+$ ]] || die "expert_episodes 必须是 all / 整数 / A:B: $NSUB"
if [[ $NSUB =~ ^([0-9]+):([0-9]+)$ ]]; then SUB_A=${BASH_REMATCH[1]}; SUB_B=${BASH_REMATCH[2]}; [ "$SUB_B" -gt "$SUB_A" ] || die "A:B 需要 B>A"; else SUB_A=0; SUB_B=$NSUB; fi
WORK=$WORK_ROOT/${TASK}_${TAG}; ROLL=$WORK/rollout_v30/rollout_merged
need_file "$ROLL/meta/info.json"; need_file "$ROLL/episode_success.json"; need_file "$EXPERT/meta/info.json"
[ -e "$WORK/merged_v30" ] && die "$WORK/merged_v30 已存在;删除后重跑"

# 守卫:专家源若含失败集(上一轮合并池)却没有边车,拒绝——否则会被整体标成 success
"$PY_SIM" - "$EXPERT" <<'PYEOF' || die "专家源含 failure 标签但缺 episode_success.json 边车(上轮发布目录里有,复制过来)"
import sys, os, glob, json
d = sys.argv[1]
if os.path.exists(f"{d}/episode_success.json"): sys.exit(0)
bad = False
for f in glob.glob(f"{d}/meta/episodes.jsonl"):
    bad |= any('"failure"' in l for l in open(f))
for f in glob.glob(f"{d}/meta/episodes/**/*.parquet", recursive=True):
    import pyarrow.parquet as pq
    t = pq.read_table(f); bad |= ("episode_success" in t.column_names and "failure" in t.column("episode_success").to_pylist())
sys.exit(1 if bad else 0)
PYEOF

# --- 专家 v3.0(缓存) ---
ver=$("$PY_SIM" -c "import json;print(json.load(open('$EXPERT/meta/info.json')).get('codebase_version',''))")
case "$ver" in
  v2.1)
    fp=$("$PY_SIM" -c "import hashlib,os;p=os.path.realpath('$EXPERT');print(hashlib.sha1((p+open(p+'/meta/info.json').read()).encode()).hexdigest()[:16])")
    CACHE=$WORK_ROOT/_cache/expert_v30_$fp
    if [ -f "$CACHE/.fingerprint" ]; then log "专家缓存命中: $CACHE"; else
        log "专家 v2.1 -> v3.0(首次,复制后原地上转,原数据不动)"; rm -rf "$CACHE"; mkdir -p "$WORK_ROOT/_cache"; cp -r "$EXPERT" "$CACHE"
        "$PY_SIM" -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id "expert_v30_$fp" --root "$WORK_ROOT/_cache" --push-to-hub false || die "上转失败"
        [ -f "$EXPERT/episode_success.json" ] && cp "$EXPERT/episode_success.json" "$CACHE/"; echo "$fp" > "$CACHE/.fingerprint"; fi
    EXP_V30=$CACHE ;;
  v3*)
    fp=$("$PY_SIM" -c "import hashlib,os;p=os.path.realpath('$EXPERT');print(hashlib.sha1((p+open(p+'/meta/info.json').read()).encode()).hexdigest()[:16])")
    CACHE=$WORK_ROOT/_cache/expert_v30_$fp
    if [ -f "$CACHE/.fingerprint" ]; then log "专家缓存命中: $CACHE"; else
        log "专家 v3.0 复制进缓存(不原地改写源目录)"; rm -rf "$CACHE"; mkdir -p "$WORK_ROOT/_cache"; cp -r "$EXPERT" "$CACHE"; echo "$fp" > "$CACHE/.fingerprint"; fi
    EXP_V30=$CACHE ;;
  *) die "无法识别专家数据版本 '$ver'" ;;
esac
[ -f "$EXPERT/episode_success.json" ] && cp "$EXPERT/episode_success.json" "$EXP_V30/episode_success.json"
# 上一轮发布的 reward 池带 complementary_info.* 三列,新 rollout 没有 → 合并会因 features 不同失败;去掉这些列
if "$PY_SIM" -c "import json,sys;f=json.load(open('$EXP_V30/meta/info.json'))['features'];sys.exit(0 if any(k.startswith('complementary_info.') for k in f) else 1)"; then
    CLEAN=${EXP_V30}_noci
    if [ ! -f "$CLEAN/.fingerprint" ]; then
        log "去掉专家池中的 complementary_info.* 列 -> $(basename "$CLEAN")"; rm -rf "$CLEAN"
        PYTHONPATH="$EVO_SRC" "$PY_EVO" - "$EXP_V30" "$CLEAN" <<'PYEOF' || die "去列失败"
import sys, json, shutil, os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import remove_feature
src, out = sys.argv[1], sys.argv[2]
feats = [k for k in json.load(open(f"{src}/meta/info.json"))["features"] if k.startswith("complementary_info.")]
ds = LeRobotDataset("local/expert", root=src)
remove_feature(ds, feature_names=feats, output_dir=out)
print("  已去列:", feats)
PYEOF
        [ -f "$EXP_V30/episode_success.json" ] && cp "$EXP_V30/episode_success.json" "$CLEAN/"; echo ok > "$CLEAN/.fingerprint"
    else log "去列缓存命中: $CLEAN"; [ -f "$EXP_V30/episode_success.json" ] && cp "$EXP_V30/episode_success.json" "$CLEAN/"; fi
    EXP_V30=$CLEAN
fi
ntot=$("$PY_SIM" -c "import json;print(json.load(open('$EXP_V30/meta/info.json'))['total_episodes'])")

# --- 可选子集(前 N 集或 A:B 区间) ---
rm -rf "$WORK/expert_v30" "$WORK/expert_v30_splits"
if [ "$NSUB" != all ] && { [ "$SUB_A" -gt 0 ] || [ "$SUB_B" -lt "$ntot" ]; }; then
    [ "$SUB_B" -le "$ntot" ] || die "区间上界 $SUB_B 超过专家集数 $ntot"
    log "专家取第 $SUB_A..$((SUB_B-1)) 集(共 $((SUB_B-SUB_A))/$ntot)-> expert_v30"
    PYTHONPATH="$EVO_SRC" "$PY_EVO" - "$EXP_V30" "$WORK/expert_v30" "$SUB_A" "$SUB_B" <<'PYEOF' || die "取子集失败"
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import split_dataset
src, out, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ds = LeRobotDataset("local/expert", root=src)
split_dataset(ds, splits={"sub": list(range(a, b))}, output_dir=out + "_splits")
import shutil, os; shutil.move(out + "_splits/sub", out); shutil.rmtree(out + "_splits", ignore_errors=True)
print("  子集完成")
PYEOF
    if [ -f "$EXP_V30/episode_success.json" ]; then "$PY_SIM" -c "
import json; sc=json.load(open('$EXP_V30/episode_success.json')); sc['episodes']=[dict(e, episode_index=e['episode_index']-$SUB_A) for e in sc['episodes'] if $SUB_A<=e['episode_index']<$SUB_B]; sc['saved_episode_count']=len(sc['episodes']); json.dump(sc,open('$WORK/expert_v30/episode_success.json','w'),indent=2)"; fi
    NEXP=$((SUB_B-SUB_A))
else ln -sfn "$EXP_V30" "$WORK/expert_v30"; NEXP=$ntot; fi

# --- 合并 [专家, rollout] ---
strip_meta "$(readlink -f "$WORK/expert_v30")"; strip_meta "$ROLL"
log "合并 expert_v30($NEXP) + rollout($(basename "$ROLL")) -> merged_v30"
merge_v30 "$WORK" merged_v30 expert_v30 "rollout_v30/rollout_merged" || die "合并失败"

# --- 打标 ---
args=(--dataset "$WORK/merged_v30" --sidecar "$ROLL/episode_success.json" --prefix-success "$NEXP")
[ -f "$WORK/expert_v30/episode_success.json" ] && args+=(--prefix-sidecar "$WORK/expert_v30/episode_success.json")
"$PY_SIM" "$REPO/scripts/label_rollout_dataset.py" "${args[@]}" || die "打标失败"
log "完成: $WORK/merged_v30  |  下一步: 04_value_train.sh $TASK $TAG"
