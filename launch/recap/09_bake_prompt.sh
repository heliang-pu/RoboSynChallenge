#!/usr/bin/env bash
# 把 acp_indicator 烤进 task 文本(per-frame task_index → "…\nAdvantage: positive/negative"),
# 供不走 openpi ACPAdvantageTag 的 VLA 框架直接按 task 训练。非破坏性:输出到新目录 + NAS。
# 用法: 09_bake_prompt.sh <task> <tag>          例: 09_bake_prompt.sh sample_loading round1
# 依赖: 本地已有 lerobot_dataset/simrecap_<task>_<tag>(reward 版,带 acp_indicator_<tag> 列)
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}
SRC=$REPO/lerobot_dataset/simrecap_${TASK}_${TAG}
DST=$REPO/lerobot_dataset/simrecap_${TASK}_${TAG}_prompt
NAS_DST=$NAS/recap_reward_dataset/simrecap_${TASK}_${TAG}_prompt
need_file "$SRC/meta/info.json"
[ -e "$DST" ] && die "$DST 已存在;删除后重跑"
log "复制 $SRC -> $DST"; cp -r "$SRC" "$DST"
"$PY_SIM" - "$DST" "$TAG" <<'PYEOF' || die "烤 prompt 失败"
import sys, json, glob, numpy as np, pyarrow as pa, pyarrow.parquet as pq
d, tag = sys.argv[1], sys.argv[2]
ind_col = f"complementary_info.acp_indicator_{tag}"
base = json.loads(open(f"{d}/meta/tasks.jsonl").readline())["task"]
# 已经烤过就拒绝(base 里已含 Advantage)
if "Advantage:" in base: sys.exit("源数据集的 task 已含 Advantage,疑似已烤过")
POS, NEG = f"{base}\nAdvantage: positive", f"{base}\nAdvantage: negative"
# 新任务表:0=positive, 1=negative
open(f"{d}/meta/tasks.jsonl","w").write(json.dumps({"task_index":0,"task":POS})+"\n"+json.dumps({"task_index":1,"task":NEG})+"\n")
# 逐 episode 改 data parquet 的 task_index,并重算 episodes.jsonl / episodes_stats.jsonl 的 task_index
eps_meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(f"{d}/meta/episodes.jsonl")}
estats = {json.loads(l)["episode_index"]: json.loads(l) for l in open(f"{d}/meta/episodes_stats.jsonl")}
n_pos = n_tot = 0
for f in sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet")):
    t = pq.read_table(f)
    ind = np.asarray(t.column(ind_col).to_pylist()).reshape(-1).astype(int)
    ti = np.where(ind == 1, 0, 1).astype(np.int64)          # positive->0, negative->1
    ep = int(t.column("episode_index").to_pylist()[0])
    cols = t.column_names; arrs = [t.column(c) for c in cols]
    ft = t.schema.field("task_index").type
    is_list = pa.types.is_list(ft) or pa.types.is_fixed_size_list(ft)
    vals = [[int(x)] for x in ti] if is_list else [int(x) for x in ti]
    arrs[cols.index("task_index")] = pa.array(vals, type=ft)
    pq.write_table(pa.table(arrs, names=cols), f + ".tmp", compression="snappy"); __import__("os").replace(f + ".tmp", f)
    n_pos += int((ind==1).sum()); n_tot += len(ind)
    used = sorted(set(ti.tolist()))
    eps_meta[ep]["tasks"] = [POS if i==0 else NEG for i in used]
    s = estats[ep]["stats"]["task_index"]
    s["min"], s["max"], s["mean"], s["std"], s["count"] = [int(ti.min())], [int(ti.max())], [float(ti.mean())], [float(ti.std())], [len(ti)]
    for q,v in zip(("q01","q10","q50","q90","q99"),np.percentile(ti,[1,10,50,90,99])): s[q]=[float(v)]
with open(f"{d}/meta/episodes.jsonl","w") as w:
    for ep in sorted(eps_meta): w.write(json.dumps(eps_meta[ep])+"\n")
with open(f"{d}/meta/episodes_stats.jsonl","w") as w:
    for ep in sorted(estats): w.write(json.dumps(estats[ep])+"\n")
info = json.load(open(f"{d}/meta/info.json")); info["total_tasks"] = 2; json.dump(info, open(f"{d}/meta/info.json","w"), indent=4)
print(f"  烤好: {n_pos}/{n_tot} 帧 positive ({100*n_pos/n_tot:.1f}%), 其余 negative")
PYEOF
log "校验(pi05 环境训练读取门)"
"$REPO/policy/pi05/.venv/bin/python" "$REPO/scripts/validate_lerobot_dataset.py" "$DST" \
    --expected-episodes "$("$PY_SIM" -c "import json;print(json.load(open('$DST/meta/info.json'))['total_episodes'])")" \
    --producer-exit-code 0 >/dev/null 2>&1 && log "校验门全过" || log "警告: 校验未过,手动查"
log "同步到 NAS: $NAS_DST"; mkdir -p "$NAS_DST" && rsync -a --delete "$DST/" "$NAS_DST/"
log "完成: 本地 $DST  |  NAS $NAS_DST"
