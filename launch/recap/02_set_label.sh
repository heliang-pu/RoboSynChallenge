#!/usr/bin/env bash
# 人工复核后改某集成败标签,三处边车同步(v2.1 交付版 / rollout_v30 / 对应分片)
# 用法: 02_set_label.sh <task> <tag> <episode_index> <success|failure>
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}; EP=${3:?episode_index}; LABEL=${4:?success|failure}
[[ "$LABEL" == success || "$LABEL" == failure ]] || die "标签只能是 success 或 failure"
WORK=$WORK_ROOT/${TASK}_${TAG}
"$PY_SIM" - "$WORK" "$REPO/lerobot_dataset/rollouts/${TASK}_${TAG}" "$EP" "$LABEL" <<'PYEOF'
import sys, json, glob, os
work, deliver, ep, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "success"
targets = [f"{deliver}/episode_success.json", f"{work}/rollout_v30/rollout_merged/episode_success.json"]
# 分片:按顺序累计集数定位
off = 0
for d in sorted(glob.glob(f"{work}/shards/s*/*/episode_success.json"), key=lambda q: int(q.split("/shards/s")[1].split("/")[0])):
    n = len(json.load(open(d))["episodes"])
    if off <= ep < off + n: targets.append((d, ep - off)); break
    off += n
for t in targets:
    path, idx = (t, ep) if isinstance(t, str) else t
    if not os.path.exists(path): print(f"  跳过(不存在): {path}"); continue
    sc = json.load(open(path)); recs = [r for r in sc["episodes"] if r["episode_index"] == idx]
    if not recs: print(f"  未找到 episode {idx}: {path}"); continue
    old = recs[0]["success"]; recs[0]["success"] = label
    json.dump(sc, open(path, "w"), indent=2)
    n_s = sum(r["success"] for r in sc["episodes"])
    print(f"  {path.split('lerobot_dataset/')[-1]}: ep{idx} {old} -> {label}  (现 {n_s}/{len(sc['episodes'])} 成功)")
PYEOF
log "注意: 若已跑过 03_build_pool.sh,需重跑它让合并池标签更新"
