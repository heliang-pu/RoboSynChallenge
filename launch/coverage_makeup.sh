#!/usr/bin/env bash
# 方案 A 配额转移:主队列全部结束后,统计各任务净成功缺口,把缺口转给该任务实测成功率最高的 strat/可行组补采
# 用法: coverage_makeup.sh <plan.json> <task 过滤/all> <workers> [queue_log 路径]
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PLAN=${1:?plan}; FILTER=${2:?filter}; WORKERS=${3:?workers}; QLOG=${4:-.launch/coverage_queue.log}
cd "$REPO_ROOT"
log(){ echo "[makeup] $(date +%H:%M:%S) $*"; }
log "等待主队列结束 ($QLOG)"
until grep -q "全部作业处理完毕" "$QLOG" 2>/dev/null; do sleep 300; done
python3 - "$PLAN" "$FILTER" > /tmp/makeup_jobs.$$ <<'PYEOF'
import json, sys, glob, os
plan = json.load(open(sys.argv[1]))["tasks"]
flt = None if sys.argv[2] == "all" else set(sys.argv[2].split(","))
for task, spec in plan.items():
    if flt and task not in flt: continue
    target = sum(c["episodes"] for c in spec["configs"])
    got = 0; rates = []
    for c in spec["configs"]:
        d = f"lerobot_dataset/coverage/{task}/{c['name']}"
        sc = os.path.join(d, ".validated")
        n = 0
        for info in glob.glob(f"{d}/*/meta/info.json"):
            try: n = max(n, json.load(open(info))["total_episodes"])
            except Exception: pass
        got += n
        # 实测产率代理:已采到的越接近目标,认为该组越可行
        rates.append((n / max(1, c["episodes"]), c["name"], c["episodes"]))
    deficit = target - got
    if deficit <= 0: continue
    rates.sort(reverse=True)
    best = [r for r in rates if r[0] >= 0.5] or rates[:1]
    # 缺口平摊给最可行的组(最多 3 个)
    best = best[:3]
    share = -(-deficit // len(best))
    for _, name, _ in best:
        print(f"{task}|{name}|{min(share, deficit)}|makeup")
        deficit -= share
        if deficit <= 0: break
PYEOF
n=$(wc -l < /tmp/makeup_jobs.$$)
log "补采作业 $n 个:"; cat /tmp/makeup_jobs.$$
while IFS='|' read -r task slug eps _; do
    [ -n "$task" ] || continue
    seed_master=$(python3 -c "import zlib;print(zlib.crc32('$task/$slug/makeup'.encode())%900000+100000)")
    logf=".launch/coverage_logs/${task}_${slug}_makeup.log"
    cap=$(( eps * 8 > 200 ? eps * 8 : 200 ))
    log "补采 $task/$slug +$eps (SEED_MASTER=$seed_master)"
    SEED_MASTER=$seed_master MAX_QUEUE_ATTEMPTS=2 bash launch/collect_until_valid.sh "$task" "$slug" "$eps" 3_0 \
        --report_task_success --save_only_success --success_settle_steps 75 \
        --max_generation_attempts "$cap" > "$logf" 2>&1 \
      && { ds=$(sed -n 's/^VALIDATED_DATASET=//p' "$logf" | tail -1); mkdir -p "lerobot_dataset/coverage/$task/${slug}_makeup"; echo "$ds" > "lerobot_dataset/coverage/$task/${slug}_makeup/.validated"; log "✅ $task/$slug 补采完成"; } \
      || log "❌ $task/$slug 补采失败"
done < /tmp/makeup_jobs.$$
rm -f /tmp/makeup_jobs.$$
log "补采阶段结束"
