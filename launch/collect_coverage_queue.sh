#!/usr/bin/env bash
# 覆盖补采总调度:按 PLAN.json 排队,N 个 worker 并行,每组走 collect_until_valid(隔离-验证-晋升)
# 用法: collect_coverage_queue.sh <plan.json> <task 过滤,逗号分隔或 all> <workers> [extra run_env args...]
# 断点续采:已产出 .validated 标记的组自动跳过;每组 SEED_MASTER 由"任务+组名"哈希派生,重启不换种子流。
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PLAN=${1:?plan.json}; FILTER=${2:?task 过滤}; WORKERS=${3:?workers}; shift 3; EXTRA=("$@")
cd "$REPO_ROOT"
QUEUE=$(mktemp); LOCK="$QUEUE.lock"
python3 - "$PLAN" "$FILTER" > "$QUEUE" <<'PYEOF'
import json, sys
plan = json.load(open(sys.argv[1]))["tasks"]
flt = None if sys.argv[2] == "all" else set(sys.argv[2].split(","))
for task, spec in plan.items():
    if flt and task not in flt: continue
    for c in spec["configs"]:
        print(f"{task}|{c['name']}|{c['episodes']}")
PYEOF
total=$(wc -l < "$QUEUE"); echo "[queue] 共 $total 组作业, $WORKERS 个 worker"
worker() {
    local wid=$1
    while true; do
        local job
        job=$(flock "$LOCK" bash -c "head -1 '$QUEUE' && sed -i 1d '$QUEUE'")
        [ -n "$job" ] || break
        local task=${job%%|*}; local rest=${job#*|}; local slug=${rest%%|*}; local eps=${rest##*|}
        local marker="lerobot_dataset/coverage/$task/$slug/.validated"
        if [ -f "$marker" ]; then echo "[w$wid] 跳过已完成: $task/$slug"; continue; fi
        if [ -f "lerobot_dataset/coverage/$task/$slug/.infeasible" ]; then echo "[w$wid] 跳过不可行区: $task/$slug"; continue; fi
        local seed_master=$(python3 -c "import zlib;print(zlib.crc32('$task/$slug'.encode())%900000+100000)")
        mkdir -p "lerobot_dataset/coverage/$task" ".launch/coverage_logs"
        local log=".launch/coverage_logs/${task}_${slug}.log"
        echo "[w$wid] $(date +%H:%M) 开始 $task/$slug ($eps 集, SEED_MASTER=$seed_master) -> $log"
        local cap=$(( eps * 8 > 200 ? eps * 8 : 200 ))
        if SEED_MASTER=$seed_master MAX_QUEUE_ATTEMPTS=2 bash launch/collect_until_valid.sh "$task" "$slug" "$eps" 3_0 \
              --report_task_success --save_only_success --success_settle_steps 75 \
              --max_generation_attempts "$cap" "${EXTRA[@]}" > "$log" 2>&1; then
            local ds=$(sed -n 's/^VALIDATED_DATASET=//p' "$log" | tail -1)
            mkdir -p "$(dirname "$marker")"; echo "$ds" > "$marker"
            echo "[w$wid] $(date +%H:%M) ✅ $task/$slug 完成: $ds"
        else
            mkdir -p "lerobot_dataset/coverage/$task/$slug"
            date > "lerobot_dataset/coverage/$task/$slug/.infeasible"
            echo "[w$wid] $(date +%H:%M) ❌ $task/$slug 失败(标记 .infeasible,见 $log),继续下一组"
        fi
    done
}
pids=()
for i in $(seq 1 "$WORKERS"); do worker "$i" & pids+=($!); sleep 45; done
for p in "${pids[@]}"; do wait "$p"; done
echo "[queue] 全部作业处理完毕"
