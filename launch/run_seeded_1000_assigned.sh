#!/usr/bin/env bash
# Persistently run one seeded collector per assigned task.  Optionally wait for
# an incompatible GPU job (for example one using --kill-competitors) first.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WAIT_PROCESS_PATTERN=${WAIT_PROCESS_PATTERN:-}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-60}
RESTART_DELAY_SECONDS=${RESTART_DELAY_SECONDS:-60}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [task ...]" >&2
    exit 2
fi

cd "$REPO_ROOT"
mkdir -p .launch

if [ -n "$WAIT_PROCESS_PATTERN" ]; then
    while pgrep -f "$WAIT_PROCESS_PATTERN" >/dev/null; do
        echo "[scheduler] $(date -Is) waiting for process: $WAIT_PROCESS_PATTERN"
        sleep "$WAIT_POLL_SECONDS"
    done
fi

echo "[scheduler] $(date -Is) starting assigned tasks: $*"

worker() {
    local task=$1 status
    while true; do
        bash "$SCRIPT_DIR/collect_seeded_1000.sh" "$task"
        status=$?
        [ "$status" -eq 0 ] && return 0
        echo "[watchdog:$task] $(date -Is) exited $status; restarting in ${RESTART_DELAY_SECONDS}s"
        sleep "$RESTART_DELAY_SECONDS"
    done
}

pids=()
for task in "$@"; do
    worker "$task" > ".launch/seeded_1000_${task}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
exit "$status"
