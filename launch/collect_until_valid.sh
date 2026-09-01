#!/usr/bin/env bash

# Keep producing isolated candidates until one complete batch passes every
# validation gate. Failed candidates are never merged or reused.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-10}

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <task> <setting> <episodes> <format> [extra run_task args...]" >&2
    exit 2
fi

stop_requested=0
trap 'stop_requested=1' INT TERM

attempt=0
MAX_QUEUE_ATTEMPTS=${MAX_QUEUE_ATTEMPTS:-0}   # >0 时:整批重试这么多次仍失败就放弃(配合 run_env 熔断用)
while [ "$stop_requested" -eq 0 ]; do
    if [ "$MAX_QUEUE_ATTEMPTS" -gt 0 ] && [ "$attempt" -ge "$MAX_QUEUE_ATTEMPTS" ]; then
        echo "[queue] giving up after $attempt failed candidates" >&2
        exit 42
    fi
    attempt=$((attempt + 1))
    echo "[queue] starting isolated candidate attempt $attempt"
    # SEED_MASTER 设置时:每次尝试换一个派生主种子(同种子重试只会复现同样的失败)
    seed_args=()
    [ -n "${SEED_MASTER:-}" ] && seed_args=(--seed $((SEED_MASTER + attempt - 1)))
    if "$SCRIPT_DIR/collect_validated_batch.sh" "$@" "${seed_args[@]}"; then
        echo "[queue] attempt $attempt passed all gates"
        exit 0
    else
        status=$?
        echo "[queue] attempt $attempt rejected with status $status; candidate remains isolated" >&2
    fi

    if [ "$stop_requested" -eq 0 ]; then
        echo "[queue] retrying in ${RETRY_DELAY_SECONDS}s"
        sleep "$RETRY_DELAY_SECONDS" &
        wait $!
    fi
done

echo "[queue] stopped by signal" >&2
exit 130
