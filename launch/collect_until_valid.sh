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
while [ "$stop_requested" -eq 0 ]; do
    attempt=$((attempt + 1))
    echo "[queue] starting isolated candidate attempt $attempt"
    if "$SCRIPT_DIR/collect_validated_batch.sh" "$@"; then
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
