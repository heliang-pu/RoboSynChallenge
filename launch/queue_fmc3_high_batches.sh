#!/usr/bin/env bash
# Queue high-numbered seeded batches until the existing GPU benchmark is done.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/fmc3/workspace/coverage/RoboSynChallenge}"
PYTHON_BIN="${PYTHON_BIN:-/home/fmc3/workspace/RoboSynChallenge/.venv/bin/python}"
NAS_ROOT="${NAS_ROOT:-/home/fmc3/FermiBotNas/dataset/RoboSynChallenge/Syn/seeded_1000_20260827}"
BATCH_IDS="${BATCH_IDS:-9 8 7 6 5}"
QUIET_CHECKS="${QUIET_CHECKS:-60}"
CHECK_INTERVAL="${CHECK_INTERVAL:-5}"
START_INTERVAL="${START_INTERVAL:-120}"
BUSY_PATTERN='benchmark_named_yolo|aggregate_named_yolo|yolo_gpu_guard|candidate_v2_latency_3x3'

quiet=0
while [ "$quiet" -lt "$QUIET_CHECKS" ]; do
    if pgrep -af "$BUSY_PATTERN" | grep -v queue_fmc3_high_batches >/dev/null; then
        quiet=0
        echo "[queue] $(date -Is) existing GPU benchmark active; waiting"
    else
        quiet=$((quiet + 1))
        echo "[queue] $(date -Is) GPU benchmark quiet check $quiet/$QUIET_CHECKS"
    fi
    sleep "$CHECK_INTERVAL"
done

cd "$REPO_ROOT"
for task in manipulate_pipette table_rearrangement water_pouring; do
    unit="robosyn-high-${task//_/-}.service"
    systemctl --user stop "$unit" 2>/dev/null || true
    systemctl --user reset-failed "$unit" 2>/dev/null || true
    systemd-run --user --unit="$unit" --description="RoboSyn high batches $task" \
        --property=Restart=on-failure --property=RestartSec=60s \
        --working-directory="$REPO_ROOT" \
        /usr/bin/env PYTHON_BIN="$PYTHON_BIN" NAS_ROOT="$NAS_ROOT" BATCH_IDS="$BATCH_IDS" \
        /bin/bash launch/collect_seeded_1000.sh "$task"
    echo "[queue] $(date -Is) started $unit"
    sleep "$START_INTERVAL"
done
