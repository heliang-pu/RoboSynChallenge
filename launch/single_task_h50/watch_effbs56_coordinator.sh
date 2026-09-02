#!/usr/bin/env bash
set -uo pipefail

session=${ROBOSYN_COORDINATOR_SESSION:-pi05-single-task-h50-effbs56-frozenvlm-v6-coordinator}
script=${ROBOSYN_COORDINATOR_SCRIPT:-$HOME/workspace/RoboSynChallenge/launch/single_task_h50/orchestrate_effbs56_after_upload.sh}
nas_root=${ROBOSYN_SINGLE_NAS_ROOT:-$HOME/FermiBotNas/models/RoboSynChallenge/pi05_single_task_h50_from_all10_67500_merged_v21_effbs56_frozenvlm_v6_2ep}
poll=${ROBOSYN_WATCHDOG_POLL_SECONDS:-120}
log_file="$nas_root/_orchestrator/watchdog.log"

mkdir -p "$nas_root/_orchestrator"
exec >>"$log_file" 2>&1
echo "[$(date -Is)] effective-BS56 watchdog started"
while ! test -f "$nas_root/PIPELINE_DONE"; do
    if test -f "$nas_root/PIPELINE_FAILED"; then
        echo "[$(date -Is)] pipeline failure marker found; stopping for review"
        exit 1
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
        echo "[$(date -Is)] coordinator absent; restarting"
        tmux new-session -d -s "$session" \
            "env ROBOSYN_DCU_REMOTE='$ROBOSYN_DCU_REMOTE' ROBOSYN_DCU_PORT='${ROBOSYN_DCU_PORT:-22}' '$script'"
    fi
    sleep "$poll"
done
echo "[$(date -Is)] pipeline complete; watchdog exiting"
