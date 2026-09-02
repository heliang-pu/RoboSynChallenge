#!/usr/bin/env bash
# Keep the local coordinator alive across transient DNS/SSH/NAS failures.
set -uo pipefail

session=${ROBOSYN_COORDINATOR_SESSION:-pi05-single-task-h50-coordinator}
script=${ROBOSYN_COORDINATOR_SCRIPT:-$HOME/workspace/RoboSynChallenge/launch/single_task_h50/orchestrate_after_merge.sh}
nas_root=${ROBOSYN_SINGLE_NAS_ROOT:-$HOME/FermiBotNas/models/RoboSynChallenge/pi05_single_task_h50_from_all10_67500_merged_v21}
poll=${ROBOSYN_WATCHDOG_POLL_SECONDS:-120}
log_file="$nas_root/_orchestrator/watchdog.log"

mkdir -p "$nas_root/_orchestrator"
exec >>"$log_file" 2>&1

echo "[$(date -Is)] coordinator watchdog started"
while ! test -f "$nas_root/PIPELINE_DONE"; do
    if test -f "$nas_root/PIPELINE_FAILED"; then
        echo "[$(date -Is)] pipeline has a failure marker; watchdog exiting for operator review"
        exit 1
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
        echo "[$(date -Is)] coordinator is absent; restarting"
        tmux new-session -d -s "$session" "$script"
    fi
    sleep "$poll"
done
echo "[$(date -Is)] pipeline complete; watchdog exiting"
