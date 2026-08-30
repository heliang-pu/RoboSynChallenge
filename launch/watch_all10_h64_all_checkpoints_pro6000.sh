#!/usr/bin/env bash
# Relaunch the resumable PRO-6000 sweep if its tmux session exits unexpectedly.
set -uo pipefail

repo="${ALL10_EVAL_REPO:-/workspace/shared/RoboSynChallenge-eval-all10}"
session="${ALL10_EVAL_SESSION:-pi05-all10-eval-allckpts}"
results_root="${ALL10_EVAL_RESULTS_ROOT:-/mnt/FermiBotNas/models/RoboSynChallenge/pi05_all10_h64_expert/all10_expert_base_h64_bs64_steps100k_eval_all_ckpts_h10_30_50_64_20eps_4view}"
workers="${ALL10_EVAL_WORKERS:-2}"
xla_mem_fraction="${ALL10_EVAL_XLA_MEM_FRACTION:-0.35}"
poll_seconds="${ALL10_EVAL_WATCH_POLL_SECONDS:-60}"

mkdir -p "$results_root"

while [[ ! -f "$results_root/.complete" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
        printf '[%s] session=%s missing; restarting resumable sweep\n' \
            "$(date -Is)" "$session" >>"$results_root/watchdog.log"
        tmux new-session -d -s "$session" \
            "env ALL10_EVAL_WORKERS=$workers ALL10_EVAL_XLA_MEM_FRACTION=$xla_mem_fraction bash $repo/launch/eval_all10_h64_all_checkpoints_pro6000.sh"
    fi
    sleep "$poll_seconds"
done

printf '[%s] sweep complete; watchdog exiting\n' "$(date -Is)" >>"$results_root/watchdog.log"
