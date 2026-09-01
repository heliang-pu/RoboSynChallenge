#!/usr/bin/env bash
# One-shot controller: pause at a complete checkpoint-200, find the largest
# stable integer per-device batch up to 256, then resume production.
set -uo pipefail

RUN_ROOT="${DM05_WORK:-/tmp/dm05}"
OUTPUT_DIR="$RUN_ROOT/user_checkpoints/dm05_sample_loading_relative_hygon_bs192"
CHECKPOINT="$OUTPUT_DIR/checkpoint-200"
START_SCRIPT="$RUN_ROOT/start_dm05_hygon.sh"
WATCH_SCRIPT="$RUN_ROOT/watch_dm05_hygon.sh"
PROBE_SCRIPT="$RUN_ROOT/policy_dm05/dm05/probe_dm05_batch_hygon.sh"
BATCH_FILE="$RUN_ROOT/production_batch_size"
LOG_FILE="$RUN_ROOT/logs/dm05_batch_autotune.log"
STOP_AT="${DM05_STOP_AT:-2026-08-26T11:00:00-04:00}"
STOP_EPOCH="$(date -d "$STOP_AT" +%s)" || exit 2
POLL_SECONDS="${DM05_AUTOTUNE_POLL_SECONDS:-30}"
PROBE_STEPS="${DM05_PROBE_STEPS:-12}"
production_paused=0

mkdir -p "$RUN_ROOT/logs"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

checkpoint_complete() {
  [[ -f "$CHECKPOINT/trainer_state.json" ]] || return 1
  [[ -f "$CHECKPOINT/norm_stats.json" ]] || return 1
  local count bytes
  read -r count bytes < <(
    find "$CHECKPOINT" -type f -printf '%s\n' \
      | awk '{n++; b+=$1} END {printf "%d %.0f\n", n, b}'
  )
  (( count >= 20 && bytes >= 80000000000 ))
}

trainer_pids() {
  pgrep -f 'dm05_sft_demo.py|torch.distributed.run.*master_port=2954[0-9]' || true
}

stop_all_trainers() {
  local signal="${1:-TERM}"
  local pids=()
  mapfile -t pids < <(trainer_pids)
  if (( ${#pids[@]} > 0 )); then
    kill "-$signal" "${pids[@]}" 2>/dev/null || true
  fi
}

wait_no_trainers() {
  local rounds="${1:-24}"
  for _ in $(seq 1 "$rounds"); do
    [[ -z "$(trainer_pids)" ]] && return 0
    sleep 5
  done
  return 1
}

start_production() {
  local selected="$1"
  local batch_tmp="$BATCH_FILE.tmp.$$"
  printf '%s\n' "$selected" > "$batch_tmp"
  mv -f -- "$batch_tmp" "$BATCH_FILE"

  if ! tmux has-session -t dm05-train 2>/dev/null; then
    tmux new-session -d -s dm05-train "bash $START_SCRIPT"
  fi
  if ! tmux has-session -t dm05-watchdog 2>/dev/null; then
    tmux new-session -d -s dm05-watchdog "bash $WATCH_SCRIPT"
  fi
  log "production resumed with batch=$selected"
}

recover_on_exit() {
  local status=$?
  trap - EXIT INT TERM
  if (( production_paused == 1 )) && (( $(date +%s) < STOP_EPOCH )); then
    log "controller exit status=$status while production paused; recovering at batch=192"
    stop_all_trainers INT
    wait_no_trainers 12 || true
    stop_all_trainers TERM
    wait_no_trainers 6 || true
    start_production 192
  fi
  exit "$status"
}
trap recover_on_exit EXIT INT TERM

if [[ ! -x "$PROBE_SCRIPT" || ! -x "$START_SCRIPT" || ! -x "$WATCH_SCRIPT" ]]; then
  log "required script missing or not executable"
  exit 3
fi

log "waiting for complete checkpoint-200"
while (( $(date +%s) < STOP_EPOCH )); do
  if checkpoint_complete; then
    log "checkpoint-200 complete; beginning batch autotune"
    break
  fi
  if ! tmux has-session -t dm05-train 2>/dev/null; then
    log "production tmux disappeared before checkpoint-200; watchdog owns recovery"
  fi
  sleep "$POLL_SECONDS"
done

if ! checkpoint_complete; then
  log "deadline reached before checkpoint-200; exiting without interruption"
  exit 0
fi

# Disable the production watchdog before intentionally pausing its tmux.
tmux kill-session -t dm05-watchdog 2>/dev/null || true
production_paused=1
tmux send-keys -t dm05-train C-c 2>/dev/null || true
for _ in $(seq 1 24); do
  tmux has-session -t dm05-train 2>/dev/null || break
  sleep 5
done
tmux kill-session -t dm05-train 2>/dev/null || true

if ! wait_no_trainers 12; then
  log "trainer processes still alive; sending TERM"
  stop_all_trainers TERM
  wait_no_trainers 12 || true
fi
if [[ -n "$(trainer_pids)" ]]; then
  log "trainer processes still alive after TERM; sending KILL"
  stop_all_trainers KILL
  wait_no_trainers 6 || true
fi
if [[ -n "$(trainer_pids)" ]]; then
  log "unable to clear trainers; aborting autotune"
  exit 4
fi

# 192 is already proven stable for >100 production steps. Use 257 as an
# exclusive upper sentinel so a stable 256 remains selectable.
low_batch=192
high_batch=257
probe_batch=256
while true; do
  log "probing batch=$probe_batch for $PROBE_STEPS completed steps"
  DM05_PROBE_STEPS="$PROBE_STEPS" "$PROBE_SCRIPT" "$probe_batch" >> "$LOG_FILE" 2>&1
  probe_status=$?

  if (( probe_status == 0 )); then
    low_batch=$probe_batch
    log "batch=$probe_batch stable"
  elif (( probe_status == 42 )); then
    high_batch=$probe_batch
    log "batch=$probe_batch OOM"
  else
    log "batch=$probe_batch probe error status=$probe_status; selecting last proven batch=$low_batch"
    break
  fi

  # A stable cap or adjacent stable/OOM bounds finish the search.
  if (( low_batch == 256 || high_batch - low_batch <= 1 )); then
    break
  fi
  probe_batch=$(( (low_batch + high_batch) / 2 ))
done

selected=$low_batch
start_production "$selected"

# Confirm that the resumed command adopted the selected batch. The watchdog
# remains active even if this bounded verification times out.
verified=0
for _ in $(seq 1 36); do
  if pgrep -af '[d]m05_sft_demo.py' \
      | grep -q -- "--trainer-config.per-device-train-batch-size $selected"; then
    verified=1
    break
  fi
  sleep 5
done
if (( verified == 0 )); then
  log "resume command did not expose batch=$selected within 180s"
  exit 5
fi

production_paused=0
log "AUTOTUNE_COMPLETE selected_batch=$selected checkpoint=200"
exit 0
