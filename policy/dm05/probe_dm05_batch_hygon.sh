#!/usr/bin/env bash
# Short, isolated 8-DCU batch-size probe. Run only while production is paused.
set -euo pipefail

BATCH_SIZE="${1:-256}"
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive_batch_size]" >&2
  exit 2
fi

RUN_ROOT="${DM05_WORK:-/tmp/dm05}"
POLICY_ROOT="$RUN_ROOT/policy_dm05/dm05"
LOG_DIR="$RUN_ROOT/logs"
PROBE_STEPS="${DM05_PROBE_STEPS:-12}"
POLL_SECONDS="${DM05_PROBE_POLL_SECONDS:-10}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="$RUN_ROOT/user_checkpoints/dm05_batch_probe_${BATCH_SIZE}_$STAMP"
LOG_FILE="$LOG_DIR/dm05_batch_probe_${BATCH_SIZE}_$STAMP.log"
RESULT_FILE="$LOG_FILE.result"

mkdir -p "$LOG_DIR"

# Never contend with the production job for the eight DCUs.
if pgrep -af '[d]m05_sft_demo.py' >/dev/null; then
  echo "[ERROR] production/another DM05 trainer is still running" >&2
  exit 20
fi

cd "$POLICY_ROOT"
export MASTER_PORT="${MASTER_PORT:-29546}"
export DM05_RMSNORM_BATCH_CHUNK=2
export DM05_ATTN_BATCH_CHUNK=2
export DM05_MLP_BATCH_CHUNK=2
export DM05_SAVED_TENSORS_CPU_OFFLOAD=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true

printf '[%s] START bs=%s target_steps=%s output=%s\n' \
  "$(date -Is)" "$BATCH_SIZE" "$PROBE_STEPS" "$OUTPUT_DIR" | tee "$RESULT_FILE"

# Start in a dedicated process group so a stop signal reaches torchrun and all
# eight workers. save-strategy=no prevents an 82 GiB probe checkpoint.
setsid bash finetune_dcu.sh sample_loading_relative 8 "$OUTPUT_DIR" \
  --trainer-config.per-device-train-batch-size "$BATCH_SIZE" \
  --trainer-config.gradient-accumulation-steps 1 \
  --trainer-config.num-train-steps 1000 \
  --trainer-config.save-strategy no \
  >"$LOG_FILE" 2>&1 &
probe_pid=$!

stop_group() {
  local signal="$1"
  kill "-$signal" -- "-$probe_pid" 2>/dev/null || true
}

terminate_probe() {
  stop_group INT
  for _ in $(seq 1 18); do
    kill -0 "$probe_pid" 2>/dev/null || break
    sleep 5
  done
  if kill -0 "$probe_pid" 2>/dev/null; then
    stop_group TERM
    for _ in $(seq 1 6); do
      kill -0 "$probe_pid" 2>/dev/null || break
      sleep 5
    done
  fi
  if kill -0 "$probe_pid" 2>/dev/null; then
    stop_group KILL
  fi
  wait "$probe_pid" 2>/dev/null || true
}

while kill -0 "$probe_pid" 2>/dev/null; do
  if grep -Eqi 'out of memory|OutOfMemoryError|MemoryError' "$LOG_FILE"; then
    printf '[%s] RESULT=OOM\n' "$(date -Is)" | tee -a "$RESULT_FILE"
    terminate_probe
    exit 42
  fi
  if grep -Eq 'Traceback \(most recent call last\)|ChildFailedError' "$LOG_FILE"; then
    printf '[%s] RESULT=ERROR\n' "$(date -Is)" | tee -a "$RESULT_FILE"
    terminate_probe
    exit 43
  fi

  last_step="$(tr '\r' '\n' < "$LOG_FILE" \
    | sed -n "s/.*'step': \([0-9][0-9]*\).*/\1/p" | tail -n 1)"
  if [[ -n "$last_step" ]] && (( last_step >= PROBE_STEPS )); then
    printf '[%s] RESULT=STABLE last_step=%s\n' \
      "$(date -Is)" "$last_step" | tee -a "$RESULT_FILE"
    terminate_probe
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

wait "$probe_pid" || probe_status=$?
probe_status="${probe_status:-0}"
if grep -Eqi 'out of memory|OutOfMemoryError|MemoryError' "$LOG_FILE"; then
  printf '[%s] RESULT=OOM post_exit_status=%s\n' \
    "$(date -Is)" "$probe_status" | tee -a "$RESULT_FILE"
  exit 42
fi
printf '[%s] RESULT=EARLY_EXIT status=%s\n' \
  "$(date -Is)" "$probe_status" | tee -a "$RESULT_FILE"
exit 44
