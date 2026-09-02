#!/usr/bin/env bash
# Launch the production DM0.5 run on the Hygon 8-DCU host.
set -uo pipefail

RUN_ROOT="/tmp/dm05"
OUTPUT_DIR="$RUN_ROOT/user_checkpoints/dm05_sample_loading_relative_hygon_bs192"
LOG_FILE="$RUN_ROOT/logs/dm05_sample_loading_relative_hygon_bs192.log"
BATCH_SIZE_FILE="$RUN_ROOT/production_batch_size"
BATCH_SIZE="${DM05_BATCH_SIZE:-192}"
if [[ -z "${DM05_BATCH_SIZE:-}" && -f "$BATCH_SIZE_FILE" ]]; then
  BATCH_SIZE="$(tr -d '[:space:]' < "$BATCH_SIZE_FILE")"
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  printf '[%s] invalid production batch size: %q\n' \
    "$(date -Is)" "$BATCH_SIZE" >> "$LOG_FILE"
  exit 2
fi

cd "$RUN_ROOT/policy_dm05/dm05" || exit 1
export MASTER_PORT=29540
export DM05_RMSNORM_BATCH_CHUNK=2
export DM05_ATTN_BATCH_CHUNK=2
export DM05_MLP_BATCH_CHUNK=2
export WANDB_MODE=online
export WANDB_RUN_ID="${WANDB_RUN_ID:-77jw5pyp}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

printf '\n[%s] START_DM05 bs=%s stop_at=2026-08-26T11:00:00-04:00\n' \
  "$(date -Is)" "$BATCH_SIZE" >> "$LOG_FILE"
LOG_START_SIZE="$(stat -c %s "$LOG_FILE" 2>/dev/null || printf '0')"
bash finetune_dcu.sh sample_loading_relative 8 "$OUTPUT_DIR" \
  --trainer-config.per-device-train-batch-size "$BATCH_SIZE" \
  --trainer-config.gradient-accumulation-steps 1 \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.save-steps 100 \
  --trainer-config.save-total-limit 4 \
  --trainer-config.wandb-project robosynchallenge-dm05 \
  --trainer-config.stop-at 2026-08-26T11:00:00-04:00 \
  >> "$LOG_FILE" 2>&1
train_status=$?
printf '\nTRAIN_EXIT_CODE=%s\n' "$train_status" >> "$LOG_FILE"

# A batch selected by the short probe may still hit a rarer high-memory sample
# later. Persistently fall back to the long-proven batch 192 so the watchdog
# cannot enter an OOM restart loop.
if (( train_status != 0 && BATCH_SIZE > 192 )); then
  if tail -c "+$((LOG_START_SIZE + 1))" "$LOG_FILE" \
      | grep -Eqi 'out of memory|OutOfMemoryError|MemoryError'; then
    batch_tmp="$BATCH_SIZE_FILE.tmp.$$"
    printf '192\n' > "$batch_tmp"
    mv -f -- "$batch_tmp" "$BATCH_SIZE_FILE"
    printf '[%s] batch=%s OOM; persistent fallback batch=192\n' \
      "$(date -Is)" "$BATCH_SIZE" >> "$LOG_FILE"
  fi
fi
exit "$train_status"
