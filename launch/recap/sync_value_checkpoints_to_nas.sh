#!/usr/bin/env bash
set -euo pipefail

train_pid=${1:?usage: sync_value_checkpoints_to_nas.sh TRAIN_PID CHECKPOINT_ROOT NAS_ROOT}
checkpoint_root=${2:?usage: sync_value_checkpoints_to_nas.sh TRAIN_PID CHECKPOINT_ROOT NAS_ROOT}
nas_root=${3:?usage: sync_value_checkpoints_to_nas.sh TRAIN_PID CHECKPOINT_ROOT NAS_ROOT}
mkdir -p "$nas_root"

sync_checkpoints() {
  [[ -d "$checkpoint_root" ]] || return 0
  local checkpoint step partial final differences
  while IFS= read -r -d '' checkpoint; do
    step=$(basename "$checkpoint")
    [[ "$step" =~ ^[0-9]{6}$ ]] || continue
    [[ -f "$checkpoint/pretrained_model/model.safetensors" ]] || continue
    [[ -f "$checkpoint/training_state/optimizer_state.safetensors" ]] || continue
    final="$nas_root/$step"
    partial="$nas_root/.$step.partial"
    [[ -f "$final/SYNC_COMPLETE" ]] && continue

    rm -rf -- "$partial"
    mkdir -p "$partial"
    rsync -a --delete "$checkpoint/" "$partial/"
    # CIFS/NAS mounts may not preserve Unix mode bits or sub-second mtimes.
    # Verify names, sizes, and file contents instead of metadata, otherwise a
    # correct copy is falsely rejected forever.
    differences=$(rsync -rcni --delete "$checkpoint/" "$partial/")
    [[ -z "$differences" ]] || { rm -rf -- "$partial"; continue; }
    printf 'source=%s\nsynced_at=%s\n' "$checkpoint" "$(date --iso-8601=seconds)" \
      >"$partial/SYNC_COMPLETE"
    [[ ! -e "$final" ]] || mv "$final" "$nas_root/.incomplete_${step}_$(date +%s)"
    mv "$partial" "$final"
    echo "[$(date --iso-8601=seconds)] synced $step to $final"

    # Keep the newest local checkpoint for crash recovery. Older complete
    # checkpoints are already durable on NAS and can be removed safely.
    local other other_step
    for other in "$checkpoint_root"/[0-9]*; do
      [[ -d "$other" ]] || continue
      other_step=$(basename "$other")
      [[ "$other_step" =~ ^[0-9]{6}$ ]] || continue
      [[ "$other_step" != "$step" ]] || continue
      [[ -f "$nas_root/$other_step/SYNC_COMPLETE" ]] || continue
      rm -rf -- "$other"
      echo "[$(date --iso-8601=seconds)] removed older local checkpoint $other_step"
    done
  done < <(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -print0 | sort -zV)
}

while kill -0 "$train_pid" 2>/dev/null; do
  sync_checkpoints
  sleep 20
done
for _ in $(seq 1 15); do
  sync_checkpoints
  sleep 10
done
