#!/usr/bin/env bash
# Poll the Hygon training host, copy each completed checkpoint to NAS, and keep
# only the newest completed checkpoint both remotely and on NAS.
set -uo pipefail

# 训练机地址不入库：DM05_REMOTE_HOST=<host> DM05_REMOTE_PORT=<port>
REMOTE_HOST="${DM05_REMOTE_HOST:?set DM05_REMOTE_HOST=<training host>}"
REMOTE_PORT="${DM05_REMOTE_PORT:-22}"
REMOTE_USER="${DM05_REMOTE_USER:-root}"
SSH_KEY="${DM05_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_OUTPUT="${DM05_REMOTE_OUTPUT:-/tmp/dm05/user_checkpoints/dm05_sample_loading_relative_hygon_bs192}"
NAS_OUTPUT="${DM05_NAS_OUTPUT:-$HOME/FermiBotNas/mymodels/RobosynChallenge/sample_loading/dm05_relative_hygon_bs192}"
POLL_SECONDS="${DM05_CHECKPOINT_POLL_SECONDS:-120}"

SSH_OPTS=(-p "$REMOTE_PORT" -o ConnectTimeout=15 -o BatchMode=yes
  -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "$SSH_KEY")
RSYNC_SSH="ssh -p $REMOTE_PORT -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $SSH_KEY"

mkdir -p "$NAS_OUTPUT"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

while true; do
  latest="$({ ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "find '$REMOTE_OUTPUT' -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\\n' 2>/dev/null" || true; } \
    | awk -F- '/^checkpoint-[0-9]+$/ {print $2, $0}' | sort -n | tail -1 | awk '{print $2}')"

  if [[ ! "$latest" =~ ^checkpoint-[0-9]+$ ]]; then
    log "尚无完整 checkpoint，${POLL_SECONDS}s 后重查"
    sleep "$POLL_SECONDS"
    continue
  fi

  # Transformers writes trainer_state.json near the end of _save_checkpoint.
  # Never copy/delete while the distributed checkpoint is still being written.
  if ! ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
      "test -f '$REMOTE_OUTPUT/$latest/trainer_state.json'"; then
    log "$latest 仍在写入，${POLL_SECONDS}s 后重查"
    sleep "$POLL_SECONDS"
    continue
  fi

  final_dir="$NAS_OUTPUT/$latest"
  partial_dir="$NAS_OUTPUT/.${latest}.partial"
  latest_step="${latest#checkpoint-}"
  if [[ ! -f "$final_dir/.nas_sync_complete" ]]; then
    mkdir -p "$partial_dir"
    log "开始同步 $latest 到 NAS 临时目录"
    if rsync -a --partial --delete --info=progress2 -e "$RSYNC_SSH" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_OUTPUT/$latest/" "$partial_dir/"; then
      remote_inventory="$(ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
        "find '$REMOTE_OUTPUT/$latest' -type f -printf '%s\\n' | awk '{count++; bytes+=\$1} END {printf \"%d %.0f\", count, bytes}'")"
      local_inventory="$(find "$partial_dir" -type f -printf '%s\n' \
        | awk '{count++; bytes+=$1} END {printf "%d %.0f", count, bytes}')"
      if [[ "$remote_inventory" != "$local_inventory" ]]; then
        log "$latest 清单校验失败: 远端=[$remote_inventory] NAS=[$local_inventory]"
        sleep "$POLL_SECONDS"
        continue
      fi
      log "$latest 清单校验通过: files/bytes=[$local_inventory]"
      printf 'source=%s@%s:%s/%s\nsynced_at=%s\n' \
        "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_OUTPUT" "$latest" "$(date -Is)" \
        > "$partial_dir/.nas_sync_complete"
      if [[ -d "$final_dir" ]]; then
        log "目标目录已存在但不完整，保留临时副本等待人工检查"
        sleep "$POLL_SECONDS"
        continue
      fi
      mv "$partial_dir" "$final_dir"
      log "$latest 已完整落盘 NAS"
    else
      log "$latest 同步失败，将从临时目录断点续传"
      sleep "$POLL_SECONDS"
      continue
    fi
  fi

  # A newer checkpoint is safely on NAS. Only now remove older copies.
  for path in "$NAS_OUTPUT"/checkpoint-*; do
    [[ -d "$path" ]] || continue
    name="${path##*/}"
    [[ "$name" =~ ^checkpoint-[0-9]+$ ]] || continue
    step="${name#checkpoint-}"
    if (( 10#$step < 10#$latest_step )); then
      log "删除 NAS 旧 checkpoint: $name"
      rm -rf -- "$path"
    fi
  done

  # A failed older transfer may leave a large hidden partial directory. Once a
  # newer checkpoint is safely complete, those older partials are obsolete.
  for path in "$NAS_OUTPUT"/.checkpoint-*.partial; do
    [[ -d "$path" ]] || continue
    name="${path##*/}"
    [[ "$name" =~ ^\.checkpoint-([0-9]+)\.partial$ ]] || continue
    step="${BASH_REMATCH[1]}"
    if (( 10#$step < 10#$latest_step )); then
      log "删除 NAS 旧断点目录: $name"
      rm -rf -- "$path"
    fi
  done

  remote_names="$(ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
    "find '$REMOTE_OUTPUT' -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\\n' 2>/dev/null" || true)"
  while IFS= read -r name; do
    [[ "$name" =~ ^checkpoint-[0-9]+$ ]] || continue
    step="${name#checkpoint-}"
    if (( 10#$step < 10#$latest_step )); then
      log "删除训练机旧 checkpoint: $name"
      ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" \
        "rm -rf -- '$REMOTE_OUTPUT/$name'" || true
    fi
  done <<< "$remote_names"

  sleep "$POLL_SECONDS"
done
