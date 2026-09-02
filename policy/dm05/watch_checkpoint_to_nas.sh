#!/usr/bin/env bash
# Keep the local NAS sync tmux alive through the final checkpoint transfer.
set -uo pipefail

STOP_AT="${DM05_STOP_AT:-2026-08-26T11:00:00-04:00}"
STOP_EPOCH="$(date -d "$STOP_AT" +%s)" || exit 1
WATCH_UNTIL_EPOCH=$((STOP_EPOCH + 4 * 60 * 60))
SYNC_SCRIPT="${DM05_SYNC_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/checkpoint_to_nas.sh}"
SYNC_LOG="/tmp/dm05_checkpoint_to_nas.log"
WATCH_LOG="/tmp/dm05_checkpoint_sync_watchdog.log"

while (( $(date +%s) < WATCH_UNTIL_EPOCH )); do
  if ! tmux has-session -t dm05-checkpoint-sync 2>/dev/null; then
    printf '[%s] dm05-checkpoint-sync 不在线，自动重启\n' "$(date -Is)" \
      >> "$WATCH_LOG"
    tmux new-session -d -s dm05-checkpoint-sync \
      "exec $SYNC_SCRIPT >> $SYNC_LOG 2>&1"
  fi
  sleep 60
done

printf '[%s] 已超过最终同步保护窗口，watchdog 正常退出\n' "$(date -Is)" \
  >> "$WATCH_LOG"
