#!/usr/bin/env bash
# Restart the production tmux if it exits unexpectedly before the deadline.
set -uo pipefail

STOP_AT="${DM05_STOP_AT:-2026-08-26T11:00:00-04:00}"
STOP_EPOCH="$(date -d "$STOP_AT" +%s)" || exit 1
TRAIN_SESSION="${DM05_TRAIN_SESSION:-dm05-train}"
TRAIN_SCRIPT="${DM05_TRAIN_SCRIPT:-/tmp/dm05/start_dm05_hygon.sh}"
WATCH_LOG="${DM05_WATCH_LOG:-/tmp/dm05/logs/dm05_watchdog.log}"
TRAIN_LOG="${DM05_TRAIN_LOG:-/tmp/dm05/logs/dm05_sample_loading_relative_hygon_bs192.log}"
STALE_SECONDS="${DM05_STALE_SECONDS:-3600}"

while (( $(date +%s) < STOP_EPOCH )); do
  if ! tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    printf '[%s] %s 不在线，自动重启\n' "$(date -Is)" "$TRAIN_SESSION" >> "$WATCH_LOG"
    tmux new-session -d -s "$TRAIN_SESSION" "bash $TRAIN_SCRIPT"
  elif [[ -f "$TRAIN_LOG" ]]; then
    log_age=$(( $(date +%s) - $(stat -c %Y "$TRAIN_LOG") ))
    if (( log_age > STALE_SECONDS )); then
      printf '[%s] 训练日志已 %ss 未更新，重启并从最新 checkpoint 续训\n' \
        "$(date -Is)" "$log_age" >> "$WATCH_LOG"
      tmux send-keys -t "$TRAIN_SESSION" C-c 2>/dev/null || true
      sleep 30
      tmux kill-session -t "$TRAIN_SESSION" 2>/dev/null || true
      tmux new-session -d -s "$TRAIN_SESSION" "bash $TRAIN_SCRIPT"
    fi
  fi
  sleep 60
done

printf '[%s] 已到训练截止时间，等待 Trainer 完成最终保存\n' "$(date -Is)" >> "$WATCH_LOG"
for _ in $(seq 1 30); do
  if ! tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    printf '[%s] Trainer 已保存并退出，watchdog 正常退出\n' "$(date -Is)" >> "$WATCH_LOG"
    exit 0
  fi
  sleep 60
done

printf '[%s] 截止时间后 30 分钟训练仍在，发送 SIGINT 兜底停止\n' \
  "$(date -Is)" >> "$WATCH_LOG"
tmux send-keys -t "$TRAIN_SESSION" C-c 2>/dev/null || true
