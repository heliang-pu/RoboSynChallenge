#!/usr/bin/env bash
set -uo pipefail

session=pi05-h64-expert-24h-v2
pattern='^python3 scripts/train.py pi05_sample_loading_h64_expert --exp-name=sample_loading_expert_base_h64_24h_v2 --overwrite$'
log=/tmp/pi05/logs/sample_loading_expert_base_h64_24h_v2.log

created=$(tmux display-message -p -t "$session":0 '#{session_created}') || exit 1
deadline=$((created + 24 * 60 * 60))
printf '[%s] 24h watchdog armed; session_created=%s deadline=%s\n' \
    "$(date -Is)" "$(date -Is -d "@$created")" "$(date -Is -d "@$deadline")" >> "$log"

while (( $(date +%s) < deadline )); do
    sleep 60
done

pids=$(pgrep -f "$pattern" || true)
if [[ -n "$pids" ]]; then
    printf '[%s] 24h deadline reached; sending SIGINT to pid(s): %s\n' \
        "$(date -Is)" "$pids" >> "$log"
    kill -INT $pids
fi

for _ in $(seq 1 60); do
    pgrep -f "$pattern" >/dev/null || exit 0
    sleep 10
done
pids=$(pgrep -f "$pattern" || true)
[[ -z "$pids" ]] || kill -TERM $pids
