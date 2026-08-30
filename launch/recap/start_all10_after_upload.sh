#!/usr/bin/env bash
set -euo pipefail

# 训练机地址不入库：运行前 export RECAP_REMOTE=user@host RECAP_PORT=<port> RECAP_KEY=~/.ssh/id_ed25519

repo=${ROBOSYN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
upload_log=/tmp/pi05_all10_upload.log
coordinator_log=/tmp/pi05_all10_coordinator.log
remote=${RECAP_REMOTE:?set RECAP_REMOTE=user@host}
ssh_opts=(-p "${RECAP_PORT:-22}" -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "${RECAP_KEY:-$HOME/.ssh/id_ed25519}")

exec >>"$coordinator_log" 2>&1
echo "[$(date -Is)] waiting for all-10 expert upload"
while tmux has-session -t pi05-all10-upload 2>/dev/null; do
    sleep 30
done
grep -q "all 10 expert datasets are ready" "$upload_log"
echo "[$(date -Is)] upload verified; waiting for RobotWin2 60K base params"

while ! ssh "${ssh_opts[@]}" "$remote" \
    "test -f /tmp/pi05/base_weights/Hoshipu_pi05-robotwin2-random-60k/.download_complete"; do
    sleep 30
done
echo "[$(date -Is)] data and RobotWin2 base verified; starting remote norm-stats/training session"

ssh "${ssh_opts[@]}" "$remote" '
set -e
if pgrep -f "^python3 scripts/train.py" >/dev/null; then
    echo "another training process is active" >&2
    exit 1
fi
tmux kill-session -t pi05-all10-h64 2>/dev/null || true
tmux new-session -d -s pi05-all10-h64 /root/code/RoboSynChallenge/launch/recap/pi05_all10_h64_expert_hygon_train.sh
sleep 5
tmux has-session -t pi05-all10-h64
tail -20 /tmp/pi05/logs/all10_expert_base_h64_bs64_steps100k.log
'

tmux kill-session -t pi05-all10-checkpoint-sync 2>/dev/null || true
tmux new-session -d -s pi05-all10-checkpoint-sync "$repo/launch/recap/sync_pi05_all10_h64_checkpoints.sh"
echo "[$(date -Is)] training session and NAS watcher started"
