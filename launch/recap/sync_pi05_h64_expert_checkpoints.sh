#!/usr/bin/env bash
set -uo pipefail

# 训练机地址不入库：运行前 export RECAP_REMOTE=user@host RECAP_PORT=<port> RECAP_KEY=~/.ssh/id_ed25519

remote=${RECAP_REMOTE:?set RECAP_REMOTE=user@host}
port=${RECAP_PORT:-22}
key=${RECAP_KEY:-$HOME/.ssh/id_ed25519}
remote_root=/tmp/pi05/checkpoints/pi05_sample_loading_h64_expert/sample_loading_expert_base_h64_24h_v2
nas_root=$HOME/FermiBotNas/models/RoboSynChallenge/pi05_sample_loading_h64_expert/sample_loading_expert_base_h64_24h_v2
poll=120
ssh_opts=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "$key")
rsync_rsh="ssh -p $port -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $key"

mkdir -p "$nas_root"
exec >>"$nas_root/sync.log" 2>&1
echo "[$(date -Is)] H64 expert checkpoint watcher started"

remote_cmd() { ssh "${ssh_opts[@]}" "$remote" "$@"; }

while true; do
    remote_steps="$({ remote_cmd \
        "find '$remote_root' -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null" || true; } \
        | awk '/^[0-9]+$/' | sort -n)"
    while IFS= read -r step; do
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        final="$nas_root/$step"
        partial="$nas_root/.$step.partial"
        if [[ -d "$final" ]]; then
            remote_cmd "test ! -e '$remote_root/$step' || rm -rf -- '$remote_root/$step'" || true
            continue
        fi
        remote_cmd "test -f '$remote_root/$step/_CHECKPOINT_METADATA'" || continue
        echo "[$(date -Is)] syncing checkpoint $step"
        mkdir -p "$partial"
        if ! rsync -a --partial --delete -e "$rsync_rsh" \
            "$remote:$remote_root/$step/" "$partial/"; then
            echo "[$(date -Is)] rsync failed for $step; retrying later"
            continue
        fi
        read -r remote_files remote_bytes < <(
            remote_cmd "find '$remote_root/$step' -type f -printf '%s\\n' | awk '{n++; s+=\$1} END {printf \"%d %.0f\\n\", n, s}'"
        )
        read -r local_files local_bytes < <(
            find "$partial" -type f -printf '%s\n' | awk '{n++; s+=$1} END {printf "%d %.0f\n", n, s}'
        )
        if [[ "$remote_files/$remote_bytes" != "$local_files/$local_bytes" ]]; then
            echo "[$(date -Is)] verification mismatch step=$step remote=$remote_files/$remote_bytes local=$local_files/$local_bytes"
            continue
        fi
        mv "$partial" "$final"
        echo "[$(date -Is)] checkpoint $step complete files=$local_files bytes=$local_bytes"
        remote_cmd "rm -rf -- '$remote_root/$step'"
        echo "[$(date -Is)] removed remote checkpoint $step after verified NAS sync"
    done <<< "$remote_steps"

    if ! remote_cmd "tmux has-session -t pi05-h64-expert-24h-v2 2>/dev/null"; then
        remaining="$({ remote_cmd \
            "find '$remote_root' -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf x 2>/dev/null" || true; })"
        if [[ -z "$remaining" ]]; then
            echo "[$(date -Is)] training ended and all completed checkpoints are on NAS"
            exit 0
        fi
    fi
    sleep "$poll"
done
