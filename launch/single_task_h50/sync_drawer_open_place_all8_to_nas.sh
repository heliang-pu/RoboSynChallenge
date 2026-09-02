#!/usr/bin/env bash
# Copy-only watcher: verify each cloud checkpoint on NAS and retain cloud data.
set -uo pipefail

remote=root@lb-m9m5yqlh-7fkybp0x59lcnom4.clb.bj-tencentclb.com
port=42000
key=/home/phl/.ssh/id_ed25519
source_root=/tmp/pi05/drawer_open_place_h50_all8_checkpoints/pi05_drawer_open_place/drawer_open_place_from_all10_67500_h50_bs64_2ep
nas_root=/home/phl/FermiBotNas/models/RoboSynChallenge/pi05_drawer_open_place_h50_from_all10_67500/drawer_open_place_h50_bs64_2ep
poll=120
ssh_opts=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$key" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
rsync_rsh="ssh -p $port -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $key -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

mkdir -p "$nas_root"
exec >>"$nas_root/sync.log" 2>&1
echo "[$(date -Is)] drawer_open_place all8 checkpoint watcher started"
remote_cmd() { ssh "${ssh_opts[@]}" "$remote" "$@"; }

while true; do
    steps=$(remote_cmd "find '$source_root' -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null" \
        | awk '/^[0-9]+$/' | sort -n || true)
    while IFS= read -r step; do
        test -n "$step" || continue
        final="$nas_root/$step"
        partial="$nas_root/.$step.partial"
        verified="$nas_root/$step.NAS_VERIFIED"
        test -f "$verified" && continue
        remote_cmd "test -f '$source_root/$step/_CHECKPOINT_METADATA' && test -f '$source_root/$step/params/manifest.ocdbt' && test -f '$source_root/$step/train_state/manifest.ocdbt' && ! find '$source_root/$step' -name '.orbax-checkpoint-tmp-*' -print -quit | grep -q ." || continue
        read -r remote_files remote_bytes < <(remote_cmd "find '$source_root/$step' -type f -printf '%s\\n' | awk '{n++; s+=\$1} END {printf \"%d %.0f\\n\", n, s}'")
        if test -f "$final/_CHECKPOINT_METADATA"; then
            read -r local_files local_bytes < <(find "$final" -type f -printf '%s\n' | awk '{n++;s+=$1} END{printf "%d %.0f\n",n,s}')
        else
            mkdir -p "$partial"
            until rsync -a --partial -e "$rsync_rsh" "$remote:$source_root/$step/" "$partial/"; do sleep 30; done
            read -r local_files local_bytes < <(find "$partial" -type f -printf '%s\n' | awk '{n++;s+=$1} END{printf "%d %.0f\n",n,s}')
            if test "$remote_files/$remote_bytes" != "$local_files/$local_bytes"; then
                echo "[$(date -Is)] mismatch step=$step remote=$remote_files/$remote_bytes local=$local_files/$local_bytes"
                continue
            fi
            test ! -e "$final" || continue
            mv "$partial" "$final"
        fi
        if test "$remote_files/$remote_bytes" = "$local_files/$local_bytes"; then
            printf 'verified_at=%s\nfiles=%s\nbytes=%s\ncloud_retained=true\n' "$(date -Is)" "$local_files" "$local_bytes" >"$verified"
            echo "[$(date -Is)] verified step=$step files=$local_files bytes=$local_bytes"
        fi
    done <<<"$steps"

    if remote_cmd "test -f '$source_root/26718/_CHECKPOINT_METADATA'" \
        && test -f "$nas_root/26718.NAS_VERIFIED"; then
        echo "[$(date -Is)] final checkpoint complete"
        exit 0
    fi
    sleep "$poll"
done
