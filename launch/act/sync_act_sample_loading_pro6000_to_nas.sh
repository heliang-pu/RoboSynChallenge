#!/usr/bin/env bash
# Copy-only ACT checkpoint watcher running on the Pro6000 workstation.
set -uo pipefail

source_root=/workspace/shared/RoboSynChallenge-act-sample-loading/outputs/act_sample_loading_merged_h50_bs64_2ep/checkpoints
nas_root=/mnt/FermiBotNas/models/RoboSynChallenge/act_sample_loading_merged_h50_bs64_2ep
poll=120

mkdir -p "$nas_root"
exec >>"$nas_root/sync.log" 2>&1
echo "[$(date -Is)] ACT sample_loading checkpoint watcher started"

stats() {
    find "$1" -type f -printf '%s\n' | awk '{n++;s+=$1} END{printf "%d %.0f\n",n,s}'
}

while true; do
    checkpoints=$(find "$source_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
        | awk '/^[0-9]+$/' | sort -n || true)
    while IFS= read -r step_dir; do
        test -n "$step_dir" || continue
        source="$source_root/$step_dir"
        final="$nas_root/$step_dir"
        partial="$nas_root/.$step_dir.partial"
        verified="$nas_root/$step_dir.NAS_VERIFIED"
        test -f "$verified" && continue
        test -f "$source/pretrained_model/model.safetensors" || continue
        test -f "$source/pretrained_model/config.json" || continue
        test -f "$source/training_state/training_step.json" || continue
        read -r files1 bytes1 < <(stats "$source")
        sleep 5
        read -r files2 bytes2 < <(stats "$source")
        test "$files1/$bytes1" = "$files2/$bytes2" || continue
        if ! test -f "$final/pretrained_model/model.safetensors"; then
            mkdir -p "$partial"
            rsync -a --partial "$source/" "$partial/" || continue
            read -r local_files local_bytes < <(stats "$partial")
            if test "$files2/$bytes2" != "$local_files/$local_bytes"; then
                echo "[$(date -Is)] mismatch checkpoint=$step_dir source=$files2/$bytes2 nas=$local_files/$local_bytes"
                continue
            fi
            test ! -e "$final" || continue
            mv "$partial" "$final"
        fi
        read -r local_files local_bytes < <(stats "$final")
        if test "$files2/$bytes2" = "$local_files/$local_bytes"; then
            train_step=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$final/training_state/training_step.json")
            printf 'verified_at=%s\ntrain_step=%s\nfiles=%s\nbytes=%s\nsource_retained=true\n' \
                "$(date -Is)" "$train_step" "$local_files" "$local_bytes" >"$verified"
            echo "[$(date -Is)] verified checkpoint=$step_dir train_step=$train_step files=$local_files bytes=$local_bytes"
            if test "$train_step" -eq 21162; then
                printf 'completed_at=%s\ncheckpoint=%s\n' "$(date -Is)" "$nas_root/$step_dir/pretrained_model" >"$nas_root/FINAL_COMPLETE"
                exit 0
            fi
        fi
    done <<<"$checkpoints"
    sleep "$poll"
done
