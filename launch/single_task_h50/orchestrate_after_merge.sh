#!/usr/bin/env bash
# Persistent local coordinator:
#   1. wait for the validated official+seeded datasets on NAS;
#   2. preserve the all10 final checkpoint on NAS without deleting cloud data;
#   3. upload the ten merged task datasets and the proven 67.5K base;
#   4. launch the eight-device task queue;
#   5. copy every complete single-task checkpoint to NAS after verification.
set -euo pipefail

# 训练机地址不入库：ROBOSYN_DCU_REMOTE=user@host ROBOSYN_DCU_PORT=<port>
remote=${ROBOSYN_DCU_REMOTE:?set ROBOSYN_DCU_REMOTE=user@host}
port=${ROBOSYN_DCU_PORT:-22}
key=${ROBOSYN_DCU_KEY:-$HOME/.ssh/id_ed25519}
merge_host=${ROBOSYN_MERGE_HOST:-fmc3-1-4090-outer}
repo=${ROBOSYN_LOCAL_REPO:-$HOME/workspace/RoboSynChallenge}
merge_root=${ROBOSYN_MERGED_DATA_ROOT:-$HOME/FermiBotNas/dataset/RoboSynChallenge/Sim_official_plus_seeded_clean_v21}
base_nas=${ROBOSYN_SINGLE_BASE_NAS:-$HOME/FermiBotNas/models/RoboSynChallenge/pi05_all10_h64_expert/all10_expert_base_h64_bs64_steps100k/67500}
all10_nas=${ROBOSYN_ALL10_NAS:-$HOME/FermiBotNas/models/RoboSynChallenge/pi05_all10_h64_expert/all10_expert_base_h64_bs64_steps100k}
all10_remote=/tmp/pi05/checkpoints/pi05_all10_h64_expert/all10_expert_base_h64_bs64_steps100k
data_remote=/tmp/pi05/training_data/RoboSynChallenge
base_remote=/tmp/pi05/base_weights/all10_h64_67500
queue_remote_script=/root/code/RoboSynChallenge/launch/single_task_h50/run_cloud_single_task_h50_queue.sh
checkpoint_remote=/tmp/pi05/single_task_h50_checkpoints
status_remote=/tmp/pi05/single_task_h50_status
logs_remote=/tmp/pi05/logs/single_task_h50_from_all10_67500
nas_root=${ROBOSYN_SINGLE_NAS_ROOT:-$HOME/FermiBotNas/models/RoboSynChallenge/pi05_single_task_h50_from_all10_67500_merged_v21}
exp=${ROBOSYN_SINGLE_EXP:-from_all10_67500_h50_merged_bs4_8k}
steps=${ROBOSYN_SINGLE_STEPS:-8000}
poll=${ROBOSYN_SINGLE_POLL_SECONDS:-120}

tasks=(
    click_bell drawer_open_place handle_basket item_assembly items_handover
    manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring
)

ssh_opts=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$key" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
rsync_rsh="ssh -p $port -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $key -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

mkdir -p "$nas_root/_orchestrator" "$all10_nas"
exec >>"$nas_root/_orchestrator/coordinator.log" 2>&1

log() {
    echo "[$(date -Is)] $*"
}

remote_cmd() {
    ssh "${ssh_opts[@]}" "$remote" "$@"
}

stats_local() {
    find "$1" -type f -printf '%s\n' | awk '{n++; s+=$1} END {printf "%d %.0f\n", n, s}'
}

stats_remote() {
    remote_cmd "find '$1' -type f -printf '%s\\n' | awk '{n++; s+=\$1} END {printf \"%d %.0f\\n\", n, s}'"
}

wait_for_merge() {
    local merge_cmd
    merge_cmd="/home/fmc3/workspace/RoboSynChallenge/.venv/bin/python /home/fmc3/workspace/RoboSynChallenge/scripts/merge_clean_official_v21.py --seeded-clean-root /home/fmc3/FermiBotNas/dataset/RoboSynChallenge/Syn/seeded_1000_20260827_lerobot_v21_merged_clean --official-root /home/fmc3/FermiBotNas/dataset/RoboSynChallenge/Sim_clean_filtered_pruned --output-root /home/fmc3/FermiBotNas/dataset/RoboSynChallenge/Sim_official_plus_seeded_clean_v21 --validation-python /home/fmc3/workspace/RoboSynChallenge/policy/pi05/.venv/bin/python --validator /home/fmc3/workspace/RoboSynChallenge/scripts/validate_lerobot_dataset.py --build-workers 3 --resume"

    while ! test -f "$merge_root/.complete.json" || ! test -f "$merge_root/MERGE_MANIFEST.json"; do
        if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$merge_host" \
            "pgrep -f 'scripts/merge_clean_official(_chunk)?[.]py' >/dev/null"; then
            log "merge validation is not running; starting a persistent resume unit"
            ssh -o BatchMode=yes -o ConnectTimeout=10 "$merge_host" \
                "systemctl --user reset-failed robosyn-official-seeded-merge.service 2>/dev/null || true; systemd-run --user --collect --unit=robosyn-official-seeded-merge --description='RoboSyn official plus seeded merge' --property=Restart=on-failure --property=RestartSec=60s --working-directory=/home/fmc3/workspace/RoboSynChallenge $merge_cmd" || true
        fi
        published=$(find "$merge_root" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -printf x 2>/dev/null | wc -c)
        log "waiting for validated merge: published=$published/10"
        sleep "$poll"
    done

    jq -e '.tasks == 10 and .episodes == 19510 and .frames == 5612112 and .videos == 58530' \
        "$merge_root/.complete.json" >/dev/null
    for task in "${tasks[@]}"; do
        test -f "$merge_root/$task/.complete.json"
        jq -e '.passed == true and .gates.parquet_and_counts == "pass" and .gates.videos == "pass" and .gates.lerobot_training_read == "pass"' \
            "$merge_root/$task/VALIDATION_REPORT.json" >/dev/null
    done
    log "validated merge ready: 19510 episodes, 5612112 frames"
}

sync_all10_final() {
    local step=99999 source="$all10_remote/99999" final="$all10_nas/99999" partial="$all10_nas/.99999.partial"
    if test -f "$final/_CHECKPOINT_METADATA"; then
        log "all10 final checkpoint already present on NAS"
        return 0
    fi

    while ! remote_cmd "test -f '$source/_CHECKPOINT_METADATA' && test -f '$source/params/manifest.ocdbt' && test -f '$source/train_state/manifest.ocdbt' && ! find '$source' -name '.orbax-checkpoint-tmp-*' -print -quit | grep -q ."; do
        log "waiting for finalized all10 step 99999"
        sleep "$poll"
    done

    mkdir -p "$partial"
    until rsync -a --partial -e "$rsync_rsh" "$remote:$source/" "$partial/"; do
        log "all10 99999 rsync interrupted; retrying"
        sleep 60
    done
    read -r remote_files remote_bytes < <(stats_remote "$source")
    read -r local_files local_bytes < <(stats_local "$partial")
    test "$remote_files/$remote_bytes" = "$local_files/$local_bytes"
    test ! -e "$final"
    mv "$partial" "$final"
    log "all10 final checkpoint 99999 verified on NAS files=$local_files bytes=$local_bytes; cloud copy retained"
}

upload_base() {
    local staging="${base_remote}.uploading" local_files local_bytes remote_files remote_bytes
    test -f "$base_nas/params/manifest.ocdbt"
    test -f "$base_nas/assets/RoboSynChallenge/all10_expert_h64/norm_stats.json"
    read -r local_files local_bytes < <(
        find "$base_nas/params" "$base_nas/assets" -type f -printf '%s\n' \
            | awk '{n++; s+=$1} END {printf "%d %.0f\n", n, s}'
    )
    if remote_cmd "test -f '$base_remote/params/manifest.ocdbt' && test -f '$base_remote/assets/RoboSynChallenge/all10_expert_h64/norm_stats.json'"; then
        read -r remote_files remote_bytes < <(stats_remote "$base_remote")
        test "$remote_files/$remote_bytes" = "$local_files/$local_bytes"
        log "67.5K base already verified on cloud"
        return 0
    fi
    remote_cmd "mkdir -p '$staging'"
    until rsync -a --partial -e "$rsync_rsh" "$base_nas/params" "$base_nas/assets" "$remote:$staging/"; do
        log "base upload interrupted; retrying"
        sleep 60
    done
    read -r remote_files remote_bytes < <(stats_remote "$staging")
    test "$remote_files/$remote_bytes" = "$local_files/$local_bytes"
    remote_cmd "test ! -e '$base_remote' && mv '$staging' '$base_remote'"
    log "67.5K base uploaded and verified files=$local_files bytes=$local_bytes"
}

upload_dataset() {
    local task=$1
    local source="$merge_root/$task"
    local name="official_plus_seeded_clean_v21_${task}"
    local final="$data_remote/$name" staging="$data_remote/.${name}.uploading"
    local local_files local_bytes remote_files remote_bytes
    read -r local_files local_bytes < <(stats_local "$source")
    if remote_cmd "test -f '$final/.complete.json' && test -f '$final/meta/info.json'"; then
        read -r remote_files remote_bytes < <(stats_remote "$final")
        test "$remote_files/$remote_bytes" = "$local_files/$local_bytes"
        log "dataset already verified on cloud: $task"
        return 0
    fi
    remote_cmd "mkdir -p '$staging'"
    until rsync -a --partial -e "$rsync_rsh" "$source/" "$remote:$staging/"; do
        log "dataset upload interrupted for $task; retrying"
        sleep 60
    done
    read -r remote_files remote_bytes < <(stats_remote "$staging")
    test "$remote_files/$remote_bytes" = "$local_files/$local_bytes"
    remote_cmd "test ! -e '$final' && mv '$staging' '$final'"
    log "dataset uploaded and verified: $task files=$local_files bytes=$local_bytes"
}

deploy_and_launch_queue() {
    local script="$repo/launch/single_task_h50/run_cloud_single_task_h50_queue.sh"
    remote_cmd "mkdir -p /root/code/RoboSynChallenge/launch/single_task_h50"
    rsync -a -e "$rsync_rsh" "$script" "$remote:$queue_remote_script"
    remote_cmd "chmod +x '$queue_remote_script'; mkdir -p '$status_remote' '$logs_remote'"

    # Bracket the first character so pgrep does not match the remote shell
    # command that is performing this check.
    while remote_cmd "pgrep -f '[s]cripts/train.py pi05_all10_h64_expert' >/dev/null"; do
        log "waiting for all10 training to release all eight devices"
        sleep "$poll"
    done

    if remote_cmd "test -f '$status_remote/QUEUE_DONE'"; then
        log "single-task queue already complete"
        return 0
    fi
    if remote_cmd "test -f '$status_remote/QUEUE_FAILED'"; then
        log "single-task queue has a failure marker; refusing an automatic destructive/repetitive retry"
        return 1
    fi
    if remote_cmd "tmux has-session -t pi05-single-task-h50 2>/dev/null"; then
        log "single-task queue tmux already running"
        return 0
    fi
    remote_cmd "tmux new-session -d -s pi05-single-task-h50 '$queue_remote_script'"
    sleep 5
    remote_cmd "tmux has-session -t pi05-single-task-h50 2>/dev/null"
    log "single-task H50 queue launched on eight devices"
}

sync_one_checkpoint() {
    local task=$1 step=$2
    local source="$checkpoint_remote/pi05_${task}/$exp/$step"
    local task_root="$nas_root/$task/$exp" final="$nas_root/$task/$exp/$step" partial="$nas_root/$task/$exp/.$step.partial"
    local remote_files remote_bytes local_files local_bytes
    test -f "$final/_CHECKPOINT_METADATA" && return 0
    remote_cmd "test -f '$source/_CHECKPOINT_METADATA' && test -f '$source/params/manifest.ocdbt' && test -f '$source/train_state/manifest.ocdbt'" || return 0
    mkdir -p "$task_root" "$partial"
    until rsync -a --partial -e "$rsync_rsh" "$remote:$source/" "$partial/"; do
        log "checkpoint rsync interrupted: $task step=$step; retrying"
        sleep 60
    done
    read -r remote_files remote_bytes < <(stats_remote "$source")
    read -r local_files local_bytes < <(stats_local "$partial")
    test "$remote_files/$remote_bytes" = "$local_files/$local_bytes"
    test ! -e "$final"
    mv "$partial" "$final"
    log "checkpoint verified on NAS: $task step=$step files=$local_files bytes=$local_bytes; cloud copy retained"
}

sync_single_task_outputs() {
    local task step remote_steps queue_state all_final final_step=$((steps - 1))
    while true; do
        for task in "${tasks[@]}"; do
            remote_steps=$(remote_cmd "find '$checkpoint_remote/pi05_${task}/$exp' -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null" \
                | awk '/^[0-9]+$/' | sort -n || true)
            while IFS= read -r step; do
                test -n "$step" || continue
                sync_one_checkpoint "$task" "$step"
            done <<<"$remote_steps"
        done

        mkdir -p "$nas_root/_remote_status" "$nas_root/_remote_logs"
        rsync -a -e "$rsync_rsh" "$remote:$status_remote/" "$nas_root/_remote_status/" 2>/dev/null || true
        rsync -a -e "$rsync_rsh" "$remote:$logs_remote/" "$nas_root/_remote_logs/" 2>/dev/null || true

        queue_state=$(remote_cmd "if test -f '$status_remote/QUEUE_DONE'; then echo done; elif test -f '$status_remote/QUEUE_FAILED'; then echo failed; else echo running; fi" || echo unreachable)
        log "single-task queue state=$queue_state"
        case "$queue_state" in
            done)
                all_final=1
                for task in "${tasks[@]}"; do
                    if ! test -f "$nas_root/$task/$exp/$final_step/_CHECKPOINT_METADATA"; then
                        all_final=0
                        break
                    fi
                done
                if test "$all_final" -eq 1; then
                    return 0
                fi
                log "queue is done; waiting for all ten final checkpoints to finish NAS verification"
                ;;
            failed) return 1 ;;
        esac
        sleep "$poll"
    done
}

log "coordinator started"
sync_all10_final &
all10_sync_pid=$!

wait_for_merge
wait "$all10_sync_pid"
upload_base
for task in "${tasks[@]}"; do
    upload_dataset "$task"
done
queue_status=0
if deploy_and_launch_queue; then
    sync_single_task_outputs || queue_status=$?
else
    queue_status=1
fi

if test "$queue_status" -eq 0; then
    printf 'completed_at=%s\nexp=%s\n' "$(date -Is)" "$exp" >"$nas_root/PIPELINE_DONE"
    log "pipeline completed"
    exit 0
fi

printf 'failed_at=%s\nexp=%s\n' "$(date -Is)" "$exp" >"$nas_root/PIPELINE_FAILED"
log "pipeline stopped because one or more task runs failed"
exit "$queue_status"
