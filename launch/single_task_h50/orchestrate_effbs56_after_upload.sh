#!/usr/bin/env bash
# Local coordinator for one-model-per-DCU H50 fine-tuning.
# Prerequisites (merged datasets and 67.5K base) are already uploaded.
set -euo pipefail

remote=${ROBOSYN_DCU_REMOTE:?set ROBOSYN_DCU_REMOTE=user@host}
port=${ROBOSYN_DCU_PORT:-22}
key=${ROBOSYN_DCU_KEY:-$HOME/.ssh/id_ed25519}
repo=${ROBOSYN_LOCAL_REPO:-$HOME/workspace/RoboSynChallenge}
merge_root=${ROBOSYN_MERGED_DATA_ROOT:-$HOME/FermiBotNas/dataset/RoboSynChallenge/Sim_official_plus_seeded_clean_v21}
data_remote=/tmp/pi05/training_data/RoboSynChallenge
base_remote=/tmp/pi05/base_weights/all10_h64_67500
queue_remote_script=/root/code/RoboSynChallenge/launch/single_task_h50/run_cloud_single_task_h50_parallel_effbs56.sh
accum_remote_script=/root/code/RoboSynChallenge/policy/pi05/scripts/train_accum.py
checkpoint_remote=/tmp/pi05/single_task_h50_effbs56_frozenvlm_v6_checkpoints
status_remote=/tmp/pi05/single_task_h50_effbs56_frozenvlm_v6_status
logs_remote=/tmp/pi05/logs/single_task_h50_from_all10_67500_effbs56_frozenvlm_v6_2ep
exp_prefix=${ROBOSYN_SINGLE_EXP_PREFIX:-from_all10_67500_h50_merged_effbs56_frozenvlm_v6_2ep}
nas_root=${ROBOSYN_SINGLE_NAS_ROOT:-$HOME/FermiBotNas/models/RoboSynChallenge/pi05_single_task_h50_from_all10_67500_merged_v21_effbs56_frozenvlm_v6_2ep}
poll=${ROBOSYN_SINGLE_POLL_SECONDS:-120}

tasks=(
    click_bell drawer_open_place handle_basket item_assembly items_handover
    manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring
)

ssh_opts=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$key" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
rsync_rsh="ssh -p $port -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $key -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

mkdir -p "$nas_root/_orchestrator"
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

verify_prerequisites() {
    jq -e '.tasks == 10 and .episodes == 19510 and .frames == 5612112 and .videos == 58530' \
        "$merge_root/.complete.json" >/dev/null
    until remote_cmd "test -f '$base_remote/params/manifest.ocdbt' && test -f '$base_remote/assets/RoboSynChallenge/all10_expert_h64/norm_stats.json'"; do
        log "waiting for cloud base verification"
        sleep 60
    done
    for task in "${tasks[@]}"; do
        until remote_cmd "test -f '$data_remote/official_plus_seeded_clean_v21_${task}/.complete.json' && test -f '$data_remote/official_plus_seeded_clean_v21_${task}/meta/info.json'"; do
            log "waiting for uploaded dataset: $task"
            sleep 60
        done
    done
    log "prerequisites verified: base 67.5K and ten merged datasets"
}

deploy_and_launch() {
    local local_queue="$repo/launch/single_task_h50/run_cloud_single_task_h50_parallel_effbs56.sh"
    local local_accum="$repo/policy/pi05/scripts/train_accum.py"
    remote_cmd "mkdir -p /root/code/RoboSynChallenge/launch/single_task_h50 /root/code/RoboSynChallenge/policy/pi05/scripts '$status_remote' '$logs_remote'"
    until rsync -a -e "$rsync_rsh" "$local_queue" "$remote:$queue_remote_script"; do
        log "queue script upload interrupted; retrying"
        sleep 30
    done
    until rsync -a -e "$rsync_rsh" "$local_accum" "$remote:$accum_remote_script"; do
        log "accumulation wrapper upload interrupted; retrying"
        sleep 30
    done
    remote_cmd "chmod +x '$queue_remote_script' '$accum_remote_script'; python3 -m py_compile '$accum_remote_script'"

    if remote_cmd "test -f '$status_remote/QUEUE_DONE'"; then
        log "effective-BS56 queue already complete"
        return 0
    fi
    if remote_cmd "test -f '$status_remote/QUEUE_FAILED'"; then
        log "effective-BS56 queue has a failure marker; stopping for review"
        return 1
    fi
    while remote_cmd "pgrep -f '[s]cripts/train(_accum)?[.]py pi05_' >/dev/null"; do
        log "waiting for previous policy trainer to release devices"
        sleep 30
    done
    if remote_cmd "tmux has-session -t pi05-single-task-h50-effbs56-frozenvlm-v6 2>/dev/null"; then
        log "effective-BS56 queue tmux already running"
        return 0
    fi
    remote_cmd "tmux new-session -d -s pi05-single-task-h50-effbs56-frozenvlm-v6 '$queue_remote_script'"
    sleep 5
    remote_cmd "tmux has-session -t pi05-single-task-h50-effbs56-frozenvlm-v6 2>/dev/null"
    log "effective-BS56 queue launched; GPU0 accumulation smoke is running"
}

sync_one_checkpoint() {
    local task=$1 step=$2
    local exp="${exp_prefix}_${task}"
    local source="$checkpoint_remote/pi05_${task}/$exp/$step"
    local task_root="$nas_root/$task/$exp" final="$nas_root/$task/$exp/$step" partial="$nas_root/$task/$exp/.$step.partial"
    local verified="$nas_root/$task/$exp/$step.NAS_VERIFIED"
    local remote_files remote_bytes local_files local_bytes
    test -f "$verified" && return 0
    remote_cmd "test -f '$source/_CHECKPOINT_METADATA' && test -f '$source/params/manifest.ocdbt' && test -f '$source/train_state/manifest.ocdbt' && ! find '$source' -name '.orbax-checkpoint-tmp-*' -print -quit | grep -q ." || return 0
    read -r remote_files remote_bytes < <(stats_remote "$source")
    if test -f "$final/_CHECKPOINT_METADATA"; then
        read -r local_files local_bytes < <(stats_local "$final")
        if test "$remote_files/$remote_bytes" != "$local_files/$local_bytes"; then
            log "existing NAS checkpoint failed verification: $task micro_step=$step"
            return 1
        fi
        printf 'verified_at=%s\nremote=%s\nfiles=%s\nbytes=%s\n' \
            "$(date -Is)" "$source" "$local_files" "$local_bytes" >"$verified"
        return 0
    fi
    mkdir -p "$task_root" "$partial"
    until rsync -a --partial -e "$rsync_rsh" "$remote:$source/" "$partial/"; do
        log "checkpoint sync interrupted: $task micro_step=$step; retrying"
        sleep 60
    done
    read -r local_files local_bytes < <(stats_local "$partial")
    if test "$remote_files/$remote_bytes" != "$local_files/$local_bytes"; then
        log "checkpoint verification mismatch: $task micro_step=$step remote=$remote_files/$remote_bytes local=$local_files/$local_bytes"
        return 1
    fi
    test ! -e "$final"
    mv "$partial" "$final"
    printf 'verified_at=%s\nremote=%s\nfiles=%s\nbytes=%s\n' \
        "$(date -Is)" "$source" "$local_files" "$local_bytes" >"$verified"
    log "checkpoint verified on NAS: $task micro_step=$step files=$local_files bytes=$local_bytes; cloud retained"
}

all_final_checkpoints_on_nas() {
    local task exp final_step
    for task in "${tasks[@]}"; do
        exp="${exp_prefix}_${task}"
        final_step=$(remote_cmd "awk -F= '\$1==\"final_step\" {print \$2}' '$status_remote/${task}.done' 2>/dev/null" || true)
        test -n "$final_step" || return 1
        test -f "$nas_root/$task/$exp/$final_step.NAS_VERIFIED" || return 1
    done
}

sync_outputs() {
    local task exp step remote_steps queue_state
    while true; do
        for task in "${tasks[@]}"; do
            exp="${exp_prefix}_${task}"
            remote_steps=$(remote_cmd "find '$checkpoint_remote/pi05_${task}/$exp' -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null" \
                | awk '/^[0-9]+$/' | sort -n || true)
            while IFS= read -r step; do
                test -n "$step" || continue
                if ! sync_one_checkpoint "$task" "$step"; then
                    log "checkpoint sync/verification failed: $task micro_step=$step"
                    return 1
                fi
            done <<<"$remote_steps"
        done

        mkdir -p "$nas_root/_remote_status" "$nas_root/_remote_logs"
        rsync -a -e "$rsync_rsh" "$remote:$status_remote/" "$nas_root/_remote_status/" 2>/dev/null || true
        rsync -a -e "$rsync_rsh" "$remote:$logs_remote/" "$nas_root/_remote_logs/" 2>/dev/null || true
        queue_state=$(remote_cmd "if test -f '$status_remote/QUEUE_DONE'; then echo done; elif test -f '$status_remote/QUEUE_FAILED'; then echo failed; else echo running; fi" || echo unreachable)
        log "effective-BS56 queue state=$queue_state"
        case "$queue_state" in
            done)
                if all_final_checkpoints_on_nas; then
                    return 0
                fi
                log "queue done; waiting for all final checkpoints to finish NAS verification"
                ;;
            failed) return 1 ;;
            running)
                if ! remote_cmd "tmux has-session -t pi05-single-task-h50-effbs56-frozenvlm-v6 2>/dev/null" \
                    && ! remote_cmd "pgrep -f '[s]cripts/train_accum[.]py pi05_' >/dev/null"; then
                    log "remote queue disappeared without a completion marker"
                    return 1
                fi
                ;;
        esac
        sleep "$poll"
    done
}

log "effective-BS56 coordinator started"
verify_prerequisites
queue_status=0
if deploy_and_launch; then
    sync_outputs || queue_status=$?
else
    queue_status=1
fi

if test "$queue_status" -eq 0; then
    printf 'completed_at=%s\neffective_batch_size=56\nepochs=2\n' "$(date -Is)" >"$nas_root/PIPELINE_DONE"
    log "pipeline completed"
    exit 0
fi
printf 'failed_at=%s\n' "$(date -Is)" >"$nas_root/PIPELINE_FAILED"
log "pipeline stopped for operator review"
exit "$queue_status"
