#!/usr/bin/env bash
# Run ten independent H50 fine-tunes on an eight-device DCU host.
#
# Each worker sees exactly one physical device.  The first eight tasks start
# immediately (staggered to avoid a checkpoint restore thundering herd); when
# one finishes, that device picks up one of the two remaining tasks.
set -uo pipefail

repo=${ROBOSYN_REPO_ROOT:-/root/code/RoboSynChallenge}
pi05="$repo/policy/pi05"
data_home=${ROBOSYN_HF_LEROBOT_HOME:-/tmp/pi05/training_data}
base=${ROBOSYN_SINGLE_BASE:-/tmp/pi05/base_weights/all10_h64_67500}
checkpoint_root=${ROBOSYN_SINGLE_CHECKPOINT_ROOT:-/tmp/pi05/single_task_h50_checkpoints}
status_root=${ROBOSYN_SINGLE_STATUS_ROOT:-/tmp/pi05/single_task_h50_status}
log_root=${ROBOSYN_SINGLE_LOG_ROOT:-/tmp/pi05/logs/single_task_h50_from_all10_67500}
exp=${ROBOSYN_SINGLE_EXP:-from_all10_67500_h50_merged_bs4_8k}
steps=${ROBOSYN_SINGLE_STEPS:-8000}
batch_size=${ROBOSYN_SINGLE_BATCH_SIZE:-4}
workers=${ROBOSYN_SINGLE_NUM_WORKERS:-4}
save_interval=${ROBOSYN_SINGLE_SAVE_INTERVAL:-2000}
stagger_seconds=${ROBOSYN_SINGLE_STAGGER_SECONDS:-90}

tasks=(
    click_bell
    drawer_open_place
    handle_basket
    item_assembly
    items_handover
    manipulate_pipette
    mixer_operating
    sample_loading
    table_rearrangement
    water_pouring
)

mkdir -p "$checkpoint_root" "$status_root" "$log_root" /tmp/pi05/wandb_single_h50
exec >>"$log_root/queue.log" 2>&1

log() {
    echo "[$(date -Is)] $*"
}

final_step=$((steps - 1))
base_asset="$base/assets/RoboSynChallenge/all10_expert_h64/norm_stats.json"
test -f "$base/params/manifest.ocdbt" || { log "missing base params: $base"; exit 2; }
test -f "$base_asset" || { log "missing all10 norm stats: $base_asset"; exit 2; }

for task in "${tasks[@]}"; do
    repo_id="RoboSynChallenge/official_plus_seeded_clean_v21_${task}"
    dataset="$data_home/$repo_id"
    test -f "$dataset/.complete.json" || { log "missing dataset completion marker: $dataset"; exit 2; }
    test -f "$dataset/meta/info.json" || { log "missing dataset info: $dataset"; exit 2; }
done

if test -f "$status_root/QUEUE_DONE"; then
    log "queue already complete"
    exit 0
fi

printf 'started_at=%s\nexp=%s\nsteps=%s\nbatch_size=%s\nworkers=%s\nbase=%s\n' \
    "$(date -Is)" "$exp" "$steps" "$batch_size" "$workers" "$base" \
    >"$status_root/QUEUE_CONFIG"

run_task() {
    local task=$1 gpu=$2 config repo_id checkpoint_dir task_log rc mode=()
    config="pi05_${task}"
    repo_id="RoboSynChallenge/official_plus_seeded_clean_v21_${task}"
    checkpoint_dir="$checkpoint_root/$config/$exp"
    task_log="$log_root/${task}.log"

    if test -f "$status_root/${task}.done"; then
        log "task already complete: $task"
        return 0
    fi

    if test -d "$checkpoint_dir"; then
        latest=$(find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
            | awk '/^[0-9]+$/' | sort -n | tail -n 1)
        if test -n "$latest" \
            && test -f "$checkpoint_dir/$latest/_CHECKPOINT_METADATA" \
            && test -f "$checkpoint_dir/$latest/train_state/manifest.ocdbt"; then
            mode=(--resume)
            log "resuming $task on device $gpu from step $latest"
        else
            log "refusing to overwrite incomplete existing output: $checkpoint_dir"
            printf 'task=%s\ngpu=%s\nfailed_at=%s\nreason=existing_output_without_resumable_checkpoint\n' \
                "$task" "$gpu" "$(date -Is)" >"$status_root/${task}.failed"
            return 91
        fi
    else
        log "starting $task on device $gpu"
    fi

    printf 'task=%s\ngpu=%s\nstarted_at=%s\nlog=%s\n' \
        "$task" "$gpu" "$(date -Is)" "$task_log" >"$status_root/${task}.running"

    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export HF_LEROBOT_HOME="$data_home"
        export LD_PRELOAD=/opt/mpi/lib/libmpi.so
        export PYTHONPATH="$pi05/src:$pi05/packages/openpi-client/src"
        export OMP_NUM_THREADS=16
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88
        export WANDB_MODE=online
        export WANDB_DIR=/tmp/pi05/wandb_single_h50
        export WANDB_RUN_GROUP=single_task_h50_from_all10_67500
        cd "$pi05" || exit 92
        python3 scripts/train.py "$config" \
            --exp-name="$exp" \
            "${mode[@]}" \
            --model.action-horizon=50 \
            --weight-loader.params-path="$base/params" \
            --data.repo-id="$repo_id" \
            --data.assets.assets-dir="$base/assets" \
            --data.assets.asset-id=RoboSynChallenge/all10_expert_h64 \
            --checkpoint-base-dir="$checkpoint_root" \
            --num-train-steps="$steps" \
            --batch-size="$batch_size" \
            --num-workers="$workers" \
            --fsdp-devices=1 \
            --ema-decay=None \
            --save-interval="$save_interval" \
            --keep-period="$save_interval" \
            --lr-schedule.warmup-steps=500 \
            --lr-schedule.peak-lr=1e-5 \
            --lr-schedule.decay-steps="$steps" \
            --lr-schedule.decay-lr=1e-6
    ) >>"$task_log" 2>&1
    rc=$?

    if test "$rc" -eq 0 \
        && test -f "$checkpoint_dir/$final_step/_CHECKPOINT_METADATA" \
        && test -f "$checkpoint_dir/$final_step/params/manifest.ocdbt" \
        && test -f "$checkpoint_dir/$final_step/train_state/manifest.ocdbt"; then
        printf 'task=%s\ngpu=%s\ncompleted_at=%s\nfinal_step=%s\ncheckpoint=%s\n' \
            "$task" "$gpu" "$(date -Is)" "$final_step" "$checkpoint_dir/$final_step" \
            >"$status_root/${task}.done"
        rm -f "$status_root/${task}.running" "$status_root/${task}.failed"
        log "completed $task on device $gpu"
        return 0
    fi

    printf 'task=%s\ngpu=%s\nfailed_at=%s\nexit_code=%s\nlog=%s\n' \
        "$task" "$gpu" "$(date -Is)" "$rc" "$task_log" >"$status_root/${task}.failed"
    rm -f "$status_root/${task}.running"
    log "failed $task on device $gpu exit=$rc"
    return "$rc"
}

declare -A task_by_pid=()
declare -A gpu_by_pid=()
next_task=0
failures=0

launch_task() {
    local task=$1 gpu=$2 pid
    (run_task "$task" "$gpu") &
    pid=$!
    task_by_pid[$pid]=$task
    gpu_by_pid[$pid]=$gpu
    next_task=$((next_task + 1))
    sleep "$stagger_seconds"
}

for gpu in 0 1 2 3 4 5 6 7; do
    test "$next_task" -lt "${#tasks[@]}" || break
    launch_task "${tasks[$next_task]}" "$gpu"
done

while test "${#task_by_pid[@]}" -gt 0; do
    for pid in "${!task_by_pid[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        task=${task_by_pid[$pid]}
        gpu=${gpu_by_pid[$pid]}
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
        unset 'task_by_pid[$pid]' 'gpu_by_pid[$pid]'
        if test "$next_task" -lt "${#tasks[@]}"; then
            launch_task "${tasks[$next_task]}" "$gpu"
        fi
    done
    sleep 15
done

if test "$failures" -eq 0; then
    printf 'completed_at=%s\ntasks=%s\n' "$(date -Is)" "${#tasks[@]}" >"$status_root/QUEUE_DONE"
    rm -f "$status_root/QUEUE_FAILED"
    log "all ten single-task runs completed"
    exit 0
fi

printf 'failed_at=%s\nfailures=%s\n' "$(date -Is)" "$failures" >"$status_root/QUEUE_FAILED"
log "queue finished with $failures failed task(s)"
exit 1
