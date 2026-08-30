#!/usr/bin/env bash
# Train ten independent H50 task models on eight DCUs.
#
# One process owns one physical DCU.  A startup probe selects the largest
# micro batch that can complete two real optimizer updates with gradient
# accumulation.  Every full run then uses effective batch 56 and exactly two
# epochs in sample count.  Eight tasks start first; the final two are queued.
set -uo pipefail

repo=${ROBOSYN_REPO_ROOT:-/root/code/RoboSynChallenge}
pi05="$repo/policy/pi05"
data_home=${ROBOSYN_HF_LEROBOT_HOME:-/tmp/pi05/training_data}
base=${ROBOSYN_SINGLE_BASE:-/tmp/pi05/base_weights/all10_h64_67500}
checkpoint_root=${ROBOSYN_SINGLE_CHECKPOINT_ROOT:-/tmp/pi05/single_task_h50_effbs56_checkpoints}
status_root=${ROBOSYN_SINGLE_STATUS_ROOT:-/tmp/pi05/single_task_h50_effbs56_status}
log_root=${ROBOSYN_SINGLE_LOG_ROOT:-/tmp/pi05/logs/single_task_h50_from_all10_67500_effbs56_2ep}
smoke_root=${ROBOSYN_SINGLE_SMOKE_ROOT:-/tmp/pi05/single_task_h50_effbs56_smoke}
exp_prefix=${ROBOSYN_SINGLE_EXP_PREFIX:-from_all10_67500_h50_merged_effbs56_2ep}
effective_batch=${ROBOSYN_SINGLE_EFFECTIVE_BATCH_SIZE:-56}
epochs=${ROBOSYN_SINGLE_EPOCHS:-2}
workers=${ROBOSYN_SINGLE_NUM_WORKERS:-4}
stagger_seconds=${ROBOSYN_SINGLE_STAGGER_SECONDS:-90}
micro_candidates=${ROBOSYN_SINGLE_MICRO_BATCH_CANDIDATES:-"4 2 1"}

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

mkdir -p "$checkpoint_root" "$status_root" "$log_root" "$smoke_root" /tmp/pi05/wandb_single_h50_effbs56
exec >>"$log_root/queue.log" 2>&1

log() {
    echo "[$(date -Is)] $*"
}

cleanup_orphan_workers() {
    local pid parent
    for pid in $(pgrep -f "[m]ultiprocessing.spawn import spawn_main" 2>/dev/null); do
        parent=$(ps -o ppid= -p "$pid" | tr -d ' ')
        test "$parent" = 1 && kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in $(pgrep -f "[m]ultiprocessing.resource_tracker import main" 2>/dev/null); do
        parent=$(ps -o ppid= -p "$pid" | tr -d ' ')
        test "$parent" = 1 && kill -TERM "$pid" 2>/dev/null || true
    done
}

test "$effective_batch" -eq 56 || { log "effective batch must be 56"; exit 2; }
test "$epochs" -eq 2 || { log "epochs must be 2"; exit 2; }
test -f "$base/params/manifest.ocdbt" || { log "missing base params: $base"; exit 2; }
test -f "$base/assets/RoboSynChallenge/all10_expert_h64/norm_stats.json" \
    || { log "missing all10 norm stats"; exit 2; }
for task in "${tasks[@]}"; do
    dataset="$data_home/RoboSynChallenge/official_plus_seeded_clean_v21_${task}"
    test -f "$dataset/.complete.json" || { log "missing dataset marker: $dataset"; exit 2; }
    test -f "$dataset/meta/info.json" || { log "missing dataset info: $dataset"; exit 2; }
done

if test -f "$status_root/QUEUE_DONE"; then
    log "queue already complete"
    exit 0
fi
if test -f "$status_root/QUEUE_FAILED"; then
    log "failure marker exists; operator review is required"
    exit 1
fi

probe_accumulation() {
    local marker="$status_root/ACCUMULATION_SMOKE_OK" micro accumulation smoke_exp smoke_log smoke_steps rc
    if test -f "$marker"; then
        # shellcheck disable=SC1090
        source "$marker"
        test "$EFFECTIVE_BATCH_SIZE" -eq 56
        test "$MICRO_BATCH_SIZE" -gt 0
        test "$ACCUMULATION_STEPS" -gt 0
        test $((MICRO_BATCH_SIZE * ACCUMULATION_STEPS)) -eq 56
        micro_batch=$MICRO_BATCH_SIZE
        accumulation_steps=$ACCUMULATION_STEPS
        log "reusing accumulation smoke result micro=$micro_batch accumulation=$accumulation_steps"
        return 0
    fi

    cleanup_orphan_workers
    for micro in $micro_candidates; do
        test $((effective_batch % micro)) -eq 0 || continue
        accumulation=$((effective_batch / micro))
        smoke_steps=$((accumulation * 2))
        smoke_exp="smoke_effbs56_mb${micro}x${accumulation}_$(date +%s)"
        smoke_log="$log_root/${smoke_exp}.log"
        log "smoke testing GPU0 micro=$micro accumulation=$accumulation effective=$effective_batch"
        (
            export CUDA_VISIBLE_DEVICES=0
            unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
            export HF_LEROBOT_HOME="$data_home"
            export LD_PRELOAD=/opt/mpi/lib/libmpi.so
            export PYTHONPATH="$pi05/src:$pi05/packages/openpi-client/src"
            export OMP_NUM_THREADS=16
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
            export WANDB_MODE=disabled
            export OPENPI_GRADIENT_ACCUMULATION_STEPS="$accumulation"
            export OPENPI_EFFECTIVE_BATCH_SIZE="$effective_batch"
            export OPENPI_FIRST_SAVE_UPDATE=1
            export OPENPI_SAVE_EVERY_UPDATES=1
            export OPENPI_SMOKE_NO_CHECKPOINT=1
            cd "$pi05" || exit 92
            python3 scripts/train_accum.py pi05_click_bell \
                --exp-name="$smoke_exp" \
                --model.action-horizon=50 \
                --weight-loader.params-path="$base/params" \
                --data.repo-id=RoboSynChallenge/official_plus_seeded_clean_v21_click_bell \
                --data.assets.assets-dir="$base/assets" \
                --data.assets.asset-id=RoboSynChallenge/all10_expert_h64 \
                --checkpoint-base-dir="$smoke_root" \
                --num-train-steps="$smoke_steps" \
                --batch-size="$micro" \
                --num-workers=2 \
                --fsdp-devices=1 \
                --ema-decay=None \
                --save-interval=1 \
                --keep-period=1 \
                --lr-schedule.warmup-steps=1 \
                --lr-schedule.peak-lr=1e-5 \
                --lr-schedule.decay-steps=2 \
                --lr-schedule.decay-lr=1e-6
        ) >>"$smoke_log" 2>&1
        rc=$?
        if test "$rc" -eq 0 \
            && grep -q "optimizer_update=2/2" "$smoke_log" \
            && ! grep -qE "OutOfMemory|RESOURCE_EXHAUSTED|Traceback" "$smoke_log"; then
            printf 'MICRO_BATCH_SIZE=%s\nACCUMULATION_STEPS=%s\nEFFECTIVE_BATCH_SIZE=%s\nSMOKE_LOG=%q\nCOMPLETED_AT=%q\n' \
                "$micro" "$accumulation" "$effective_batch" "$smoke_log" "$(date -Is)" >"$marker"
            micro_batch=$micro
            accumulation_steps=$accumulation
            cleanup_orphan_workers
            log "accumulation smoke passed micro=$micro_batch accumulation=$accumulation_steps"
            return 0
        fi
        log "accumulation smoke failed micro=$micro accumulation=$accumulation exit=$rc"
        cleanup_orphan_workers
        sleep 15
    done
    printf 'failed_at=%s\nreason=no_micro_batch_fit\n' "$(date -Is)" >"$status_root/QUEUE_FAILED"
    return 1
}

probe_accumulation || { log "no viable effective-BS56 accumulation plan"; exit 1; }

printf 'started_at=%s\nexp_prefix=%s\nepochs=%s\neffective_batch_size=%s\nmicro_batch_size=%s\naccumulation_steps=%s\nbase=%s\n' \
    "$(date -Is)" "$exp_prefix" "$epochs" "$effective_batch" "$micro_batch" "$accumulation_steps" "$base" \
    >"$status_root/QUEUE_CONFIG"

run_task() {
    local task=$1 gpu=$2 config repo_id dataset info frames updates micro_steps final_step
    local save_every_updates exp checkpoint_dir task_log latest rc mode=()
    config="pi05_${task}"
    repo_id="RoboSynChallenge/official_plus_seeded_clean_v21_${task}"
    dataset="$data_home/$repo_id"
    info="$dataset/meta/info.json"
    frames=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["total_frames"]))' "$info")
    updates=$(((epochs * frames + effective_batch - 1) / effective_batch))
    micro_steps=$((updates * accumulation_steps))
    final_step=$((micro_steps - 1))
    save_every_updates=$(((updates + 3) / 4))
    exp="${exp_prefix}_${task}"
    checkpoint_dir="$checkpoint_root/$config/$exp"
    task_log="$log_root/${task}.log"

    if test -f "$status_root/${task}.done" \
        && test -f "$checkpoint_dir/$final_step/_CHECKPOINT_METADATA"; then
        log "already complete: $task"
        return 0
    fi
    if test -d "$checkpoint_dir"; then
        latest=$(find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
            | awk '/^[0-9]+$/' | sort -n | tail -n 1)
        if test -n "$latest" \
            && test -f "$checkpoint_dir/$latest/_CHECKPOINT_METADATA" \
            && test -f "$checkpoint_dir/$latest/train_state/manifest.ocdbt"; then
            mode=(--resume)
            log "resuming $task on GPU$gpu from micro_step=$latest"
        else
            printf 'task=%s\ngpu=%s\nfailed_at=%s\nreason=existing_output_without_resumable_checkpoint\n' \
                "$task" "$gpu" "$(date -Is)" >"$status_root/${task}.failed"
            return 91
        fi
    fi

    printf 'task=%s\ngpu=%s\nstarted_at=%s\nframes=%s\nepochs=%s\neffective_batch_size=%s\nmicro_batch_size=%s\naccumulation_steps=%s\noptimizer_updates=%s\nmicro_steps=%s\nfinal_step=%s\nlog=%s\n' \
        "$task" "$gpu" "$(date -Is)" "$frames" "$epochs" "$effective_batch" "$micro_batch" \
        "$accumulation_steps" "$updates" "$micro_steps" "$final_step" "$task_log" \
        >"$status_root/${task}.running"
    log "starting $task GPU$gpu updates=$updates micro_steps=$micro_steps effective_bs=$effective_batch"

    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
        export HF_LEROBOT_HOME="$data_home"
        export LD_PRELOAD=/opt/mpi/lib/libmpi.so
        export PYTHONPATH="$pi05/src:$pi05/packages/openpi-client/src"
        export OMP_NUM_THREADS=16
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
        export WANDB_MODE=online
        export WANDB_DIR=/tmp/pi05/wandb_single_h50_effbs56
        export WANDB_RUN_GROUP=single_task_h50_from_all10_67500_effbs56_2ep
        export OPENPI_GRADIENT_ACCUMULATION_STEPS="$accumulation_steps"
        export OPENPI_EFFECTIVE_BATCH_SIZE="$effective_batch"
        export OPENPI_FIRST_SAVE_UPDATE=100
        export OPENPI_SAVE_EVERY_UPDATES="$save_every_updates"
        unset OPENPI_SMOKE_NO_CHECKPOINT
        cd "$pi05" || exit 92
        python3 scripts/train_accum.py "$config" \
            --exp-name="$exp" \
            "${mode[@]}" \
            --model.action-horizon=50 \
            --weight-loader.params-path="$base/params" \
            --data.repo-id="$repo_id" \
            --data.assets.assets-dir="$base/assets" \
            --data.assets.asset-id=RoboSynChallenge/all10_expert_h64 \
            --checkpoint-base-dir="$checkpoint_root" \
            --num-train-steps="$micro_steps" \
            --batch-size="$micro_batch" \
            --num-workers="$workers" \
            --fsdp-devices=1 \
            --ema-decay=None \
            --save-interval=1 \
            --keep-period=1 \
            --lr-schedule.warmup-steps=500 \
            --lr-schedule.peak-lr=1e-5 \
            --lr-schedule.decay-steps="$updates" \
            --lr-schedule.decay-lr=1e-6
    ) >>"$task_log" 2>&1
    rc=$?

    if test "$rc" -eq 0 \
        && test -f "$checkpoint_dir/$final_step/_CHECKPOINT_METADATA" \
        && test -f "$checkpoint_dir/$final_step/params/manifest.ocdbt" \
        && test -f "$checkpoint_dir/$final_step/train_state/manifest.ocdbt"; then
        printf 'task=%s\ngpu=%s\ncompleted_at=%s\nframes=%s\nepochs=%s\neffective_batch_size=%s\nmicro_batch_size=%s\naccumulation_steps=%s\noptimizer_updates=%s\nmicro_steps=%s\nfinal_step=%s\ncheckpoint=%s\n' \
            "$task" "$gpu" "$(date -Is)" "$frames" "$epochs" "$effective_batch" "$micro_batch" \
            "$accumulation_steps" "$updates" "$micro_steps" "$final_step" "$checkpoint_dir/$final_step" \
            >"$status_root/${task}.done"
        rm -f "$status_root/${task}.running" "$status_root/${task}.failed"
        log "completed $task on GPU$gpu"
        return 0
    fi
    printf 'task=%s\ngpu=%s\nfailed_at=%s\nexit_code=%s\nlog=%s\n' \
        "$task" "$gpu" "$(date -Is)" "$rc" "$task_log" >"$status_root/${task}.failed"
    rm -f "$status_root/${task}.running"
    log "failed $task on GPU$gpu exit=$rc"
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
    printf 'completed_at=%s\ntasks=%s\nepochs=%s\neffective_batch_size=%s\nmicro_batch_size=%s\naccumulation_steps=%s\n' \
        "$(date -Is)" "${#tasks[@]}" "$epochs" "$effective_batch" "$micro_batch" "$accumulation_steps" \
        >"$status_root/QUEUE_DONE"
    log "all ten independent task models completed"
    exit 0
fi

printf 'failed_at=%s\nfailures=%s\n' "$(date -Is)" "$failures" >"$status_root/QUEUE_FAILED"
log "queue finished with $failures failed task(s)"
exit 1
