#!/usr/bin/env bash
# Resume the interrupted ACT sample_loading run once the Pro6000 has enough
# free memory. The dataset is revalidated immediately before training so a
# stale in-memory LeRobot index cannot survive another dataset repair.
set -euo pipefail

project_root="${ACT_PROJECT_ROOT:-/workspace/shared/RoboSynChallenge-act-sample-loading}"
act_root="${ACT_ROOT:-$project_root/policy/act}"
dataset_root="${ACT_DATASET_ROOT:-/mnt/FermiBotNas/dataset/RoboSynChallenge/Sim_official_plus_seeded_clean_v21/sample_loading}"
output_dir="${ACT_OUTPUT_DIR:-$project_root/outputs/act_sample_loading_merged_h50_bs64_2ep}"
validator="${ACT_VALIDATOR:-$project_root/scripts/validate_lerobot_dataset.py}"
python_bin="${ACT_PYTHON:-$act_root/.venv/bin/python}"
gpu_memory_limit_mib="${ACT_MAX_EXISTING_GPU_MEMORY_MIB:-8192}"
poll_seconds="${ACT_WAIT_POLL_SECONDS:-30}"
capacity_checks_required="${ACT_CAPACITY_CHECKS_REQUIRED:-4}"
wait_log="$project_root/logs/act_sample_loading_resume_wait.log"
train_log="$project_root/logs/act_sample_loading_merged_h50_bs64_2ep_resume_from_008000.log"
status_file="$output_dir/RESUME_STATUS"
validation_report="$dataset_root/POSTFIX_VALIDATION_REPORT_20260831.json"

mkdir -p "$project_root/logs" "$output_dir"
exec >>"$wait_log" 2>&1

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

gpu_used_memory_mib() {
    nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk '{ total += $1 } END { print total + 0 }'
}

wait_for_gpu_capacity() {
    local checks=0 used_memory_mib
    while (( checks < capacity_checks_required )); do
        if ! used_memory_mib="$(gpu_used_memory_mib)"; then
            checks=0
            log "unable to query GPU memory; waiting"
        elif (( used_memory_mib <= gpu_memory_limit_mib )); then
            (( checks += 1 ))
            log "GPU capacity check $checks/$capacity_checks_required: existing=${used_memory_mib} MiB, limit=${gpu_memory_limit_mib} MiB"
        else
            checks=0
            log "GPU busy: existing=${used_memory_mib} MiB exceeds ${gpu_memory_limit_mib} MiB; waiting"
        fi
        (( checks >= capacity_checks_required )) || sleep "$poll_seconds"
    done
}

require_resume_checkpoint() {
    local last_checkpoint
    last_checkpoint="$(readlink -f "$output_dir/checkpoints/last")"
    [[ "$(basename "$last_checkpoint")" == "008000" ]]
    [[ "$(<"$last_checkpoint/training_state/training_step.json")" == *'"step": 8000'* ]]
    [[ -s "$last_checkpoint/pretrained_model/model.safetensors" ]]
    [[ -s "$last_checkpoint/training_state/optimizer_state.safetensors" ]]
    log "resume checkpoint verified: $last_checkpoint"
}

validate_dataset() {
    log "validating repaired NAS dataset"
    "$python_bin" "$validator" "$dataset_root" \
        --expected-episodes 1754 \
        --action-horizon 50 \
        --skip-video-decode \
        --report "$validation_report"
    log "NAS dataset validation passed"
}

log "ACT sample_loading resume waiter started"
require_resume_checkpoint
wait_for_gpu_capacity
validate_dataset

log "starting ACT resume from checkpoint 008000"
printf 'state=running\nstarted_at=%s\ncheckpoint=008000\n' "$(date -Is)" >"$status_file"

set +e
(
    cd "$act_root"
    env CUDA_VISIBLE_DEVICES=0 \
        ACCELERATE_MIXED_PRECISION=bf16 \
        HF_HOME=/root/.cache/huggingface \
        "$python_bin" scripts/train.py \
            --dataset-root "$dataset_root" \
            --output-dir "$output_dir" \
            --job-name act_sample_loading_merged_h50_bs64_2ep \
            --steps 21162 \
            --batch-size 64 \
            --chunk-size 50 \
            --n-action-steps 50 \
            --num-workers 4 \
            --log-freq 100 \
            --save-freq 2000 \
            --eval-freq 0 \
            --use-amp \
            --wandb \
            --wandb-project robosynchallenge \
            --wandb-name act_sample_loading_merged_h50_bs64_2ep \
            --resume
) >>"$train_log" 2>&1
exit_code=$?
set -e

if (( exit_code == 0 )); then
    printf 'state=complete\nfinished_at=%s\nexit_code=0\n' "$(date -Is)" >"$status_file"
    log "ACT resume completed successfully"
else
    printf 'state=failed\nfinished_at=%s\nexit_code=%s\n' "$(date -Is)" "$exit_code" >"$status_file"
    log "ACT resume failed with exit code $exit_code; see $train_log"
fi
exit "$exit_code"
