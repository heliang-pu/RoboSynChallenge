#!/usr/bin/env bash
# Resumable PRO-6000 evaluation queue:
# every 2.5k checkpoint x H={10,30,50,64} x 10 tasks x 20 paired episodes.
set -uo pipefail

repo="${ALL10_EVAL_REPO:-/workspace/shared/RoboSynChallenge-eval-all10}"
pi05="$repo/policy/pi05"
config=pi05_all10_h64_expert
model=all10_expert_base_h64_bs64_steps100k
checkpoint_root="${ALL10_EVAL_CHECKPOINT_ROOT:-/mnt/FermiBotNas/models/RoboSynChallenge/pi05_all10_h64_expert/$model}"
results_root="${ALL10_EVAL_RESULTS_ROOT:-/mnt/FermiBotNas/models/RoboSynChallenge/pi05_all10_h64_expert/${model}_eval_all_ckpts_h10_30_50_64_20eps_4view}"
cache_root="${ALL10_EVAL_CACHE_ROOT:-/tmp/pi05_all10_h64_eval_checkpoints}"
expected_final_step="${ALL10_EVAL_FINAL_STEP:-99999}"
episodes="${ALL10_EVAL_EPISODES:-20}"
worker_count="${ALL10_EVAL_WORKERS:-1}"
poll_seconds="${ALL10_EVAL_POLL_SECONDS:-120}"
retry_cooldown="${ALL10_EVAL_RETRY_COOLDOWN:-900}"
claim_init_grace_seconds="${ALL10_EVAL_CLAIM_INIT_GRACE_SECONDS:-60}"
claim_heartbeat_seconds="${ALL10_EVAL_CLAIM_HEARTBEAT_SECONDS:-15}"
claim_stale_seconds="${ALL10_EVAL_CLAIM_STALE_SECONDS:-300}"
legacy_claim_stale_seconds="${ALL10_EVAL_LEGACY_CLAIM_STALE_SECONDS:-7200}"
report_lock_stale_seconds="${ALL10_EVAL_REPORT_LOCK_STALE_SECONDS:-900}"
xla_mem_fraction="${ALL10_EVAL_XLA_MEM_FRACTION:-0.35}"
protocol_revision="${ALL10_EVAL_PROTOCOL_REVISION:-all10_h64_v2_bounded_texture_pool}"
python_bin="${ALL10_EVAL_PYTHON:-$pi05/.venv/bin/python}"
runtime_overlay="${ALL10_EVAL_RUNTIME_OVERLAY:-/workspace/shared/.venvs/pi05-eval-torch210-overlay}"
embodichain_root="${ALL10_EVAL_EMBODICHAIN_ROOT:-/workspace/shared/EmbodiChain-eval-all10}"
worker_host="${ALL10_EVAL_HOSTNAME:-$(hostname -f 2>/dev/null || hostname)}"

tasks=(
    click_bell drawer_open_place handle_basket item_assembly items_handover
    manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring
)
horizons=(10 30 50 64)
expected_steps=()
for ((expected_step=2500; expected_step<expected_final_step; expected_step+=2500)); do
    expected_steps+=("$expected_step")
done
expected_steps+=("$expected_final_step")

mkdir -p "$results_root" "$results_root/logs"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$results_root/queue.log"
}

checkpoint_is_complete() {
    local path="$checkpoint_root/$1"
    [[ -f "$path/_CHECKPOINT_METADATA" \
        && -f "$path/params/manifest.ocdbt" \
        && -n "$(find "$path/assets" -type f -name norm_stats.json -print -quit 2>/dev/null)" ]]
}

cached_checkpoint_is_complete() {
    local path="$cache_root/$1"
    [[ -f "$path/_CHECKPOINT_METADATA" \
        && -f "$path/params/manifest.ocdbt" \
        && -n "$(find "$path/assets" -type f -name norm_stats.json -print -quit 2>/dev/null)" ]]
}

ensure_cached_checkpoint() {
    local step="$1" source="$checkpoint_root/$step" final="$cache_root/$step"
    local partial="$cache_root/.$step.partial" lock="$cache_root/.$step.lock"
    mkdir -p "$cache_root"
    (
        flock -x 9
        cached_checkpoint_is_complete "$step" && exit 0
        checkpoint_is_complete "$step" || exit 2
        rm -rf -- "$partial"
        mkdir -p "$partial"
        log "caching checkpoint=$step params+assets from NAS"
        rsync -a --delete "$source/params" "$source/assets" "$source/_CHECKPOINT_METADATA" "$partial/" \
            || exit 3
        [[ -f "$partial/params/manifest.ocdbt" \
            && -n "$(find "$partial/assets" -type f -name norm_stats.json -print -quit 2>/dev/null)" ]] \
            || exit 4
        rm -rf -- "$final"
        mv "$partial" "$final"
        log "cached checkpoint=$step bytes=$(du -sb "$final" | awk '{print $1}')"
    ) 9>"$lock"
}

available_steps_desc() {
    local path step run_dir started
    for path in "$checkpoint_root"/[0-9]*; do
        [[ -d "$path" ]] || continue
        step="${path##*/}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        checkpoint_is_complete "$step" || continue
        run_dir=$(printf '%s/runs/checkpoint_%06d' "$results_root" "$step")
        started=0
        if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" \
            \( -type f -name .complete -o -type d -name .claim -o -type d -name attempts \) \
            -print -quit 2>/dev/null)" ]]; then
            started=1
        fi
        printf '%s %s\n' "$started" "$step"
    done | sort -k1,1nr -k2,2nr | awk '{print $2}'
}

job_dir_for() {
    printf '%s/runs/checkpoint_%06d/h_%02d/%s' "$results_root" "$1" "$2" "$3"
}

retry_is_cooling_down() {
    local marker="$1"
    [[ -f "$marker" ]] || return 1
    local age=$(( $(date +%s) - $(stat -c %Y "$marker") ))
    (( age < retry_cooldown ))
}

claim_job() {
    local job_dir="$1" worker_id="$2" worker_pid="$3"
    local claim="$job_dir/.claim" reclaim_lock="$job_dir/.claim_reclaim"
    local owner_version="" owner_host="" owner_pid="" owner_worker="" owner_time=""
    local owner_line="" marker age marker_mtime current_owner_line current_marker current_marker_mtime reclaim_age
    local reclaim_acquired=false
    mkdir -p "$job_dir"
    if mkdir "$claim" 2>/dev/null; then
        if ! printf 'v2 %s %s %s %s\n' \
            "$worker_host" "$worker_pid" "$worker_id" "$(date -Is)" >"$claim/owner"; then
            rm -rf -- "$claim"
            return 1
        fi
        touch "$claim/heartbeat"
        return 0
    fi

    [[ -r "$claim/owner" ]] && owner_line="$(cat "$claim/owner" 2>/dev/null || true)"
    read -r owner_version owner_host owner_pid owner_worker owner_time <<<"$owner_line"
    marker="$claim"
    [[ -e "$claim/heartbeat" ]] && marker="$claim/heartbeat"
    marker_mtime="$(stat -c %Y "$marker" 2>/dev/null || printf '0')"
    age=$(( $(date +%s) - marker_mtime ))
    # v2 claims are cross-host safe: a local owner is checked by PID while a
    # remote owner is protected by its heartbeat.  Legacy PID-only claims are
    # preserved for a conservative timeout during rolling upgrades.
    local reclaim=false
    if [[ "$owner_version" == "v2" ]]; then
        if [[ "$owner_host" == "$worker_host" && "$owner_pid" =~ ^[0-9]+$ ]]; then
            kill -0 "$owner_pid" 2>/dev/null || reclaim=true
        elif [[ -n "$owner_host" && "$owner_pid" =~ ^[0-9]+$ ]]; then
            (( age > claim_stale_seconds )) && reclaim=true
        elif (( age > claim_init_grace_seconds )); then
            reclaim=true
        fi
    elif [[ "$owner_version" =~ ^[0-9]+$ ]]; then
        (( age > legacy_claim_stale_seconds )) && reclaim=true
    elif (( age > claim_init_grace_seconds )); then
        reclaim=true
    fi
    if [[ "$reclaim" == true ]]; then
        reclaim_acquired=false
        if mkdir "$reclaim_lock" 2>/dev/null; then
            reclaim_acquired=true
        else
            reclaim_age=$(( $(date +%s) - $(stat -c %Y "$reclaim_lock" 2>/dev/null || printf '0') ))
            (( reclaim_age > claim_stale_seconds )) && rmdir "$reclaim_lock" 2>/dev/null || true
            mkdir "$reclaim_lock" 2>/dev/null && reclaim_acquired=true || true
        fi
        if [[ "$reclaim_acquired" == true ]]; then
            current_owner_line=""
            [[ -r "$claim/owner" ]] && current_owner_line="$(cat "$claim/owner" 2>/dev/null || true)"
            current_marker="$claim"
            [[ -e "$claim/heartbeat" ]] && current_marker="$claim/heartbeat"
            current_marker_mtime="$(stat -c %Y "$current_marker" 2>/dev/null || printf '0')"
            # Remove only the exact stale generation we inspected.  Another
            # worker may already have replaced it with a fresh claim.
            if [[ "$current_owner_line" == "$owner_line" \
                && "$current_marker_mtime" == "$marker_mtime" ]]; then
                rm -rf -- "$claim"
            fi
            rmdir "$reclaim_lock" 2>/dev/null || true
        fi
        if mkdir "$claim" 2>/dev/null; then
            if ! printf 'v2 %s %s %s %s\n' \
                "$worker_host" "$worker_pid" "$worker_id" "$(date -Is)" >"$claim/owner"; then
                rm -rf -- "$claim"
                return 1
            fi
            touch "$claim/heartbeat"
            return 0
        fi
    fi
    return 1
}

claim_is_owned_by() {
    local claim="$1" worker_pid="$2"
    local owner_version="" owner_host="" owner_pid="" owner_worker="" owner_time=""
    [[ -r "$claim/owner" ]] || return 1
    read -r owner_version owner_host owner_pid owner_worker owner_time <"$claim/owner" || return 1
    if [[ "$owner_version" == "v2" ]]; then
        [[ "$owner_host" == "$worker_host" && "$owner_pid" == "$worker_pid" ]]
    else
        [[ "$owner_version" == "$worker_pid" ]]
    fi
}

claim_heartbeat_pid=""
start_claim_heartbeat() {
    local step="$1" horizon="$2" task="$3" worker_pid="$4"
    local claim="$(job_dir_for "$step" "$horizon" "$task")/.claim"
    (
        while kill -0 "$worker_pid" 2>/dev/null && claim_is_owned_by "$claim" "$worker_pid"; do
            touch "$claim/heartbeat" 2>/dev/null || exit 0
            sleep "$claim_heartbeat_seconds"
        done
    ) >/dev/null 2>&1 &
    claim_heartbeat_pid="$!"
}

stop_claim_heartbeat() {
    local heartbeat_pid="${1:-}"
    [[ "$heartbeat_pid" =~ ^[0-9]+$ ]] || return 0
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
}

release_job_claim() {
    local step="$1" horizon="$2" task="$3" worker_pid="$4"
    local claim="$(job_dir_for "$step" "$horizon" "$task")/.claim"
    claim_is_owned_by "$claim" "$worker_pid" && rm -rf -- "$claim"
}

checkpoint_jobs_complete() {
    local step="$1" horizon task
    for horizon in "${horizons[@]}"; do
        for task in "${tasks[@]}"; do
            [[ -f "$(job_dir_for "$step" "$horizon" "$task")/.complete" ]] || return 1
        done
    done
    return 0
}

all_expected_jobs_complete() {
    local step
    for step in "${expected_steps[@]}"; do
        checkpoint_jobs_complete "$step" || return 1
    done
    return 0
}

cleanup_cache_if_step_complete() {
    local step="$1" final="$cache_root/$step" lock="$cache_root/.$step.lock"
    checkpoint_jobs_complete "$step" || return 0
    (
        flock -x 9
        checkpoint_jobs_complete "$step" || exit 0
        if [[ -d "$final" ]]; then
            rm -rf -- "$final"
            log "released completed checkpoint cache=$step"
        fi
    ) 9>"$lock"
}

next_job() {
    local worker_id="$1" worker_pid="$2" step h task hi ti job_dir
    while IFS= read -r step; do
        for hi in "${!horizons[@]}"; do
            h="${horizons[$hi]}"
            for ti in "${!tasks[@]}"; do
                task="${tasks[$ti]}"
                job_dir="$(job_dir_for "$step" "$h" "$task")"
                [[ -f "$job_dir/.complete" ]] && continue
                retry_is_cooling_down "$job_dir/.retry_after" && continue
                claim_job "$job_dir" "$worker_id" "$worker_pid" || continue
                printf '%s %s %s\n' "$step" "$h" "$task"
                return 0
            done
        done
    done < <(available_steps_desc)
    return 1
}

update_report() {
    local lock="$results_root/.report.lock" lock_pid="$BASHPID"
    local reclaim_lock="$results_root/.report.lock.reclaim"
    local owner_version="" owner_host="" owner_pid="" owner_worker="" owner_time=""
    local owner_line="" marker age marker_mtime waited=0 reclaim_age current_owner_line current_marker current_marker_mtime
    local reclaim_acquired=false
    while ! mkdir "$lock" 2>/dev/null; do
        owner_version="" owner_host="" owner_pid="" owner_worker="" owner_time=""
        owner_line=""
        [[ -r "$lock/owner" ]] && owner_line="$(cat "$lock/owner" 2>/dev/null || true)"
        read -r owner_version owner_host owner_pid owner_worker owner_time <<<"$owner_line"
        marker="$lock"
        [[ -e "$lock/heartbeat" ]] && marker="$lock/heartbeat"
        marker_mtime="$(stat -c %Y "$marker" 2>/dev/null || printf '0')"
        age=$(( $(date +%s) - marker_mtime ))
        if [[ "$owner_version" == "v2" && "$owner_host" == "$worker_host" \
            && "$owner_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$owner_pid" 2>/dev/null; then
            age=$(( report_lock_stale_seconds + 1 ))
        fi
        if (( age > report_lock_stale_seconds )); then
            reclaim_acquired=false
            if mkdir "$reclaim_lock" 2>/dev/null; then
                reclaim_acquired=true
            else
                reclaim_age=$(( $(date +%s) - $(stat -c %Y "$reclaim_lock" 2>/dev/null || printf '0') ))
                (( reclaim_age > report_lock_stale_seconds )) && rmdir "$reclaim_lock" 2>/dev/null || true
                mkdir "$reclaim_lock" 2>/dev/null && reclaim_acquired=true || true
            fi
            if [[ "$reclaim_acquired" == true ]]; then
                current_owner_line=""
                [[ -r "$lock/owner" ]] && current_owner_line="$(cat "$lock/owner" 2>/dev/null || true)"
                current_marker="$lock"
                [[ -e "$lock/heartbeat" ]] && current_marker="$lock/heartbeat"
                current_marker_mtime="$(stat -c %Y "$current_marker" 2>/dev/null || printf '0')"
                if [[ "$current_owner_line" == "$owner_line" \
                    && "$current_marker_mtime" == "$marker_mtime" ]]; then
                    rm -rf -- "$lock"
                fi
                rmdir "$reclaim_lock" 2>/dev/null || true
            fi
            continue
        fi
        (( waited++ ))
        (( waited < report_lock_stale_seconds )) || return 1
        sleep 1
    done
    printf 'v2 %s %s report %s\n' "$worker_host" "$lock_pid" "$(date -Is)" >"$lock/owner"
    touch "$lock/heartbeat"
    local rc=0
    "$python_bin" "$repo/scripts/summarize_all10_checkpoint_horizon_eval.py" \
        --results-root "$results_root" \
        --checkpoint-root "$checkpoint_root" \
        --expected-final-step "$expected_final_step" \
        --expected-episodes "$episodes" \
        --protocol-revision "$protocol_revision" \
        >>"$results_root/logs/report.log" 2>&1 || rc=$?
    local current_version="" current_host="" current_pid="" current_worker="" current_time=""
    [[ -r "$lock/owner" ]] \
        && read -r current_version current_host current_pid current_worker current_time <"$lock/owner" || true
    if [[ "$current_version" == "v2" && "$current_host" == "$worker_host" \
        && "$current_pid" == "$lock_pid" ]]; then
        rm -rf -- "$lock"
    fi
    return "$rc"
}

run_job() {
    local worker_id="$1" step="$2" horizon="$3" task="$4"
    local task_index=0 i seed job_dir attempt attempt_dir log_file rc
    for i in "${!tasks[@]}"; do
        [[ "${tasks[$i]}" == "$task" ]] && task_index="$i"
    done
    seed=$((832500 + task_index * 1000))
    job_dir="$(job_dir_for "$step" "$horizon" "$task")"
    mkdir -p "$job_dir/attempts"
    if ! ensure_cached_checkpoint "$step"; then
        printf '[%s] CACHE_FAILED worker=%s checkpoint=%s\n' \
            "$(date -Is)" "$worker_id" "$step" | tee -a "$results_root/queue.log"
        date -Is >"$job_dir/.retry_after"
        return 1
    fi

    for attempt in 1 2 3; do
        attempt_dir="$job_dir/attempts/$(date +%Y%m%dT%H%M%S)_worker${worker_id}_attempt${attempt}"
        log_file="$attempt_dir/eval.log"
        mkdir -p "$attempt_dir"
        printf '[%s] START worker=%s checkpoint=%s h=%s task=%s seed=%s attempt=%s\n' \
            "$(date -Is)" "$worker_id" "$step" "$horizon" "$task" "$seed" "$attempt" \
            | tee -a "$results_root/queue.log" >"$log_file"
        (
            cd "$pi05" || exit 1
            export PYTHONUNBUFFERED=1
            export PYTHONFAULTHANDLER=1
            export EMBODICHAIN_SIM_EXIT_PROCESS=0
            export MALLOC_ARENA_MAX=2
            export XLA_PYTHON_CLIENT_MEM_FRACTION="$xla_mem_fraction"
            export JAX_COMPILATION_CACHE_DIR="${ALL10_EVAL_JAX_CACHE:-/workspace/shared/.cache/jax-all10-eval}"
            export PI05_CHECKPOINT_RUN_ROOT="$cache_root"
            export EMBODICHAIN_ROOT="$embodichain_root"
            export PYTHON_BIN="$python_bin"
            if [[ -d "$runtime_overlay" ]]; then
                export PYTHONPATH="$runtime_overlay:$embodichain_root/embodichain_tasks:$embodichain_root${PYTHONPATH:+:$PYTHONPATH}"
            fi
            bash eval.sh "$task" random_3p "$config" "$model" 0 \
                --checkpoint_id "$step" \
                --pi0_step "$horizon" \
                --max_episodes "$episodes" \
                --seed "$seed" \
                --headless true \
                --pytorch_device cuda \
                --eval_video_log true \
                --eval_video_obs_keys "['cam_left_wrist','cam_right_wrist','cam_high','cam_third']" \
                --eval_result_dir "$attempt_dir/results"
        ) >>"$log_file" 2>&1
        rc=$?

        if (( rc == 0 )) && "$python_bin" \
            "$repo/scripts/verify_all10_checkpoint_horizon_eval.py" \
            --result-root "$attempt_dir/results" \
            --output "$job_dir/job_result.json" \
            --complete-marker "$job_dir/.complete" \
            --checkpoint "$step" \
            --horizon "$horizon" \
            --task "$task" \
            --seed "$seed" \
            --expected-episodes "$episodes" \
            --protocol-revision "$protocol_revision" >>"$log_file" 2>&1; then
            rm -f "$job_dir/.retry_after"
            printf '[%s] COMPLETE worker=%s checkpoint=%s h=%s task=%s\n' \
                "$(date -Is)" "$worker_id" "$step" "$horizon" "$task" \
                | tee -a "$results_root/queue.log" >>"$log_file"
            update_report
            cleanup_cache_if_step_complete "$step"
            return 0
        fi

        printf '[%s] FAILED worker=%s checkpoint=%s h=%s task=%s attempt=%s rc=%s\n' \
            "$(date -Is)" "$worker_id" "$step" "$horizon" "$task" "$attempt" "$rc" \
            | tee -a "$results_root/queue.log" >>"$log_file"
        sleep 15
    done

    date -Is >"$job_dir/.retry_after"
    return 1
}

worker_loop() {
    local worker_id="$1" worker_pid="$BASHPID" job step horizon task heartbeat_pid
    log "worker=$worker_id started worker_count=$worker_count"
    while true; do
        job="$(next_job "$worker_id" "$worker_pid")"
        if [[ -n "$job" ]]; then
            read -r step horizon task <<<"$job"
            start_claim_heartbeat "$step" "$horizon" "$task" "$worker_pid"
            heartbeat_pid="$claim_heartbeat_pid"
            run_job "$worker_id" "$step" "$horizon" "$task" || true
            stop_claim_heartbeat "$heartbeat_pid"
            release_job_claim "$step" "$horizon" "$task" "$worker_pid"
            continue
        fi

        update_report
        if checkpoint_is_complete "$expected_final_step" && all_expected_jobs_complete; then
            log "worker=$worker_id observed all expected jobs complete; exiting"
            return 0
        fi
        log "worker=$worker_id waiting for checkpoints or retry cooldown"
        sleep "$poll_seconds"
    done
}

write_manifest() {
    # The shared results directory may be mounted at a different local path on
    # each host.  Preserve the first authoritative manifest instead of letting
    # later workers rewrite it with host-local paths or worker counts. Rewrite
    # only when the terminal checkpoint contract itself has changed.
    if [[ -f "$results_root/manifest.json" ]] && "$python_bin" - \
        "$results_root/manifest.json" "$expected_final_step" <<'PY'
import json, sys
try:
    manifest = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if manifest.get("expected_final_step") == int(sys.argv[2]) else 1)
PY
    then
        return 0
    fi
    "$python_bin" - "$results_root/manifest.json" "$checkpoint_root" "$episodes" "$worker_count" "$protocol_revision" "$expected_final_step" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

out, checkpoint_root, episodes, workers, protocol_revision, expected_final_step = sys.argv[1:]
expected_final_step = int(expected_final_step)
expected_checkpoints = list(range(2500, expected_final_step, 2500))
expected_checkpoints.append(expected_final_step)
payload = {
    "schema_version": 1,
    "protocol_revision": protocol_revision,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "checkpoint_root": checkpoint_root,
    "checkpoint_interval": 2500,
    "expected_final_step": expected_final_step,
    "expected_checkpoints": expected_checkpoints,
    "tasks": "click_bell drawer_open_place handle_basket item_assembly items_handover manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring".split(),
    "execution_horizons": [10, 30, 50, 64],
    "episodes_per_job": int(episodes),
    "paired_seed_base": 832500,
    "paired_seed_task_stride": 1000,
    "setting": "random_3p",
    "camera_keys": ["cam_left_wrist", "cam_right_wrist", "cam_high", "cam_third"],
    "video_layout": "horizontal_four_view_composite_2560x480",
    "workers": int(workers),
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=".manifest.", dir=str(Path(out).parent))
with os.fdopen(fd, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
os.replace(tmp, out)
PY
}

main() {
    [[ -x "$python_bin" ]] || { echo "python missing: $python_bin" >&2; exit 2; }
    [[ -f "$repo/scripts/eval_policy_parallel.py" ]] || { echo "repo incomplete: $repo" >&2; exit 2; }
    [[ "$worker_count" =~ ^[1-9][0-9]*$ ]] || { echo "invalid workers: $worker_count" >&2; exit 2; }
    mkdir -p "${ALL10_EVAL_JAX_CACHE:-/workspace/shared/.cache/jax-all10-eval}" "$cache_root"
    write_manifest
    update_report
    log "supervisor started workers=$worker_count results=$results_root checkpoints=$checkpoint_root"

    local pids=() worker
    for ((worker=0; worker<worker_count; worker++)); do
        worker_loop "$worker" >>"$results_root/logs/worker_${worker}.log" 2>&1 &
        pids+=("$!")
    done
    local rc=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    update_report
    if (( rc == 0 )) && checkpoint_is_complete "$expected_final_step" && all_expected_jobs_complete; then
        touch "$results_root/.complete"
        log "all workers complete"
    fi
    return "$rc"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
