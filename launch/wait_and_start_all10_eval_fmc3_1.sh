#!/usr/bin/env bash
# Resume the cross-host evaluation worker only after the fmc3-1 GPU has been
# continuously free.  This keeps approved SLAM/YOLO workloads ahead of eval.
set -uo pipefail

repo="${ALL10_EVAL_REPO:-/home/fmc3/workspace/RoboSynChallenge-eval-all10}"
embodichain="${ALL10_EVAL_EMBODICHAIN_ROOT:-/home/fmc3/workspace/EmbodiChain-eval-all10}"
checkpoint_root="${ALL10_EVAL_CHECKPOINT_ROOT:-/home/fmc3/FermiBotNas/models/RoboSynChallenge/pi05_all10_h64_expert/all10_expert_base_h64_bs64_steps100k}"
results_root="${ALL10_EVAL_RESULTS_ROOT:-/home/fmc3/FermiBotNas/models/RoboSynChallenge/pi05_all10_h64_expert/all10_expert_base_h64_bs64_steps100k_eval_all_ckpts_h10_30_50_64_20eps_4view}"
cache_root="${ALL10_EVAL_CACHE_ROOT:-/tmp/pi05_all10_h64_eval_checkpoints_fmc3_1}"
jax_cache="${ALL10_EVAL_JAX_CACHE:-/home/fmc3/workspace/.cache/jax-all10-eval}"
eval_unit="${ALL10_EVAL_FMC3_UNIT:-pi05-all10-eval-fmc3-1}"
idle_seconds="${ALL10_EVAL_RESUME_IDLE_SECONDS:-300}"
poll_seconds="${ALL10_EVAL_RESUME_POLL_SECONDS:-30}"
nvidia_smi="${ALL10_EVAL_NVIDIA_SMI:-/usr/bin/nvidia-smi}"
dry_run="${ALL10_EVAL_RESUME_DRY_RUN:-0}"
resume_once="${ALL10_EVAL_RESUME_ONCE:-0}"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

valid_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

start_eval() {
    if [[ "$dry_run" == 1 ]]; then
        log "dry-run: would start unit=$eval_unit after idle=${idle_seconds}s"
        return 0
    fi

    install -d -o fmc3 -g fmc3 "$cache_root" "$jax_cache"
    systemctl reset-failed "$eval_unit.service" 2>/dev/null || true
    systemd-run \
        --unit="$eval_unit" \
        --description="pi05 all10 cross-host evaluation worker on fmc3-1" \
        --property=User=fmc3 \
        --property=Group=fmc3 \
        --property=WorkingDirectory="$repo" \
        --property=Restart=on-failure \
        --property=RestartSec=60 \
        --property=OOMPolicy=continue \
        --property=Nice=5 \
        /usr/bin/env \
        ALL10_EVAL_REPO="$repo" \
        ALL10_EVAL_CHECKPOINT_ROOT="$checkpoint_root" \
        ALL10_EVAL_RESULTS_ROOT="$results_root" \
        ALL10_EVAL_CACHE_ROOT="$cache_root" \
        ALL10_EVAL_PYTHON="$repo/policy/pi05/.venv/bin/python" \
        ALL10_EVAL_RUNTIME_OVERLAY=/nonexistent \
        ALL10_EVAL_EMBODICHAIN_ROOT="$embodichain" \
        ALL10_EVAL_JAX_CACHE="$jax_cache" \
        ALL10_EVAL_WORKERS=1 \
        ALL10_EVAL_XLA_MEM_FRACTION=0.15 \
        ALL10_EVAL_FINAL_STEP=99999 \
        ALL10_EVAL_EPISODES=20 \
        ALL10_EVAL_POLL_SECONDS=120 \
        ALL10_EVAL_RETRY_COOLDOWN=900 \
        ALL10_EVAL_PROTOCOL_REVISION=all10_h64_v2_bounded_texture_pool \
        /bin/bash "$repo/launch/eval_all10_h64_all_checkpoints_pro6000.sh"
}

main() {
    valid_nonnegative_integer "$idle_seconds" || {
        echo "invalid idle seconds: $idle_seconds" >&2
        return 2
    }
    valid_nonnegative_integer "$poll_seconds" && (( poll_seconds > 0 )) || {
        echo "invalid poll seconds: $poll_seconds" >&2
        return 2
    }
    [[ -x "$nvidia_smi" ]] || {
        echo "nvidia-smi missing: $nvidia_smi" >&2
        return 2
    }
    [[ -x "$repo/policy/pi05/.venv/bin/python" || "$dry_run" == 1 ]] || {
        echo "evaluation environment missing: $repo" >&2
        return 2
    }

    local idle_since=0 now gpu_pids
    log "autoresume watcher started idle_required=${idle_seconds}s unit=$eval_unit"
    while true; do
        if [[ -f "$results_root/.complete" ]]; then
            log "shared sweep is already complete; exiting"
            return 0
        fi
        if systemctl is-active --quiet "$eval_unit.service"; then
            idle_since=0
            if [[ "$resume_once" == 1 ]]; then
                log "evaluation unit is already active; exiting one-shot watcher"
                return 0
            fi
            sleep "$poll_seconds"
            continue
        fi

        if ! gpu_pids="$($nvidia_smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
            idle_since=0
            log "GPU query failed; treating device as busy"
        elif [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
            idle_since=0
            log "GPU busy compute_pids=$(tr '\n' ',' <<<"$gpu_pids" | sed 's/,$//')"
        else
            now="$(date +%s)"
            (( idle_since == 0 )) && idle_since="$now"
            if (( now - idle_since >= idle_seconds )); then
                log "GPU continuously idle; starting evaluation worker"
                if start_eval; then
                    idle_since=0
                    [[ "$resume_once" == 1 ]] && return 0
                else
                    idle_since=0
                    log "evaluation start failed; will retry after a new idle window"
                fi
            fi
            log "GPU idle elapsed=$(( now - idle_since ))s/${idle_seconds}s"
        fi
        sleep "$poll_seconds"
    done
}

main "$@"
