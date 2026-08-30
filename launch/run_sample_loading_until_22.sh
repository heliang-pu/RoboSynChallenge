#!/usr/bin/env bash
# Run the validated single-GPU sample_loading PPO probe until an absolute deadline.
# This is a one-host supervisor: it waits for an exclusive GPU, bounds disk usage by
# retaining only this run's two newest complete checkpoints, and resumes a paused
# evaluation-suite parent when training ends.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLINF_ROOT="${RLINF_ROOT:-$HOME/workspace/RLinf}"
BASE_MODEL="${ROBOSYN_PI05_TORCH_CKPT:-$HOME/workspace/models/pi05_pt/sample_loading_28000}"
DEADLINE="${RLINF_DEADLINE:-$(date +%F) 22:00:00}"
DEADLINE_EPOCH="$(date -d "$DEADLINE" +%s)"
RUN_TAG="$(date -d "@$DEADLINE_EPOCH" +%Y%m%d_%H%M)"
CONTROL_DIR="$RLINF_ROOT/scheduled/sample_loading_until_$RUN_TAG"
STATUS_FILE="$CONTROL_DIR/status.env"
SUPERVISOR_LOG="$CONTROL_DIR/supervisor.log"
PAUSE_PID="${RLINF_PAUSE_PID:-}"
MIN_FREE_MB="${RLINF_MIN_FREE_GPU_MB:-44000}"
SAVE_INTERVAL="${SAVE_INTER:-2}"
KEEP_CHECKPOINTS="${RLINF_KEEP_CHECKPOINTS:-2}"
MIN_DISK_FREE_GB="${RLINF_MIN_DISK_FREE_GB:-25}"

mkdir -p "$CONTROL_DIR"
exec >>"$SUPERVISOR_LOG" 2>&1

TRAIN_PID=""
PAUSED=0
RUN_DIR=""

terminate_training() {
    if [[ -z "$TRAIN_PID" ]] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        return
    fi

    echo "$(date --iso-8601=seconds) terminating training process group pgid=$TRAIN_PID"
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || kill -TERM "$TRAIN_PID" 2>/dev/null || true
    for _ in {1..12}; do
        kill -0 "$TRAIN_PID" 2>/dev/null || return
        sleep 5
    done
    echo "$(date --iso-8601=seconds) training did not stop after 60s; sending KILL"
    kill -KILL -- "-$TRAIN_PID" 2>/dev/null || kill -KILL "$TRAIN_PID" 2>/dev/null || true
}

write_status() {
    local state="$1"
    local tmp="$STATUS_FILE.tmp"
    {
        echo "STATE=$state"
        echo "UPDATED_AT=$(date --iso-8601=seconds)"
        echo "DEADLINE=$(date -d "@$DEADLINE_EPOCH" --iso-8601=seconds)"
        echo "BASE_MODEL=$BASE_MODEL"
        echo "RUN_DIR=$RUN_DIR"
        echo "TRAIN_PID=$TRAIN_PID"
        echo "PAUSED_SUITE_PID=$PAUSE_PID"
    } >"$tmp"
    mv "$tmp" "$STATUS_FILE"
}

resume_suite() {
    if [[ "$PAUSED" == "1" && -n "$PAUSE_PID" ]] && kill -0 "$PAUSE_PID" 2>/dev/null; then
        kill -CONT "$PAUSE_PID" || true
        echo "$(date --iso-8601=seconds) resumed evaluation suite pid=$PAUSE_PID"
    fi
}

cleanup() {
    local code=$?
    trap - EXIT TERM INT
    terminate_training
    resume_suite
    if [[ "$code" == "0" || "$code" == "124" ]]; then
        write_status "finished"
    else
        write_status "failed_$code"
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "$(date --iso-8601=seconds) supervisor start; deadline=$(date -d "@$DEADLINE_EPOCH" --iso-8601=seconds)"
echo "base_model=$BASE_MODEL save_interval=$SAVE_INTERVAL keep=$KEEP_CHECKPOINTS"

[[ -f "$BASE_MODEL/model.safetensors" ]] || { echo "missing $BASE_MODEL/model.safetensors"; exit 2; }
[[ -d "$BASE_MODEL/assets/RoboSynChallenge/cobotmagic_Sim_sample_loading" ]] \
    || { echo "missing sample_loading norm stats under $BASE_MODEL/assets"; exit 2; }

if (( $(date +%s) >= DEADLINE_EPOCH )); then
    echo "deadline is not in the future"
    exit 2
fi

# Pause only the suite parent. Its current evaluator child is allowed to finish, so
# the completed 100-episode cell remains valid. SIGCONT in cleanup makes this reversible.
if [[ -n "$PAUSE_PID" ]] && kill -0 "$PAUSE_PID" 2>/dev/null; then
    kill -STOP "$PAUSE_PID"
    PAUSED=1
    echo "$(date --iso-8601=seconds) paused evaluation suite parent pid=$PAUSE_PID"
fi

write_status "waiting_for_gpu"
stable=0
while (( $(date +%s) < DEADLINE_EPOCH )); do
    free_mb="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
        stable=$((stable + 1))
        echo "$(date --iso-8601=seconds) GPU free check $stable/3: ${free_mb}MB"
        (( stable >= 3 )) && break
    else
        stable=0
        echo "$(date --iso-8601=seconds) waiting for GPU: ${free_mb:-unknown}MB free"
    fi
    sleep 30
done

if (( stable < 3 )); then
    echo "deadline reached before GPU became available"
    exit 3
fi

remaining=$((DEADLINE_EPOCH - $(date +%s)))
if (( remaining < 900 )); then
    echo "less than 15 minutes remain; refusing to start a step that cannot finish"
    exit 3
fi

free_gb="$(df --output=avail -BG "$RLINF_ROOT" | tail -n1 | tr -dc '0-9')"
if (( free_gb < MIN_DISK_FREE_GB )); then
    echo "only ${free_gb}GB disk free; require ${MIN_DISK_FREE_GB}GB"
    exit 4
fi

before_epoch="$(date +%s)"
write_status "starting"
echo "$(date --iso-8601=seconds) launching PPO; hard runtime=${remaining}s"

# Give the whole RLinf/Ray tree its own process group. This lets the disk guard and
# supervisor cleanup stop the complete tree instead of only its immediate shell.
set +e
setsid env \
    ROBOSYN_TASK=sample_loading \
    ROBOSYN_PI05_TORCH_CKPT="$BASE_MODEL" \
    RLINF_ROOT="$RLINF_ROOT" \
    STEPS=200 \
    SAVE_INTER="$SAVE_INTERVAL" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    timeout --signal=TERM --kill-after=120s "${remaining}s" \
        bash "$REPO_ROOT/launch/rlinf_train.sh" probe &
TRAIN_PID=$!
write_status "running"

# Monitor the detached RLinf run and prune only checkpoints created by this run.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if [[ -z "$RUN_DIR" ]]; then
        RUN_DIR="$(find "$RLINF_ROOT/logs" -maxdepth 1 -type d \
            -name '*-robosynchallenge_ppo_pi05_probe' -newermt "@$before_epoch" \
            -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
        [[ -n "$RUN_DIR" ]] && write_status "running"
    fi

    if [[ -n "$RUN_DIR" ]]; then
        checkpoint_root="$RUN_DIR/sample_loading_probe_ppo_pi05/checkpoints"
        mapfile -t complete < <(
            find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' \
                -printf '%f\n' 2>/dev/null \
            | while read -r name; do
                dir="$checkpoint_root/$name"
                [[ -s "$dir/actor/model_state_dict/full_weights.pt" ]] \
                    && [[ -s "$dir/actor/dcp_checkpoint/__0_0.distcp" ]] \
                    && echo "${name#global_step_} $dir"
              done \
            | sort -n | cut -d' ' -f2-
        )
        count=${#complete[@]}
        if (( count > KEEP_CHECKPOINTS )); then
            remove_count=$((count - KEEP_CHECKPOINTS))
            for ((i=0; i<remove_count; i++)); do
                old="${complete[$i]}"
                case "$old" in
                    "$checkpoint_root"/global_step_*)
                        echo "$(date --iso-8601=seconds) pruning superseded checkpoint $old"
                        rm -rf -- "$old"
                        ;;
                    *) echo "refusing unsafe prune target: $old"; exit 5 ;;
                esac
            done
        fi
        if (( count > 0 )); then
            latest="${complete[$((count - 1))]}"
            ln -sfn "$latest" "$CONTROL_DIR/latest_checkpoint"
        fi
    fi

    free_gb="$(df --output=avail -BG "$RLINF_ROOT" | tail -n1 | tr -dc '0-9')"
    if (( free_gb < MIN_DISK_FREE_GB )); then
        echo "$(date --iso-8601=seconds) disk guard fired at ${free_gb}GB; terminating training"
        kill -TERM -- "-$TRAIN_PID" 2>/dev/null || kill -TERM "$TRAIN_PID" 2>/dev/null || true
    fi
    sleep 30
done

wait "$TRAIN_PID"
train_code=$?
set -e
TRAIN_PID=""

echo "$(date --iso-8601=seconds) training process exit=$train_code"
if [[ "$train_code" != "0" && "$train_code" != "124" ]]; then
    exit "$train_code"
fi
exit 0
