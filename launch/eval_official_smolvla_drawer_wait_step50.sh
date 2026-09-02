#!/usr/bin/env bash
# Wait for an idle PRO6000, then evaluate the official drawer_open_place
# SmolVLA revision with the official random task distribution. cam_third is
# appended only for recording; the policy still consumes the released three
# camera inputs.
set -euo pipefail

repo="${SMOLVLA_EVAL_REPO:-/workspace/shared/RoboSynChallenge-eval-all10}"
embodichain_root="${SMOLVLA_EMBODICHAIN_ROOT:-/workspace/shared/EmbodiChain-eval-all10}"
checkpoint="${SMOLVLA_CHECKPOINT:-/workspace/shared/checkpoints/RoboSynChallenge_SmolVLA_sim_drawer_open_place/c0088d84a568f93fb4401aabafcc41cf643efcdd_lr044_compat}"
worker_python="${SMOLVLA_PYTHON:-/workspace/shared/.venvs/smolvla-worker-lr044/bin/python}"
results_root="${SMOLVLA_RESULTS_ROOT:-/mnt/FermiBotNas/models/RoboSynChallenge/official_smolvla_drawer_open_place_eval/c0088d84a568f93fb4401aabafcc41cf643efcdd/step50}"
gpu_id="${SMOLVLA_GPU_ID:-0}"
poll_seconds="${SMOLVLA_WAIT_POLL_SECONDS:-30}"
idle_checks_required="${SMOLVLA_IDLE_CHECKS_REQUIRED:-4}"
max_existing_gpu_memory_mib="${SMOLVLA_MAX_EXISTING_GPU_MEMORY_MIB:-8192}"
smoke_root="$results_root/smoke_1ep_random_4view_attempt3"
formal_root="$results_root/formal_20eps_random_seed0_4view_attempt2"
log_file="$results_root/wait_and_eval.log"
claim_dir="$results_root/.runner_claim"
claim_heartbeat_seconds="${SMOLVLA_CLAIM_HEARTBEAT_SECONDS:-15}"
claim_owner="$(hostname -f 2>/dev/null || hostname) $BASHPID"
claim_heartbeat_pid=""

mkdir -p "$results_root"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$log_file"
}

gpu_used_memory_mib() {
    nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk '{ total += $1 } END { print total + 0 }'
}

wait_for_idle_gpu() {
    local idle_checks=0 used_memory_mib
    while (( idle_checks < idle_checks_required )); do
        if ! used_memory_mib="$(gpu_used_memory_mib)"; then
            idle_checks=0
            log "unable to query GPU memory; waiting"
        elif (( used_memory_mib <= max_existing_gpu_memory_mib )); then
            (( idle_checks += 1 ))
            log "GPU capacity check $idle_checks/$idle_checks_required: existing=${used_memory_mib} MiB, limit=${max_existing_gpu_memory_mib} MiB"
        else
            idle_checks=0
            log "GPU busy: existing=${used_memory_mib} MiB exceeds ${max_existing_gpu_memory_mib} MiB; waiting"
        fi
        (( idle_checks >= idle_checks_required )) || sleep "$poll_seconds"
    done
}

release_claim() {
    if [[ -n "$claim_heartbeat_pid" ]]; then
        kill "$claim_heartbeat_pid" 2>/dev/null || true
        wait "$claim_heartbeat_pid" 2>/dev/null || true
    fi
    if [[ -r "$claim_dir/owner" ]] \
        && [[ "$(cat "$claim_dir/owner" 2>/dev/null || true)" == "$claim_owner" ]]; then
        rm -rf -- "$claim_dir"
    fi
}

acquire_claim() {
    while ! mkdir "$claim_dir" 2>/dev/null; do
        if [[ -f "$formal_root/.complete" ]]; then
            log "formal evaluation completed by another host"
            exit 0
        fi
        log "another host owns evaluation claim; waiting"
        sleep "$poll_seconds"
    done
    printf '%s\n' "$claim_owner" >"$claim_dir/owner"
    touch "$claim_dir/heartbeat"
    (
        while true; do
            sleep "$claim_heartbeat_seconds"
            [[ -r "$claim_dir/owner" ]] || exit 0
            [[ "$(cat "$claim_dir/owner" 2>/dev/null || true)" == "$claim_owner" ]] || exit 0
            touch "$claim_dir/heartbeat"
        done
    ) &
    claim_heartbeat_pid="$!"
    trap release_claim EXIT INT TERM
    log "acquired evaluation claim owner=$claim_owner"
}

run_eval() {
    local episodes="$1" seed="$2" output_root="$3" output_log="$4"
    (
        cd "$repo"
        export EMBODICHAIN_ROOT="$embodichain_root"
        export EMBODICHAIN_SIM_EXIT_PROCESS=0
        export PYTHON_BIN="$repo/.venv/bin/python"
        export SMOLVLA_PYTHON="$worker_python"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export PYTHONPATH="$embodichain_root/embodichain_tasks:$embodichain_root:$repo/.venv/lib/python3.11/site-packages"
        bash policy/smolvla/eval.sh drawer_open_place random "$checkpoint" "$gpu_id" \
            --pytorch_device cuda \
            --headless true \
            --renderer auto \
            --max_episodes "$episodes" \
            --seed "$seed" \
            --eval_video_log true \
            --eval_video_obs_keys '["cam_left_wrist","cam_right_wrist","cam_high","cam_third"]' \
            --eval_add_third_camera true \
            --eval_result_dir "$output_root" \
            --smolvla_steps 50 \
            --smolvla_rescale_gripper true
    ) >"$output_log" 2>&1
}

verify_run() {
    local output_root="$1" expected_episodes="$2" verification_path="$3"
    "$repo/.venv/bin/python" - "$output_root" "$expected_episodes" "$verification_path" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
out = Path(sys.argv[3])
metrics_paths = sorted(root.rglob("evaluation_metrics.json"))
errors = []
if len(metrics_paths) != 1:
    errors.append(f"expected exactly one metrics file, found {len(metrics_paths)}")
    metrics = None
    videos = []
    tmp_videos = []
else:
    metrics = json.loads(metrics_paths[0].read_text())
    videos = sorted((metrics_paths[0].parent / "videos").glob("*.mp4"))
    tmp_videos = sorted((metrics_paths[0].parent / "videos").glob("*.tmp.mp4"))
    summary = metrics.get("summary", {})
    if int(summary.get("episode_count", -1)) != expected:
        errors.append(f"episode_count={summary.get('episode_count')}, expected={expected}")
    if len(videos) != expected:
        errors.append(f"video_count={len(videos)}, expected={expected}")
    if tmp_videos:
        errors.append(f"temporary videos remain: {len(tmp_videos)}")

probes = []
for video in videos:
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_frames,duration",
                "-of", "json", str(video),
            ],
            text=True,
        )
    )["streams"][0]
    probe["path"] = str(video)
    probes.append(probe)
    if probe.get("codec_name") != "h264":
        errors.append(f"{video.name}: codec={probe.get('codec_name')}")
    if (int(probe.get("width", 0)), int(probe.get("height", 0))) != (2560, 480):
        errors.append(f"{video.name}: size={probe.get('width')}x{probe.get('height')}")

payload = {
    "schema_version": 1,
    "validated_at": datetime.now(timezone.utc).isoformat(),
    "source_repo": "RoboSynChallenge/SmolVLA_sim_drawer_open_place",
    "source_revision": "c0088d84a568f93fb4401aabafcc41cf643efcdd",
    "model_safetensors_sha256": "7db7937d1e322e8e2416778320151d50714ff4ac9b1929061762b77fefb52e13",
    "protocol": {
        "task": "drawer_open_place",
        "setting": "random",
        "episodes": expected,
        "seed": 0,
        "smolvla_steps": 50,
        "smolvla_rescale_gripper": True,
        "policy_camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "recording_camera_keys": ["cam_left_wrist", "cam_right_wrist", "cam_high", "cam_third"],
        "video_layout": "horizontal_four_view_composite_2560x480",
    },
    "metrics_path": str(metrics_paths[0]) if len(metrics_paths) == 1 else None,
    "summary": metrics.get("summary") if metrics else None,
    "video_count": len(videos),
    "tmp_video_count": len(tmp_videos),
    "video_probes": probes,
    "errors": errors,
    "passed": not errors,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
PY
}

if [[ -f "$formal_root/.complete" ]]; then
    log "formal evaluation already complete"
    exit 0
fi

wait_for_idle_gpu
acquire_claim
if [[ -f "$smoke_root/.complete" ]]; then
    log "smoke already passed; reusing $smoke_root"
else
    log "starting 1-episode smoke"
    mkdir -p "$smoke_root"
    run_eval 1 0 "$smoke_root" "$smoke_root/eval.log"
    verify_run "$smoke_root" 1 "$smoke_root/verification.json"
    touch "$smoke_root/.complete"
    log "smoke passed"
fi

log "starting formal 20-episode evaluation"
mkdir -p "$formal_root"
run_eval 20 0 "$formal_root" "$formal_root/eval.log"
verify_run "$formal_root" 20 "$formal_root/verification.json"
touch "$formal_root/.complete"
log "formal evaluation passed"
