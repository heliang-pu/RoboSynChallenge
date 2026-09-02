#!/usr/bin/env bash
# Evaluate the final ACT sample_loading checkpoint with the exact LeRobot 0.3.3
# runtime used for training while EmbodiChain remains in its simulator venv.
set -euo pipefail

repo="${ACT_EVAL_REPO:-/workspace/shared/RoboSynChallenge-eval-all10}"
embodichain_root="${ACT_EMBODICHAIN_ROOT:-/workspace/shared/EmbodiChain-eval-all10}"
checkpoint="${ACT_CHECKPOINT:-/mnt/FermiBotNas/models/RoboSynChallenge/act_sample_loading_merged_h50_bs64_2ep/021162/pretrained_model}"
act_python="${ACT_PYTHON:-/workspace/shared/RoboSynChallenge-act-sample-loading/policy/act/.venv/bin/python}"
results_root="${ACT_RESULTS_ROOT:-/mnt/FermiBotNas/models/RoboSynChallenge/act_sample_loading_merged_h50_bs64_2ep/eval/checkpoint_021162_h50_random_seed0_4view}"
gpu_id="${ACT_GPU_ID:-0}"
smoke_root="$results_root/smoke_1ep_attempt1"
formal_root="$results_root/formal_20eps_attempt1"
launcher_log="$results_root/launcher.log"

mkdir -p "$results_root"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$launcher_log"
}

run_eval() {
    local episodes="$1" output_root="$2" output_log="$3"
    (
        cd "$repo"
        export EMBODICHAIN_ROOT="$embodichain_root"
        export EMBODICHAIN_SIM_EXIT_PROCESS=0
        export PYTHON_BIN="$repo/.venv/bin/python"
        export ACT_PYTHON="$act_python"
        export PYTHONPATH="$embodichain_root/embodichain_tasks:$embodichain_root:$repo/.venv/lib/python3.11/site-packages"
        bash policy/act/eval.sh sample_loading random "$checkpoint" "$gpu_id" \
            --pytorch_device cuda \
            --headless true \
            --renderer auto \
            --max_episodes "$episodes" \
            --seed 0 \
            --eval_video_log true \
            --eval_video_obs_keys '["cam_left_wrist","cam_right_wrist","cam_high","cam_third"]' \
            --eval_add_third_camera true \
            --eval_result_dir "$output_root" \
            --act_step 50
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
metrics = None
videos = []
tmp_videos = []
if len(metrics_paths) != 1:
    errors.append(f"expected exactly one metrics file, found {len(metrics_paths)}")
else:
    metrics = json.loads(metrics_paths[0].read_text())
    videos_dir = metrics_paths[0].parent / "videos"
    videos = sorted(videos_dir.glob("episode_*.mp4"))
    tmp_videos = sorted(videos_dir.glob("*.tmp.mp4"))
    summary = metrics.get("summary", {})
    if int(summary.get("episode_count", -1)) != expected:
        errors.append(
            f"episode_count={summary.get('episode_count')}, expected={expected}"
        )
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
        errors.append(
            f"{video.name}: size={probe.get('width')}x{probe.get('height')}"
        )

payload = {
    "schema_version": 1,
    "validated_at": datetime.now(timezone.utc).isoformat(),
    "checkpoint": "/mnt/FermiBotNas/models/RoboSynChallenge/act_sample_loading_merged_h50_bs64_2ep/021162/pretrained_model",
    "model_safetensors_sha256": "a3e20fd1bc9a600635fc757c6785b340b1a5bb9e14f29933b206b4bd928c33ec",
    "protocol": {
        "task": "sample_loading",
        "setting": "random",
        "episodes": expected,
        "seed": 0,
        "act_step": 50,
        "training_chunk_size": 50,
        "training_n_action_steps": 50,
        "policy_runtime": "LeRobot 0.3.3 isolated worker",
        "policy_camera_keys": [
            "cam_high", "cam_left_wrist", "cam_right_wrist"
        ],
        "recording_camera_keys": [
            "cam_left_wrist", "cam_right_wrist", "cam_high", "cam_third"
        ],
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

if [[ -f "$smoke_root/.complete" ]]; then
    log "smoke already complete; reusing $smoke_root"
else
    log "starting 1-episode smoke"
    mkdir -p "$smoke_root"
    run_eval 1 "$smoke_root" "$smoke_root/eval.log"
    verify_run "$smoke_root" 1 "$smoke_root/verification.json"
    touch "$smoke_root/.complete"
    log "smoke passed"
fi

log "starting formal 20-episode evaluation"
mkdir -p "$formal_root"
run_eval 20 "$formal_root" "$formal_root/eval.log"
verify_run "$formal_root" 20 "$formal_root/verification.json"
touch "$formal_root/.complete"
log "formal evaluation passed"
