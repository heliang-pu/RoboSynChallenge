#!/usr/bin/env bash
# Formally evaluate every saved ACT sample_loading checkpoint at true receding
# horizons H=10 and H=50, recording four-view videos for every episode.
set -euo pipefail

repo="${ACT_EVAL_REPO:-/workspace/shared/RoboSynChallenge-eval-all10}"
embodichain_root="${ACT_EMBODICHAIN_ROOT:-/workspace/shared/EmbodiChain-eval-all10}"
models_root="${ACT_MODELS_ROOT:-/mnt/FermiBotNas/models/RoboSynChallenge/act_sample_loading_merged_h50_bs64_2ep}"
act_python="${ACT_PYTHON:-/workspace/shared/RoboSynChallenge-act-sample-loading/policy/act/.venv/bin/python}"
results_root="${ACT_SWEEP_RESULTS_ROOT:-$models_root/eval/checkpoint_sweep_all_h10_h50_seed0_20eps_4view}"
gpu_id="${ACT_GPU_ID:-0}"
evaluation_episodes="${ACT_SWEEP_EPISODES:-20}"
launcher_log="$results_root/launcher.log"
progress_file="$results_root/progress.tsv"
sha_file="$results_root/checkpoint_sha256.tsv"

steps=(021162 020000 018000 016000 014000 012000 010000 008000 006000 004000 002000)
# User priority: finish every H=50 run before returning to the remaining H=10 runs.
horizons=(50 10)

mkdir -p "$results_root"
touch "$progress_file"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$launcher_log"
}

checkpoint_path() {
    printf '%s/%s/pretrained_model\n' "$models_root" "$1"
}

preflight_checkpoints() {
    : >"$sha_file"
    local step checkpoint
    for step in "${steps[@]}"; do
        checkpoint="$(checkpoint_path "$step")"
        [[ -s "$checkpoint/model.safetensors" ]]
        [[ -s "$checkpoint/config.json" ]]
        [[ -f "$models_root/${step}.NAS_VERIFIED" ]]
        printf '%s\t%s\n' "$step" "$(sha256sum "$checkpoint/model.safetensors" | awk '{print $1}')" \
            >>"$sha_file"
    done
    log "verified ${#steps[@]} checkpoints and recorded SHA256 manifest"
}

run_eval() {
    local step="$1" horizon="$2" episodes="$3" output_root="$4" output_log="$5"
    local checkpoint
    checkpoint="$(checkpoint_path "$step")"
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
            --act_step "$horizon"
    ) >"$output_log" 2>&1
}

verify_run() {
    local output_root="$1" expected="$2" step="$3" horizon="$4" verification="$5"
    local checkpoint_sha
    checkpoint_sha="$(awk -v s="$step" '$1 == s {print $2}' "$sha_file")"
    "$repo/.venv/bin/python" - \
        "$output_root" "$expected" "$step" "$horizon" "$checkpoint_sha" "$verification" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
step = sys.argv[3]
horizon = int(sys.argv[4])
checkpoint_sha = sys.argv[5]
out = Path(sys.argv[6])
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
    "checkpoint_step": int(step),
    "checkpoint": f"/mnt/FermiBotNas/models/RoboSynChallenge/act_sample_loading_merged_h50_bs64_2ep/{step}/pretrained_model",
    "model_safetensors_sha256": checkpoint_sha,
    "protocol": {
        "task": "sample_loading",
        "setting": "random",
        "episodes": expected,
        "seed": 0,
        "execution_horizon": horizon,
        "replan_after_actions": horizon,
        "training_chunk_size": 50,
        "policy_runtime": "LeRobot 0.3.3 isolated worker",
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
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
PY
}

build_leaderboard() {
    "$repo/.venv/bin/python" - "$results_root" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "runs").glob("step_*_h*/verification.json")):
    data = json.loads(path.read_text())
    if not data.get("passed"):
        continue
    summary = data["summary"]
    path_parts = path.parent.name.removeprefix("step_").split("_h", 1)
    checkpoint_step = int(data.get("checkpoint_step", path_parts[0]))
    protocol = data["protocol"]
    horizon = int(protocol.get("execution_horizon", protocol.get("act_step", path_parts[1])))
    rows.append(
        {
            "checkpoint_step": checkpoint_step,
            "horizon": horizon,
            "episodes": int(summary["episode_count"]),
            "success_count": int(summary["success_count"]),
            "success_rate": float(summary["success_rate"]),
            "average_action_steps": float(summary["average_action_steps"]),
            "average_inference_time_seconds": summary.get("average_inference_time_seconds"),
            "verification_path": str(path),
        }
    )
rows.sort(
    key=lambda row: (
        -row["success_rate"],
        row["average_action_steps"],
        row["checkpoint_step"] * -1,
        row["horizon"],
    )
)
payload = {
    "schema_version": 1,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "ranking_rule": [
        "higher_success_rate",
        "lower_average_action_steps",
        "later_checkpoint_step",
        "lower_execution_horizon",
    ],
    "evaluated_combinations": len(rows),
    "leaderboard": rows,
    "best": rows[0] if rows else None,
}
(root / "leaderboard.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
)
with (root / "leaderboard.csv").open("w", newline="") as handle:
    fieldnames = list(rows[0]) if rows else [
        "checkpoint_step", "horizon", "episodes", "success_count", "success_rate",
        "average_action_steps", "average_inference_time_seconds", "verification_path",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

preflight_checkpoints
log "true-H10 smoke was already validated in the preserved 5-episode sweep"

for horizon in "${horizons[@]}"; do
    for step in "${steps[@]}"; do
        output_root="$results_root/runs/step_${step}_h${horizon}"
        if [[ -f "$output_root/.complete" ]]; then
            log "evaluation already complete step=$step H=$horizon"
            continue
        fi
        if [[ -d "$output_root" ]] && find "$output_root" -mindepth 1 -print -quit | grep -q .; then
            incomplete_root="${output_root}.incomplete_$(date +%Y%m%dT%H%M%S)"
            mv "$output_root" "$incomplete_root"
            log "preserved incomplete run at $incomplete_root"
        fi
        mkdir -p "$output_root"
        printf '%s\tstarted\t%s\t%s\n' "$(date -Is)" "$step" "$horizon" >>"$progress_file"
        log "starting formal evaluation step=$step H=$horizon episodes=$evaluation_episodes"
        run_eval "$step" "$horizon" "$evaluation_episodes" "$output_root" "$output_root/eval.log"
        verify_run "$output_root" "$evaluation_episodes" "$step" "$horizon" "$output_root/verification.json"
        touch "$output_root/.complete"
        printf '%s\tcompleted\t%s\t%s\n' "$(date -Is)" "$step" "$horizon" >>"$progress_file"
        build_leaderboard
        log "formal evaluation passed step=$step H=$horizon"
    done
done

build_leaderboard
log "all 22 formal evaluations completed"
touch "$results_root/.complete"
