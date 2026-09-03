#!/usr/bin/env bash
set -uo pipefail

repo=${ROBOSYN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
pi05="$repo/policy/pi05"
checkpoint=32500
config=pi05_all10_h64_expert
model=all10_expert_base_h64_bs64_steps100k
work="$repo/.eval_all10_h64_ckpt32500_4view"
logs="$work/logs"
desktop=${ROBOSYN_EVAL_DELIVERY_DIR:-$HOME/桌面/pi05_all10_h64_ckpt32500_四视角评估}
summary="$work/SUMMARY.txt"
tasks=(click_bell drawer_open_place handle_basket item_assembly items_handover manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring)
paused_groups=(2063614 2063644)

mkdir -p "$logs"

resume_synthesis() {
    for group in "${paused_groups[@]}"; do
        kill -CONT -- "-$group" 2>/dev/null || true
    done
    printf '[%s] resumed fmc3-0 synthesis process groups\n' "$(date -Is)" >> "$summary"
}
trap resume_synthesis EXIT INT TERM

printf '[%s] checkpoint=%s tasks=%s episodes_per_task=10 views=left,right,high,third\n' \
    "$(date -Is)" "$checkpoint" "${#tasks[@]}" > "$summary"

failures=0
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    seed=$((832500 + i * 1000))
    log="$logs/$task.log"
    printf '[%s] START task=%s seed=%s\n' "$(date -Is)" "$task" "$seed" | tee -a "$summary"
    (
        cd "$pi05" || exit 1
        export PYTHONUNBUFFERED=1
        export PYTHONFAULTHANDLER=1
        export EMBODICHAIN_SIM_EXIT_PROCESS=0
        export MALLOC_ARENA_MAX=2
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
        bash eval.sh "$task" random "$config" "$model" 0 \
            --checkpoint_id "$checkpoint" --pi0_step 64 \
            --max_episodes 10 --seed "$seed" --headless true \
            --pytorch_device cuda --eval_video_log true \
            --eval_video_obs_keys "['cam_left_wrist','cam_right_wrist','cam_high','cam_third']" \
            --eval_result_dir "$work/results"
    ) >"$log" 2>&1
    rc=$?
    result=$(grep -a 'Evaluation Results Summary:' "$log" | tail -1 | sed 's/^[[:space:]]*//')
    printf '[%s] END task=%s rc=%s %s\n' "$(date -Is)" "$task" "$rc" "$result" | tee -a "$summary"
    (( rc == 0 )) || failures=$((failures + 1))
done

python3 - "$work/results" <<'PY' >> "$summary"
import json, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = [
    "click_bell", "drawer_open_place", "handle_basket", "item_assembly",
    "items_handover", "manipulate_pipette", "mixer_operating",
    "sample_loading", "table_rearrangement", "water_pouring",
]
total_videos = 0
for task in tasks:
    metrics = sorted(root.glob(f"{task}/**/evaluation_metrics.json"))
    assert len(metrics) == 1, (task, metrics)
    run = metrics[0].parent
    videos = sorted((run / "videos").glob("*.mp4"))
    assert len(videos) == 10, (task, len(videos))
    data = json.load(open(metrics[0]))
    completed = data.get("summary", {}).get("episode_count", len(videos))
    assert completed == 10, (task, completed)
    for video in videos:
        dims = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video),
        ], text=True).strip()
        assert dims == "2560x480", (video, dims)
    total_videos += len(videos)
    print(task, "videos=10", "metrics=ok")
assert total_videos == 100, total_videos
print("TOTAL_VALIDATED_VIDEOS=100")
PY
verify_rc=$?
if (( failures == 0 && verify_rc == 0 )); then
    mkdir -p "$desktop"
    rsync -a "$work/results/" "$desktop/"
    cp "$summary" "$desktop/SUMMARY.txt"
    touch "$work/.complete"
    printf '[%s] DELIVERED desktop=%s\n' "$(date -Is)" "$desktop" | tee -a "$summary"
else
    printf '[%s] INCOMPLETE eval_failures=%s verify_rc=%s\n' "$(date -Is)" "$failures" "$verify_rc" | tee -a "$summary"
    exit 1
fi
