#!/usr/bin/env bash
set -uo pipefail

repo=${ROBOSYN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
pi05="$repo/policy/pi05"
work="$repo/.eval_all10_h64_ckpt32500_4view"
results="$work/results"
desktop=${ROBOSYN_EVAL_DELIVERY_DIR:-$HOME/桌面/pi05_all10_h64_ckpt32500_四视角评估}
config=pi05_all10_h64_expert
model=all10_expert_base_h64_bs64_steps100k
checkpoint=32500
tasks=(click_bell drawer_open_place handle_basket item_assembly items_handover manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring)
paused_groups=(2063614 2063644)

resume_synthesis() {
    for group in "${paused_groups[@]}"; do kill -CONT -- "-$group" 2>/dev/null || true; done
}
trap resume_synthesis EXIT INT TERM

while tmux has-session -t pi05-all10-eval-32500 2>/dev/null; do sleep 20; done
for group in "${paused_groups[@]}"; do kill -STOP -- "-$group" 2>/dev/null || true; done

for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    count=$(find "$results/$task" -type f -name '*.mp4' 2>/dev/null | wc -l)
    (( count == 10 )) && continue
    if [[ -e "$results/$task" ]]; then
        mkdir -p "$work/failed_attempts"
        mv "$results/$task" "$work/failed_attempts/${task}_$(date +%s)"
    fi
    for attempt in 1 2 3; do
        log="$work/logs/${task}_retry${attempt}.log"
        echo "[$(date -Is)] RETRY task=$task attempt=$attempt" | tee -a "$work/SUMMARY.txt"
        (
            cd "$pi05" || exit 1
            export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 EMBODICHAIN_SIM_EXIT_PROCESS=0 MALLOC_ARENA_MAX=2
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
            bash eval.sh "$task" random "$config" "$model" 0 \
                --checkpoint_id "$checkpoint" --pi0_step 64 --max_episodes 10 \
                --seed "$((832500 + i * 1000))" --headless true --pytorch_device cuda \
                --eval_video_log true \
                --eval_video_obs_keys "['cam_left_wrist','cam_right_wrist','cam_high','cam_third']" \
                --eval_result_dir "$results"
        ) >"$log" 2>&1
        rc=$?
        count=$(find "$results/$task" -type f -name '*.mp4' 2>/dev/null | wc -l)
        echo "[$(date -Is)] RETRY_END task=$task attempt=$attempt rc=$rc videos=$count" | tee -a "$work/SUMMARY.txt"
        (( rc == 0 && count == 10 )) && break
        [[ -e "$results/$task" ]] && {
            mkdir -p "$work/failed_attempts"
            mv "$results/$task" "$work/failed_attempts/${task}_$(date +%s)_attempt${attempt}"
        }
        sleep 15
    done
done

python3 - "$results" <<'PY' >> "$work/SUMMARY.txt"
import json, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1])
tasks="click_bell drawer_open_place handle_basket item_assembly items_handover manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring".split()
total=0
for task in tasks:
    metrics=sorted(root.glob(f"{task}/**/evaluation_metrics.json"))
    assert len(metrics)==1,(task,metrics)
    run=metrics[0].parent
    videos=sorted((run/"videos").glob("*.mp4"))
    assert len(videos)==10,(task,len(videos))
    data=json.load(open(metrics[0])); assert data["summary"]["episode_count"]==10
    for video in videos:
        dims=subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height","-of","csv=p=0:s=x",str(video)],text=True).strip()
        assert dims=="2560x480",(video,dims)
    total+=10
    print(task,"videos=10 metrics=ok")
assert total==100
print("TOTAL_VALIDATED_VIDEOS=100")
PY

mkdir -p "$desktop"
rsync -a "$results/" "$desktop/"
cp "$work/SUMMARY.txt" "$desktop/SUMMARY.txt"
touch "$work/.complete"
echo "[$(date -Is)] REPAIRED_AND_DELIVERED desktop=$desktop" | tee -a "$work/SUMMARY.txt"
