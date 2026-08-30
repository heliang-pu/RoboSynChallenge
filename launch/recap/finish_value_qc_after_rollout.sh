#!/usr/bin/env bash
set -euo pipefail

repo=/home/fmc3/workspace/RoboSynChallenge
work="$repo/lerobot_dataset/.simrecap_work/value_qc_freeze_vlm_retry_single"
train="$repo/lerobot_dataset/.simrecap_work/sample_loading_round1_all/merged_v30"
run=value_sample_loading_round1_all_freeze_vlm_bs96_steps8000
ckpt=/home/fmc3/FermiBotNas/models/RoboSynChallenge/$run
baseline=/home/fmc3/FermiBotNas/models/RoboSynChallenge/value_sample_loading_round1_all_bs96_steps8000/003500
python=/home/fmc3/miniconda3/envs/evo-rl/bin/python
task='Pick up the test tube, and it to the other arm, and insert it to the rack.'
insample_eps='[0,1,2,3,4,5,6,7,8,9,10,11,12,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,869,871,877,898,919,932,949,950]'
steps=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000)

mkdir -p "$work/logs"
exec >>"$work/logs/finalize.log" 2>&1
echo "[$(date -Is)] waiting for held-out rollout"
while tmux has-session -t value-holdout-policy-single 2>/dev/null; do sleep 30; done
grep -q 'EVAL_EXIT_CODE=0' "$work/logs/policy_collect.log"

info=$(find "$work/policy_rollout_40" -maxdepth 3 -path '*/meta/info.json' -print | head -1)
fresh=$(dirname "$(dirname "$info")")
[[ -n "$fresh" && -f "$fresh/episode_success.json" ]]
episodes="$($python -c "import json; n=json.load(open('$fresh/meta/info.json'))['total_episodes']; assert n==40,n; print(json.dumps(list(range(n))))")"

echo "[$(date -Is)] verifying action-hash independence"
"$python" "$repo/scripts/verify_value_holdout.py" \
    --train "$train" --fresh "$fresh" --output "$work/holdout_provenance.json"

echo "[$(date -Is)] evaluating held-out across 16 checkpoints"
PYTHONPATH="$repo/third_party/evo_rl/src" "$python" \
    "$repo/scripts/evaluate_value_checkpoints_sparse.py" \
    --dataset "heldout_fresh=$fresh" --episodes "heldout_fresh=$episodes" \
    --group heldout_fresh=held_out --task "heldout_fresh=$task" \
    --checkpoint-root "$ckpt" --steps "${steps[@]}" \
    --phases 7 --task-max-length 600 --batch-size 8 --num-workers 2 \
    --output "$work/heldout_metrics.json"

echo "[$(date -Is)] combining in-sample and held-out metrics"
"$python" - "$work/insample_metrics.json" "$work/heldout_metrics.json" "$work/combined_metrics.json" <<'PY'
import json, sys
inside_path, held_path, out_path = sys.argv[1:]
inside = json.load(open(inside_path))
held = json.load(open(held_path))
inside_by_step = {item["step"]: item for item in inside["checkpoints"]}
for item in held["checkpoints"]:
    item["groups"]["in_sample"] = inside_by_step[item["step"]]["groups"]["in_sample"]
held["config"]["in_sample_source"] = inside_path
json.dump(held, open(out_path, "w"), indent=2, allow_nan=True)
PY

echo "[$(date -Is)] evaluating old unfrozen 3500 baseline"
PYTHONPATH="$repo/third_party/evo_rl/src" "$python" \
    "$repo/scripts/evaluate_value_checkpoints_sparse.py" \
    --dataset "insample=$train" --dataset "heldout_fresh=$fresh" \
    --episodes "insample=$insample_eps" --episodes "heldout_fresh=$episodes" \
    --group insample=in_sample --group heldout_fresh=held_out \
    --task "heldout_fresh=$task" \
    --checkpoint-root "$(dirname "$baseline")" --steps 3500 \
    --phases 7 --task-max-length 600 --batch-size 8 --num-workers 2 \
    --output "$work/baseline_metrics.json"
"$python" - "$work/baseline_metrics.json" "$baseline" <<'PY'
import json, sys
p, checkpoint = sys.argv[1:]
x = json.load(open(p)); x["config"]["checkpoint_path"] = checkpoint
json.dump(x, open(p, "w"), indent=2, allow_nan=True)
PY

echo "[$(date -Is)] generating Chinese quality report"
"$python" "$repo/scripts/generate_value_quality_report.py" \
    --metrics "$work/combined_metrics.json" \
    --baseline-metrics "$work/baseline_metrics.json" \
    --provenance "$work/holdout_provenance.json" \
    --train-log "$repo/outputs/value_train/$run.launch/train.log" \
    --checkpoint-root "$ckpt" --output "$work/QUALITY_REPORT.md"
touch "$work/.complete"
echo "[$(date -Is)] value QC complete"
