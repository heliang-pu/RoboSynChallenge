#!/usr/bin/env bash
# Wait for the frozen-VLM pistar06 run, collect a fresh held-out rollout, and
# evaluate representative checkpoints on in-sample and held-out trajectories.
set -euo pipefail

repo=/home/fmc3/workspace/RoboSynChallenge
train_pid=1971472
run=value_sample_loading_round1_all_freeze_vlm_bs96_steps8000
train_log="$repo/outputs/value_train/$run.launch/train.log"
nas_ckpt=/home/fmc3/FermiBotNas/models/RoboSynChallenge/$run
work="$repo/lerobot_dataset/.simrecap_work/value_qc_freeze_vlm_bs96_steps8000"
logdir="$work/logs"
python=/home/fmc3/miniconda3/envs/evo-rl/bin/python
baseline_ckpt=/home/fmc3/FermiBotNas/models/RoboSynChallenge/value_sample_loading_round1_all_bs96_steps8000/003500
steps=(000500 001000 001500 002000 002500 003000 003500 004000 004500 005000 005500 006000 006500 007000 007500 008000)
canonical_task='Pick up the test tube, and it to the other arm, and insert it to the rack.'

mkdir -p "$logdir"
exec > >(tee -a "$work/coordinator.log") 2>&1

echo "[$(date '+%F %T')] waiting for training pid $train_pid"
while kill -0 "$train_pid" 2>/dev/null; do
    tail -n 1 "$train_log" || true
    sleep 60
done

grep -q "End of value training" "$train_log" || {
    echo "training stopped without clean completion" >&2
    exit 1
}

echo "[$(date '+%F %T')] training complete; waiting for NAS checkpoint 008000"
for _ in $(seq 1 60); do
    if [[ -s "$nas_ckpt/008000/pretrained_model/model.safetensors" ]]; then
        break
    fi
    sleep 10
done
[[ -s "$nas_ckpt/008000/pretrained_model/model.safetensors" ]] || {
    echo "NAS checkpoint 008000 did not arrive" >&2
    exit 1
}
for step in "${steps[@]}"; do
    [[ -s "$nas_ckpt/$step/pretrained_model/model.safetensors" ]] || {
        echo "missing checkpoint $step on NAS" >&2
        exit 1
    }
done

# New seeds make these rollouts independent from the 956-episode training pool.
holdout_tag=reward_holdout_freezevlm_v2
holdout_work="$repo/lerobot_dataset/.simrecap_work/sample_loading_${holdout_tag}"
if [[ ! -f "$holdout_work/rollout_v30/rollout_merged/episode_success.json" ]]; then
    echo "[$(date '+%F %T')] collecting 80 fresh held-out rollouts"
    SEED_BASE=91029 bash "$repo/launch/recap/01_rollout.sh" \
        sample_loading "$holdout_tag" pi05_sample_loading sample_loading 28000 80 1 0 random_rollout
fi

# Read-only sources. Sparse evaluation does not write annotations.
insample="$repo/lerobot_dataset/.simrecap_work/sample_loading_round1_all/merged_v30"
insample_eps='[0,1,2,3,4,5,6,7,8,9,10,11,12,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,869,871,877,898,919,932,949,950]'

fresh_src="$holdout_work/rollout_v30/rollout_merged"
fresh_eps="$($python -c "import json; print(json.dumps(list(range(json.load(open('$fresh_src/meta/info.json'))['total_episodes']))))")"

echo "[$(date '+%F %T')] verify held-out trajectory and seed independence"
"$python" "$repo/scripts/verify_value_holdout.py" \
    --train "$insample" --fresh "$fresh_src" \
    --output "$work/holdout_provenance.json" \
    > "$logdir/verify_holdout.log" 2>&1

step_args=()
for step in "${steps[@]}"; do step_args+=("$((10#$step))"); done

echo "[$(date '+%F %T')] sparse checkpoint evaluation"
PYTHONPATH="$repo/third_party/evo_rl/src" "$python" \
    "$repo/scripts/evaluate_value_checkpoints_sparse.py" \
    --dataset "insample=$insample" \
    --dataset "heldout_fresh=$fresh_src" \
    --episodes "insample=$insample_eps" \
    --episodes "heldout_fresh=$fresh_eps" \
    --group insample=in_sample \
    --group heldout_fresh=held_out \
    --task "heldout_fresh=$canonical_task" \
    --checkpoint-root "$nas_ckpt" --steps "${step_args[@]}" \
    --phases 7 --task-max-length 600 --batch-size 8 --num-workers 4 \
    --output "$work/sparse_metrics.json" \
    > "$logdir/sparse_eval.log" 2>&1

echo "[$(date '+%F %T')] evaluate old unfrozen-VLM 3500 baseline"
PYTHONPATH="$repo/third_party/evo_rl/src" "$python" \
    "$repo/scripts/evaluate_value_checkpoints_sparse.py" \
    --dataset "insample=$insample" \
    --dataset "heldout_fresh=$fresh_src" \
    --episodes "insample=$insample_eps" \
    --episodes "heldout_fresh=$fresh_eps" \
    --group insample=in_sample \
    --group heldout_fresh=held_out \
    --task "heldout_fresh=$canonical_task" \
    --checkpoint-root "$(dirname "$baseline_ckpt")" --steps 3500 \
    --phases 7 --task-max-length 600 --batch-size 8 --num-workers 4 \
    --output "$work/baseline_metrics.json" \
    > "$logdir/baseline_eval.log" 2>&1
"$python" - "$work/baseline_metrics.json" "$baseline_ckpt" <<'PYEOF'
import json, sys
p, checkpoint = sys.argv[1:]
data = json.load(open(p))
data["config"]["checkpoint_path"] = checkpoint
json.dump(data, open(p, "w"), indent=2, allow_nan=True)
PYEOF

echo "[$(date '+%F %T')] generate quality report"
"$python" "$repo/scripts/generate_value_quality_report.py" \
    --metrics "$work/sparse_metrics.json" \
    --baseline-metrics "$work/baseline_metrics.json" \
    --provenance "$work/holdout_provenance.json" \
    --train-log "$train_log" --checkpoint-root "$nas_ckpt" \
    --output "$work/QUALITY_REPORT.md" \
    > "$logdir/generate_report.log" 2>&1

echo "[$(date '+%F %T')] evaluation complete"
touch "$work/.complete"
