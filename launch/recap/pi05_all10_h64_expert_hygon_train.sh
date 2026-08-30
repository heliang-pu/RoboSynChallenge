#!/usr/bin/env bash
set -euo pipefail

repo=/root/code/RoboSynChallenge
pi05="$repo/policy/pi05"
config=pi05_all10_h64_expert
exp=all10_expert_base_h64_bs64_steps100k
dataset_root="$pi05/training_data/RoboSynChallenge/all10_expert_v21"
log_dir=/tmp/pi05/logs
log="$log_dir/$exp.log"
tasks=(click_bell drawer_open_place handle_basket item_assembly items_handover manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring)

mkdir -p "$log_dir"
exec > >(tee -a "$log") 2>&1

export PI05_ALL10_BASE_WEIGHTS=/tmp/pi05/base_weights/Hoshipu_pi05-robotwin2-random-60k/params
export PI05_ALL10_BATCH_SIZE=64
export PI05_ALL10_NUM_WORKERS=64
export PI05_ALL10_FSDP_DEVICES=1
export PI05_ALL10_NUM_TRAIN_STEPS=100000
export PI05_ALL10_SAVE_INTERVAL=2500
export PI05_ALL10_CHECKPOINT_BASE_DIR=/tmp/pi05/checkpoints
export HF_LEROBOT_HOME="$pi05/training_data"
export LD_PRELOAD=/opt/mpi/lib/libmpi.so
export PYTHONPATH="$pi05/src:$pi05/packages/openpi-client/src"
export OMP_NUM_THREADS=16
export WANDB_MODE=online

cd "$pi05"
test -d "$PI05_ALL10_BASE_WEIGHTS"
test -f "$(dirname "$PI05_ALL10_BASE_WEIGHTS")/.download_complete"
for task in "${tasks[@]}"; do
    test -f "$dataset_root/cobotmagic_Sim_$task/meta/info.json"
done

norm_stats="$pi05/assets/$config/RoboSynChallenge/all10_expert_h64/norm_stats.json"
if [[ ! -f "$norm_stats" ]]; then
    echo "[$(date -Is)] computing all-10 H64 expert norm stats"
    python3 scripts/compute_norm_stats_no_video.py --config-name "$config" --num-workers 16
fi

echo "[$(date -Is)] starting 10-task plain expert pi0.5 SFT; action_horizon=64; steps=100000"
exec bash finetune_dcu.sh "$config" "$exp" 0,1,2,3,4,5,6,7
