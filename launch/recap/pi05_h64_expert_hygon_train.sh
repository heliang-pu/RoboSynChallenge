#!/usr/bin/env bash
set -euo pipefail

repo=/root/code/RoboSynChallenge
pi05="$repo/policy/pi05"
config=pi05_sample_loading_h64_expert
exp=sample_loading_expert_base_h64_24h_v2
dataset=RoboSynChallenge/cobotmagic_Sim_sample_loading_expert_v21
log_dir=/tmp/pi05/logs
log="$log_dir/$exp.log"

mkdir -p "$log_dir"
exec > >(tee -a "$log") 2>&1

export PI05_H64_BASE_WEIGHTS=/root/code/models/pi05_base/params
export PI05_H64_BATCH_SIZE=64
export PI05_H64_NUM_WORKERS=64
export PI05_H64_FSDP_DEVICES=1
export PI05_H64_NUM_TRAIN_STEPS=50000
export PI05_H64_SAVE_INTERVAL=2500
export PI05_H64_CHECKPOINT_BASE_DIR=/tmp/pi05/checkpoints
export HF_LEROBOT_HOME="$pi05/training_data"
export LD_PRELOAD=/opt/mpi/lib/libmpi.so
export PYTHONPATH="$pi05/src:$pi05/packages/openpi-client/src"
export OMP_NUM_THREADS=16
export WANDB_MODE=online

cd "$pi05"
test -f "$HF_LEROBOT_HOME/$dataset/meta/info.json"
test -d "$PI05_H64_BASE_WEIGHTS"

norm_stats="$pi05/assets/$config/$dataset/norm_stats.json"
if [[ ! -f "$norm_stats" ]]; then
    echo "[$(date -Is)] computing H64 expert norm stats"
    python3 scripts/compute_norm_stats_no_video.py --config-name "$config" --num-workers 16
fi

echo "[$(date -Is)] starting plain expert pi0.5 SFT; action_horizon=64; steps=50000"
exec bash finetune_dcu.sh "$config" "$exp" 0,1,2,3,4,5,6,7
