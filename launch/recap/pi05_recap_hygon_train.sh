#!/usr/bin/env bash
set -euo pipefail

REPO=/root/code/RoboSynChallenge
PI05=$REPO/policy/pi05
DATASET=simrecap_sample_loading_round1_vlm3500_baked_prompt
EXP=sample_loading_round1_vlm3500_baked_base_acp30
LOG_DIR=/tmp/pi05/logs

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/$EXP.log") 2>&1

export SIMRECAP_REPO_ID="RoboSynChallenge/$DATASET"
export SIMRECAP_INDICATOR_KEY=complementary_info.acp_indicator_round1
export SIMRECAP_ACP_ENABLE=1
export SIMRECAP_ACP_DROPOUT=0.3
export SIMRECAP_SAVE_INTERVAL=2500
export SIMRECAP_BATCH_SIZE=64
# Pure data parallelism (DDP-like): each HCU owns a full training replica.
export SIMRECAP_FSDP_DEVICES=1
export SIMRECAP_NUM_TRAIN_STEPS=20000
export SIMRECAP_CHECKPOINT_BASE_DIR=/tmp/pi05/checkpoints
export SIMRECAP_WEIGHTS=/root/code/models/pi05_base/params
export HF_LEROBOT_HOME=$PI05/training_data
export LD_PRELOAD=/opt/mpi/lib/libmpi.so
export PYTHONPATH="$PI05/src:$PI05/packages/openpi-client/src"
export OMP_NUM_THREADS=16

cd "$PI05"
test -f "$HF_LEROBOT_HOME/RoboSynChallenge/$DATASET/ACP_PROMPT_BAKED.json"
test -d "$SIMRECAP_WEIGHTS"

NORM_STATS="$PI05/assets/pi05_sim_recap/$SIMRECAP_REPO_ID/norm_stats.json"
if [[ ! -f "$NORM_STATS" ]]; then
    echo "[$(date -Is)] computing norm stats"
    python3 scripts/compute_norm_stats_no_video.py
fi

echo "[$(date -Is)] starting pi0.5 RECAP training (FSDP=$SIMRECAP_FSDP_DEVICES)"
exec bash finetune_dcu.sh pi05_sim_recap "$EXP" 0,1,2,3,4,5,6,7
