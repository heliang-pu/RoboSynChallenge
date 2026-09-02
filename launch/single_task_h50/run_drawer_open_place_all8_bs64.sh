#!/usr/bin/env bash
# Full-parameter drawer_open_place fine-tune on all eight BW1000_H devices.
set -euo pipefail

repo=/root/code/RoboSynChallenge
pi05="$repo/policy/pi05"
config=pi05_drawer_open_place
exp=drawer_open_place_from_all10_67500_h50_bs64_2ep
data_home=/tmp/pi05/training_data
repo_id=RoboSynChallenge/official_plus_seeded_clean_v21_drawer_open_place
base=/tmp/pi05/base_weights/all10_h64_67500
checkpoint_root=/tmp/pi05/drawer_open_place_h50_all8_checkpoints
log=/tmp/pi05/logs/drawer_open_place_from_all10_67500_h50_bs64_2ep.log
steps=26719

mkdir -p "$(dirname "$log")" "$checkpoint_root"
test -f "$data_home/$repo_id/.complete.json"
test -f "$base/params/manifest.ocdbt"
test -f "$base/assets/RoboSynChallenge/all10_expert_h64/norm_stats.json"

unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export HF_LEROBOT_HOME="$data_home"
export LD_PRELOAD=/opt/mpi/lib/libmpi.so
export PYTHONPATH="$pi05/src:$pi05/packages/openpi-client/src"
export OMP_NUM_THREADS=16
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export WANDB_MODE=online
export WANDB_RUN_GROUP=drawer_open_place_single_task

cd "$pi05"
python3 - <<'PY'
import jax
devices = jax.devices()
assert len(devices) == 8 and devices[0].platform != "cpu", devices
print(f"[OK] all8 devices: {devices[0].device_kind}")
PY

exec python3 scripts/train.py "$config" \
    --exp-name="$exp" \
    --model.action-horizon=50 \
    --weight-loader.params-path="$base/params" \
    --data.repo-id="$repo_id" \
    --data.assets.assets-dir="$base/assets" \
    --data.assets.asset-id=RoboSynChallenge/all10_expert_h64 \
    --checkpoint-base-dir="$checkpoint_root" \
    --num-train-steps="$steps" \
    --batch-size=64 \
    --num-workers=64 \
    --fsdp-devices=1 \
    --ema-decay=None \
    --save-interval=2500 \
    --keep-period=2500 \
    --lr-schedule.warmup-steps=500 \
    --lr-schedule.peak-lr=1e-5 \
    --lr-schedule.decay-steps="$steps" \
    --lr-schedule.decay-lr=1e-6 \
    >>"$log" 2>&1
