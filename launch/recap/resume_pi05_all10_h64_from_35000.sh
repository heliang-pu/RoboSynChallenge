#!/usr/bin/env bash
set -euo pipefail

repo=/root/code/RoboSynChallenge
pi05="$repo/policy/pi05"
config=pi05_all10_h64_expert
exp=all10_expert_base_h64_bs64_steps100k
log=/tmp/pi05/logs/all10_expert_base_h64_bs64_steps100k.log

export PI05_ALL10_BASE_WEIGHTS=/tmp/pi05/base_weights/Hoshipu_pi05-robotwin2-random-60k/params
export PI05_ALL10_BATCH_SIZE=56
export PI05_ALL10_NUM_WORKERS=64
export PI05_ALL10_FSDP_DEVICES=1
export PI05_ALL10_NUM_TRAIN_STEPS=100000
export PI05_ALL10_SAVE_INTERVAL=2500
export PI05_ALL10_CHECKPOINT_BASE_DIR=/tmp/pi05/checkpoints
export HF_LEROBOT_HOME="$pi05/training_data"
export LD_PRELOAD=/opt/mpi/lib/libmpi.so
export PYTHONPATH="$pi05/src:$pi05/packages/openpi-client/src"
export OMP_NUM_THREADS=16
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online
export WANDB_RUN_ID=y2b0fjr3
export WANDB_RESUME=allow

checkpoint=/tmp/pi05/checkpoints/$config/$exp/35000
test -f "$checkpoint/_CHECKPOINT_METADATA"
test -f "$checkpoint/train_state/manifest.ocdbt"
test -f "$checkpoint/params/manifest.ocdbt"

cd "$pi05"
python3 - <<'PY'
import jax
d=jax.devices()
assert len(d)==8 and d[0].platform!="cpu", d
print(f"[OK] DCU x{len(d)} kind={d[0].device_kind}")
PY

printf '\n[%s] RESUME step=35000 bs=56 after HIP graph OOM at ~35700\n' "$(date -Is)" | tee -a "$log"
exec python3 scripts/train.py "$config" --exp-name="$exp" --resume >>"$log" 2>&1
