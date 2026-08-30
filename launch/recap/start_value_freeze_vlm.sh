#!/usr/bin/env bash
set -euo pipefail

repo=/home/fmc3/workspace/RoboSynChallenge
work="$repo/lerobot_dataset/.simrecap_work/sample_loading_round1_all"
dataset="$work/merged_v30"
experiment=value_sample_loading_round1_all_freeze_vlm_bs96_steps8000
output="$repo/outputs/value_train/$experiment"
launch_dir="${output}.launch"
vision_model=/home/fmc3/workspace/models/google/siglip-so400m-patch14-384
language_model=/home/fmc3/workspace/models/google/gemma-3-270m
python=/home/fmc3/miniconda3/envs/evo-rl/bin/python

[[ -f "$dataset/meta/info.json" ]] || { echo "dataset missing: $dataset" >&2; exit 1; }
[[ -d "$vision_model" ]] || { echo "vision model missing: $vision_model" >&2; exit 1; }
[[ -d "$language_model" ]] || { echo "language model missing: $language_model" >&2; exit 1; }
[[ -x "$python" ]] || { echo "python missing: $python" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "output already exists: $output" >&2; exit 1; }

mkdir -p "$launch_dir"
cd "$repo"
export HF_LEROBOT_HOME="$work"
export PYTHONPATH="$repo/third_party/evo_rl/src"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

nohup "$python" -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local/merged_v30 \
  --dataset.root="$dataset" \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --batch_size=96 \
  --num_workers=2 \
  --steps=8000 \
  --save_freq=500 \
  --log_freq=20 \
  --value.use_gradient_checkpointing=false \
  --value.freeze_vision_encoder=true \
  --value.freeze_language_model=true \
  --value.vision_repo_id="$vision_model" \
  --value.language_repo_id="$language_model" \
  --output_dir="$output" \
  --job_name="$experiment" \
  --wandb.enable=true \
  >"$launch_dir/train.log" 2>&1 </dev/null &
train_pid=$!
echo "$train_pid" >"$launch_dir/train.pid"

nas_root=/home/fmc3/FermiBotNas/models/RoboSynChallenge/$experiment
nohup "$repo/launch/recap/sync_value_checkpoints_to_nas.sh" \
  "$train_pid" "$output/checkpoints" "$nas_root" \
  >"$launch_dir/checkpoint_sync.log" 2>&1 </dev/null &
sync_pid=$!
echo "$sync_pid" >"$launch_dir/checkpoint_sync.pid"

echo "training_pid=$train_pid"
echo "sync_pid=$sync_pid"
echo "log=$launch_dir/train.log"
echo "nas=$nas_root"
