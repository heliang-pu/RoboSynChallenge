#!/usr/bin/env bash
set -euo pipefail

# 训练机地址与密钥不入库，运行前用环境变量提供：
#   RECAP_SYNC_REMOTE=user@host RECAP_SYNC_PORT=22 RECAP_SYNC_KEY=~/.ssh/id_ed25519
remote=${RECAP_SYNC_REMOTE:?set RECAP_SYNC_REMOTE=user@host}
port=${RECAP_SYNC_PORT:-22}
key=${RECAP_SYNC_KEY:-$HOME/.ssh/id_ed25519}
source_dir=/tmp/pi05/checkpoints/pi05_sim_recap/sample_loading_round1_vlm3500_baked_base_acp30/29999
nas_root=${RECAP_NAS_ROOT:-$HOME/FermiBotNas/models/RoboSynChallenge/pi05_sim_recap/sample_loading_round1_vlm3500_baked_base_acp30}
final_dir="$nas_root/29999"
partial_dir="$nas_root/.29999.partial"
log="$nas_root/sync.log"
ssh_opts=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "$key")
rsync_rsh="ssh -p $port -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $key"

mkdir -p "$nas_root" "$partial_dir"
exec >>"$log" 2>&1
echo "[$(date -Is)] syncing final checkpoint 29999"
ssh "${ssh_opts[@]}" "$remote" "test -f '$source_dir/_CHECKPOINT_METADATA'"
rsync -a --partial --delete -e "$rsync_rsh" "$remote:$source_dir/" "$partial_dir/"
read -r remote_files remote_bytes < <(
  ssh "${ssh_opts[@]}" "$remote" \
    "find '$source_dir' -type f -printf '%s\\n' | awk '{n++; s+=\$1} END {printf \"%d %.0f\\n\", n, s}'"
)
read -r local_files local_bytes < <(
  find "$partial_dir" -type f -printf '%s\n' | awk '{n++; s+=$1} END {printf "%d %.0f\n", n, s}'
)
[[ "$remote_files/$remote_bytes" == "$local_files/$local_bytes" ]]
[[ ! -e "$final_dir" ]]
mv "$partial_dir" "$final_dir"
echo "[$(date -Is)] checkpoint 29999 complete files=$local_files bytes=$local_bytes"
ssh "${ssh_opts[@]}" "$remote" "rm -rf -- '$source_dir'"
echo "[$(date -Is)] removed remote checkpoint 29999 after verified NAS sync"
