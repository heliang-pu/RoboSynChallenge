#!/usr/bin/env bash
set -euo pipefail

# 训练机地址不入库：运行前 export RECAP_REMOTE=user@host RECAP_PORT=<port> RECAP_KEY=~/.ssh/id_ed25519

source_dir=$HOME/FermiBotNas/dataset/RoboSynChallenge/Sim_clean_filtered_pruned/cobotmagic_Sim_sample_loading
remote=${RECAP_REMOTE:?set RECAP_REMOTE=user@host}
port=${RECAP_PORT:-22}
key=${RECAP_KEY:-$HOME/.ssh/id_ed25519}
remote_parent=/tmp/pi05/training_data/RoboSynChallenge
name=cobotmagic_Sim_sample_loading_expert_v21
partial="$remote_parent/.$name.partial"
final="$remote_parent/$name"
link=/root/code/RoboSynChallenge/policy/pi05/training_data/RoboSynChallenge/$name
ssh_opts=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "$key")
rsync_rsh="ssh -p $port -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $key"

[[ -f "$source_dir/meta/info.json" ]]
ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$remote_parent' '$(dirname "$link")'; test ! -e '$final'"
rsync -a --partial --delete --info=progress2 -e "$rsync_rsh" "$source_dir/" "$remote:$partial/"

read -r local_files local_bytes < <(
    find "$source_dir" -type f -printf '%s\n' | awk '{n++; s+=$1} END {printf "%d %.0f\n", n, s}'
)
read -r remote_files remote_bytes < <(
    ssh "${ssh_opts[@]}" "$remote" \
        "find '$partial' -type f -printf '%s\\n' | awk '{n++; s+=\$1} END {printf \"%d %.0f\\n\", n, s}'"
)
[[ "$local_files/$local_bytes" == "$remote_files/$remote_bytes" ]]

ssh "${ssh_opts[@]}" "$remote" "
set -e
mv '$partial' '$final'
ln -s '$final' '$link'
python3 - <<'PY'
import json
p='$final/meta/info.json'
i=json.load(open(p))
assert i['codebase_version']=='v2.1'
assert i['total_episodes']==756
assert i['total_frames']==281232
assert i['total_tasks']==1
assert 'complementary_info.acp_indicator_round1' not in i['features']
print({'episodes':i['total_episodes'],'frames':i['total_frames'],'tasks':i['total_tasks']})
PY
"
printf 'expert dataset ready: %s files, %s bytes\n' "$remote_files" "$remote_bytes"
