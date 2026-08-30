#!/usr/bin/env bash
set -euo pipefail

# 训练机地址不入库：运行前 export RECAP_REMOTE=user@host RECAP_PORT=<port> RECAP_KEY=~/.ssh/id_ed25519

source_root=$HOME/FermiBotNas/dataset/RoboSynChallenge/Sim_clean_filtered_pruned
remote=${RECAP_REMOTE:?set RECAP_REMOTE=user@host}
port=${RECAP_PORT:-22}
key=${RECAP_KEY:-$HOME/.ssh/id_ed25519}
remote_root=/tmp/pi05/training_data/RoboSynChallenge/all10_expert_v21
link=/root/code/RoboSynChallenge/policy/pi05/training_data/RoboSynChallenge/all10_expert_v21
ssh_opts=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "$key")
rsync_rsh="ssh -p $port -o BatchMode=yes -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i $key"
tasks=(click_bell drawer_open_place handle_basket item_assembly items_handover manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring)

ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$remote_root' '$(dirname "$link")'; if test ! -e '$link'; then ln -s '$remote_root' '$link'; fi"

for task in "${tasks[@]}"; do
    name="cobotmagic_Sim_$task"
    source_dir="$source_root/$name"
    partial="$remote_root/.$name.partial"
    final="$remote_root/$name"
    [[ -f "$source_dir/meta/info.json" ]]
    read -r local_files local_bytes < <(
        find "$source_dir" -type f -printf '%s\n' | awk '{n++; s+=$1} END {printf "%d %.0f\n", n, s}'
    )
    if ssh "${ssh_opts[@]}" "$remote" "test -f '$final/meta/info.json'"; then
        read -r remote_files remote_bytes < <(
            ssh "${ssh_opts[@]}" "$remote" \
                "find '$final' -type f -printf '%s\\n' | awk '{n++; s+=\$1} END {printf \"%d %.0f\\n\", n, s}'"
        )
        [[ "$local_files/$local_bytes" == "$remote_files/$remote_bytes" ]]
        echo "$name already verified; skipping"
        continue
    fi
    echo "uploading $name ($local_files files, $local_bytes bytes)"
    ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$partial'; test ! -e '$final'"
    rsync -a --partial --delete --info=progress2 -e "$rsync_rsh" "$source_dir/" "$remote:$partial/"
    read -r remote_files remote_bytes < <(
        ssh "${ssh_opts[@]}" "$remote" \
            "find '$partial' -type f -printf '%s\\n' | awk '{n++; s+=\$1} END {printf \"%d %.0f\\n\", n, s}'"
    )
    [[ "$local_files/$local_bytes" == "$remote_files/$remote_bytes" ]]
    ssh "${ssh_opts[@]}" "$remote" "mv '$partial' '$final'"
    echo "$name verified"
done

ssh "${ssh_opts[@]}" "$remote" "python3 - <<'PY'
import json
from pathlib import Path
root=Path('$remote_root')
rows=[]
for path in sorted(root.glob('cobotmagic_Sim_*')):
    info=json.load(open(path/'meta/info.json'))
    assert info['codebase_version']=='v2.1'
    assert info['total_tasks']==1
    assert not any(key.startswith('complementary_info.') for key in info['features'])
    rows.append((path.name,info['total_episodes'],info['total_frames']))
assert len(rows)==10
assert sum(row[1] for row in rows)==9515
assert sum(row[2] for row in rows)==2658988
print({'datasets':len(rows),'episodes':sum(r[1] for r in rows),'frames':sum(r[2] for r in rows)})
PY"
echo "all 10 expert datasets are ready"
