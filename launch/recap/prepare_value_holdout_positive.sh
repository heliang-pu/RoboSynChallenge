#!/usr/bin/env bash
# Convert the original NAS v2.1 rollout to a local, read-only-source-derived
# v3.0 cache for value-model held-out evaluation.
set -euo pipefail

repo=/home/fmc3/workspace/RoboSynChallenge
source_dataset=/home/fmc3/FermiBotNas/dataset/RoboSynChallenge/recap_no_reward_dataset/simrecap_sample_loading_round1
work="$repo/lerobot_dataset/.simrecap_work/value_qc_freeze_vlm_bs96_steps8000"
cache_parent="$work/positive_cache"
name=original_rollout_350_v30
output="$cache_parent/$name"
ready="$output/.holdout_v30_ready"
python=/home/fmc3/miniconda3/envs/evo-rl/bin/python

mkdir -p "$cache_parent"
exec 9>"$cache_parent/.prepare.lock"
flock 9

if [[ -f "$ready" ]]; then
    echo "$output"
    exit 0
fi

[[ -f "$source_dataset/meta/info.json" ]] || {
    echo "missing source dataset: $source_dataset" >&2
    exit 1
}
[[ -f "$source_dataset/episode_success.json" ]] || {
    echo "missing corrected label sidecar: $source_dataset/episode_success.json" >&2
    exit 1
}

# Only these exact paths are derived cache artifacts. The NAS source remains a
# symlink target and is never edited by the converter.
if [[ -L "$output" ]]; then unlink "$output"; elif [[ -e "$output" ]]; then rm -rf -- "$output"; fi
[[ ! -e "${output}_v30" ]] || rm -rf -- "${output}_v30"
if [[ -L "${output}_old" ]]; then unlink "${output}_old"; elif [[ -e "${output}_old" ]]; then rm -rf -- "${output}_old"; fi
ln -s "$source_dataset" "$output"

PYTHONPATH="$repo/third_party/evo_rl/src" "$python" \
    -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id "$name" --root "$cache_parent" --push-to-hub false

[[ -L "${output}_old" ]] && unlink "${output}_old"
cp -a "$source_dataset/episode_success.json" "$output/episode_success.json"
"$python" - "$output" <<'PYEOF'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1])
info = json.load(open(root / "meta/info.json"))
labels = json.load(open(root / "episode_success.json"))["episodes"]
episode_files = sorted((root / "meta/episodes").glob("**/*.parquet"))
data_files = sorted((root / "data").glob("**/*.parquet"))
assert info["codebase_version"] == "v3.0", info["codebase_version"]
assert info["total_episodes"] == 350, info["total_episodes"]
assert len(labels) == 350, len(labels)
assert episode_files and data_files
assert sum(pq.read_metadata(path).num_rows for path in episode_files) == 350
assert sum(pq.read_metadata(path).num_rows for path in data_files) == info["total_frames"]
print({"episodes": 350, "success": sum(bool(row["success"]) for row in labels)})
PYEOF
touch "$ready"
echo "$output"
