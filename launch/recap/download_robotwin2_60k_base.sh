#!/usr/bin/env bash
set -euo pipefail

repo=Hoshipu/pi05-robotwin2-random-60k
revision=3c55cf52efd07211b7e8f8ee77e9f5f263cafc85
root=/tmp/pi05/base_weights/Hoshipu_pi05-robotwin2-random-60k

mkdir -p "$root"
rm -f "$root/.download_complete"

# The final large Xet shard frequently hits huggingface_hub's read timeout on
# this host. Preserve and resume its existing partial with curl, which has no
# short read timeout, then let hf verify/fill the small files.
target="$root/params/ocdbt.process_0/d/f5f62bb52ba694c91d36c53fc642c4f1"
partial="$root/.cache/huggingface/download/params/ocdbt.process_0/d/xckbhF5JMTets4y-loNgDFi-k2U=.605e7ea296e7e58e022d09b23a503413df4bca47c84c0f314354b24fcd7d3260.incomplete"
url="https://hf-mirror.com/$repo/resolve/$revision/params/ocdbt.process_0/d/f5f62bb52ba694c91d36c53fc642c4f1"
if [[ ! -f "$target" ]]; then
    mkdir -p "$(dirname "$partial")" "$(dirname "$target")"
    touch "$partial"
    curl -L --fail --connect-timeout 30 --retry 50 --retry-all-errors \
        --retry-delay 2 -C - -o "$partial" "$url"
    [[ "$(stat -c %s "$partial")" == 2272450752 ]]
    mv "$partial" "$target"
fi

HF_ENDPOINT=https://hf-mirror.com hf download "$repo" \
    --revision "$revision" --include 'params/*' \
    --local-dir "$root" --max-workers 2

python3 - "$root/params" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = [path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts]
count = len(files)
size = sum(path.stat().st_size for path in files)
print({"files": count, "bytes": size})
assert count == 13, count
assert size == 12_440_028_997, size
assert (root / "_METADATA").is_file()
assert (root / "manifest.ocdbt").is_file()
PY
touch "$root/.download_complete"
echo "RobotWin2 60K params ready: $root/params"
