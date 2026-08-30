#!/usr/bin/env bash
# 覆盖补采交付守护:扫描 lerobot_dataset/coverage/*/*/.validated,逐组 转v2.1 → pi05读取门 → 推 NAS Syn/
# 用法: deliver_coverage.sh [once|loop]   (loop 每 600s 一轮)
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
NAS_SYN=${NAS_SYN:-$HOME/FermiBotNas/dataset/RoboSynChallenge/Syn}
MODEL_PYTHON=${MODEL_PYTHON:-$REPO_ROOT/policy/pi05/.venv/bin/python}
PY=${PY:-$HOME/miniconda3/envs/robosyn/bin/python}
log(){ echo "[deliver] $(date +%H:%M:%S) $*"; }
pull_remote() {
  # 把 pro6000 上已过 v3.0 验证的组增量拉回本地(只拉带 .validated 的组,避免半成品)
  local host=${REMOTE_COLLECT_HOST:-root-pro6000-inner}
  local rroot=${REMOTE_COLLECT_ROOT:-/root/workspace/coverage/RoboSynChallenge/lerobot_dataset/coverage}
  local markers
  markers=$(ssh -n -o ConnectTimeout=15 "$host" "find $rroot -maxdepth 3 -name .validated 2>/dev/null" 2>/dev/null) || return 0
  for m in $markers; do
    local rel=${m#$rroot/}; rel=${rel%/.validated}
    [ -f "lerobot_dataset/coverage/$rel/.delivered" ] && continue
    log "从 pro6000 拉取 $rel"
    mkdir -p "lerobot_dataset/coverage/$rel"
    rsync -a "$host:$rroot/$rel/" "lerobot_dataset/coverage/$rel/" || log "拉取 $rel 失败(下一轮重试)"
    # 远端 marker 里的绝对路径换成本地路径
    if [ -f "lerobot_dataset/coverage/$rel/.validated" ]; then
      local ds_base=$(basename "$(cat "lerobot_dataset/coverage/$rel/.validated")")
      echo "$PWD/lerobot_dataset/coverage/$rel/$ds_base" > "lerobot_dataset/coverage/$rel/.validated"
    fi
  done
}
one_round() {
  pull_remote
  for marker in lerobot_dataset/coverage/*/*/.validated; do
    [ -f "$marker" ] || continue
    dir=$(dirname "$marker"); task=$(basename "$(dirname "$dir")"); slug=$(basename "$dir")
    [ -f "$dir/.delivered" ] && continue
    ds=$(cat "$marker"); [ -d "$ds" ] || { log "跳过 $task/$slug: 数据集不见了 $ds"; continue; }
    ver=$($PY -c "import json;print(json.load(open('$ds/meta/info.json'))['codebase_version'])")
    eps=$($PY -c "import json;print(json.load(open('$ds/meta/info.json'))['total_episodes'])")
    if [[ "$ver" == v3* ]]; then
      log "$task/$slug: v3.0 -> v2.1 ($eps 集)"
      [ -f "$ds/episode_success.json" ] && cp "$ds/episode_success.json" "$ds.sidecar.stash"
      $PY scripts/convert_lerobot3.0_to_2.1.py --repo-id "$(basename "$ds")" --root "$(dirname "$ds")" > "$dir/.convert.log" 2>&1 \
        || { log "$task/$slug 转换失败,见 $dir/.convert.log"; continue; }
      [ -f "$ds.sidecar.stash" ] && mv "$ds.sidecar.stash" "$ds/episode_success.json"
    fi
    log "$task/$slug: pi05 训练环境读取门"
    "$MODEL_PYTHON" scripts/validate_lerobot_dataset.py "$ds" --expected-episodes "$eps" --producer-exit-code 0 \
        --report "$dir/.v21_report.json" > "$dir/.v21_gate.log" 2>&1 \
      || { log "$task/$slug 读取门失败,见 $dir/.v21_gate.log"; continue; }
    dest="$NAS_SYN/${task}_coverage/$slug"
    log "$task/$slug: 推 NAS $dest"
    mkdir -p "$dest"
    rsync -a "$ds/" "$dest/" || { log "$task/$slug rsync 失败"; continue; }
    cp "$dir/.v21_report.json" "$dest/validation_report.json" 2>/dev/null
    date > "$dir/.delivered"
    log "✅ $task/$slug 已交付 ($eps 集, 含种子边车)"
  done
}
if [ "${1:-once}" = loop ]; then
  while true; do one_round; sleep 600; done
else one_round; fi
