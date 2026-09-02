#!/usr/bin/env bash
# 安全停止:按 cmdline 子串找进程(排除自己),连 image-writer 孤儿一起清;之后列出仍持卡的进程
# 用法: stop.sh <cmdline 子串>     例: stop.sh lerobot_value_train | stop.sh eval_policy | stop.sh collect_until_valid
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
PAT=${1:?cmdline 子串}; pids=$(pids_by_cmd "$PAT"); [ -n "$pids" ] || { log "没有匹配 '$PAT' 的进程"; }
for p in $pids; do echo "  kill $p :: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-90)"; kill -9 "$p" 2>/dev/null; done; sleep 4
log "GPU 上仍有:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while read pid mem; do pid=${pid%,}; echo "  $pid $mem :: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-80)"; done
nvidia-smi --query-gpu=memory.free --format=csv,noheader | sed 's/^/  空闲显存: /'
