#!/usr/bin/env bash
# 建立 LeRobot 数据根目录软链,指向 /data 上的官方数据集
set -euo pipefail
SRC=/data/RoboSynChallenge/Sim_clean_filtered_pruned
DST=/data/lerobot_home/RoboSynChallenge
mkdir -p "$DST"
n=0
for t in click_bell drawer_open_place handle_basket item_assembly items_handover \
         manipulate_pipette mixer_operating sample_loading table_rearrangement water_pouring; do
  d="cobotmagic_Sim_${t}"
  if [[ -d "$SRC/$d" ]]; then
    ln -sfn "$SRC/$d" "$DST/$d"; n=$((n+1))
  else
    echo "  [缺] $d"
  fi
done
echo "  已链接 $n/10 个任务 → $DST"
