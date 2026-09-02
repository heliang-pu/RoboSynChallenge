#!/usr/bin/env bash
# Convert the cleaned Syn v3.0 datasets to independent LeRobot v2.1 copies.
# Sources are never modified.  Outputs live under Syn_v2.1 with matching
# task grouping wherever that grouping exists in the source tree.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/robosyn/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$HOME/FermiBotNas/dataset/RoboSynChallenge/Syn}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/FermiBotNas/dataset/RoboSynChallenge/Syn_v2.1}"
CONVERTER="$REPO_ROOT/scripts/convert_lerobot3.0_to_2.1.py"

convert_one() {
    local source="$1"
    local output="$2"
    local parent name
    parent="$(dirname "$source")"
    name="$(basename "$source")"

    if [[ -e "$output" ]]; then
        echo "SKIP existing v2.1 output: $output"
        return
    fi
    echo "CONVERT $source -> $output"
    "$PYTHON_BIN" "$CONVERTER" \
        --repo-id "$name" \
        --root "$parent" \
        --output-root "$output"
}

convert_one "$SOURCE_ROOT/click_bell/cobotmagic_Sim_click_the_bell_003" "$OUTPUT_ROOT/click_bell/cobotmagic_Sim_click_the_bell_003"
convert_one "$SOURCE_ROOT/items_handover_000" "$OUTPUT_ROOT/items_handover_000"
convert_one "$SOURCE_ROOT/items_handover_001" "$OUTPUT_ROOT/items_handover_001"
convert_one "$SOURCE_ROOT/manipulate_pipette" "$OUTPUT_ROOT/manipulate_pipette"
convert_one "$SOURCE_ROOT/manipulate_pipette/manipulate_pipette_002" "$OUTPUT_ROOT/manipulate_pipette_002"
convert_one "$SOURCE_ROOT/sample_loading" "$OUTPUT_ROOT/sample_loading"
convert_one "$SOURCE_ROOT/sample_loading_coverage/coverage_rack_upper_feasible/cobotmagic_Sim_sample_loading_000" "$OUTPUT_ROOT/sample_loading_coverage/coverage_rack_upper_feasible/cobotmagic_Sim_sample_loading_000"
convert_one "$SOURCE_ROOT/sample_loading_coverage/coverage_tube_right_lower_y/cobotmagic_Sim_sample_loading_000" "$OUTPUT_ROOT/sample_loading_coverage/coverage_tube_right_lower_y/cobotmagic_Sim_sample_loading_000"
convert_one "$SOURCE_ROOT/sample_loading_coverage/coverage_yaw_low_tube_high_rack/cobotmagic_Sim_sample_loading_000" "$OUTPUT_ROOT/sample_loading_coverage/coverage_yaw_low_tube_high_rack/cobotmagic_Sim_sample_loading_000"
convert_one "$SOURCE_ROOT/sample_loading_coverage/coverage_yaw_high_tube_high_rack/cobotmagic_Sim_sample_loading_000" "$OUTPUT_ROOT/sample_loading_coverage/coverage_yaw_high_tube_high_rack/cobotmagic_Sim_sample_loading_000"
convert_one "$SOURCE_ROOT/table_rearrangement" "$OUTPUT_ROOT/table_rearrangement"
convert_one "$SOURCE_ROOT/water_pouring" "$OUTPUT_ROOT/water_pouring"
