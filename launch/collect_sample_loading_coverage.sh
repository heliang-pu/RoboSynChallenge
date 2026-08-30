#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$HOME/miniconda3/envs/robosyn/bin/python}
cd "$REPO_ROOT"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi

declare -a COLLECTION_GROUPS=(
    "coverage_rack_upper_feasible:120"
    "coverage_tube_right_lower_y:120"
    "coverage_yaw_low_tube_high_rack:80"
    "coverage_yaw_high_tube_high_rack:80"
)

for entry in "${COLLECTION_GROUPS[@]}"; do
    setting=${entry%%:*}
    episodes=${entry##*:}
    echo "Starting $setting ($episodes episodes)"
    "$PYTHON_BIN" -m scripts.run_env \
        --gym_config "configs/sample_loading/$setting/gym_config.json" \
        --action_config configs/sample_loading/action_config.json \
        --num_envs 1 --max_episodes "$episodes" --headless \
        --report_task_success "$@"
done
