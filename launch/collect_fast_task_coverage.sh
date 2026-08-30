#!/usr/bin/env bash
# Queue a compact all-task coverage top-up after the active sample-loading jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[fast-coverage] waiting for active sample_loading coverage jobs to release the GPU"
while pgrep -f 'scripts\.run_env.*configs/sample_loading/coverage_' >/dev/null; do
    sleep 120
done

echo "[fast-coverage] starting 5 groups x 10 valid episodes"
bash "$SCRIPT_DIR/collect_coverage_queue.sh" \
    report/coverage/FAST_TASK_PLAN.json \
    click_bell,items_handover,manipulate_pipette,table_rearrangement,water_pouring \
    1 \
    --max_generation_attempts 80
