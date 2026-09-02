#!/usr/bin/env bash

# Collect one isolated batch and promote it to "validated" only after every
# dataset and training-read gate passes. This script never merges datasets.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

usage() {
    cat <<'EOF'
Usage:
  collect_validated_batch.sh <task> <setting> <episodes> <format> [extra run_task args...]

Arguments:
  format: 3_0, or 2_1 for pi0.5-compatible conversion and validation

Environment:
  MODEL_PYTHON  Python from the target training environment. For format 2_1,
                defaults to policy/pi05/.venv/bin/python.
EOF
}

if [ "$#" -lt 4 ]; then
    usage >&2
    exit 2
fi

TASK_NAME=$1
SETTING=$2
EPISODES=$3
FORMAT=$4
shift 4

if [[ "$FORMAT" != "3_0" && "$FORMAT" != "2_1" ]]; then
    echo "format must be 3_0 or 2_1" >&2
    exit 2
fi
if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "episodes must be a positive integer" >&2
    exit 2
fi

cd "$REPO_ROOT"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="$REPO_ROOT/lerobot_dataset/$TASK_NAME/validation_logs"
REPORT_DIR="$REPO_ROOT/lerobot_dataset/$TASK_NAME/validation_reports"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
RUN_LOG="$LOG_DIR/${TASK_NAME}_${SETTING}_${EPISODES}_${RUN_ID}.log"

echo "[collect] generating isolated v3.0 candidate; log: $RUN_LOG"
set +e
bash "$SCRIPT_DIR/run_task_seeded.sh" "$TASK_NAME" "$SETTING" 3_0 \
    --max_episodes "$EPISODES" --headless "$@" 2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [ "$RUN_STATUS" -ne 0 ]; then
    echo "[gate 1/5] FAIL: producer exit code $RUN_STATUS; queue stopped" >&2
    exit "$RUN_STATUS"
fi
echo "[gate 1/5] PASS: producer exited naturally"

DATASET_PATH=$(sed -n 's/.*Created LeRobot dataset at: //p' "$RUN_LOG" | tail -n 1 | tr -d '\r')
if [ -z "$DATASET_PATH" ] || [ ! -d "$DATASET_PATH" ]; then
    echo "Cannot resolve the newly generated dataset from $RUN_LOG; queue stopped" >&2
    exit 1
fi

DATASET_ID=$(basename "$DATASET_PATH")
V3_REPORT="$REPORT_DIR/${DATASET_ID}_v3.0.json"
python "$REPO_ROOT/scripts/validate_lerobot_dataset.py" "$DATASET_PATH" \
    --expected-episodes "$EPISODES" --producer-exit-code 0 --report "$V3_REPORT"
echo "[gates 2-5] PASS: native v3.0 candidate validated"

if [ "$FORMAT" == "3_0" ]; then
    echo "VALIDATED_DATASET=$DATASET_PATH"
    echo "VALIDATION_REPORT=$V3_REPORT"
    exit 0
fi

# 种子边车(--seed 采集时产生)会被转换器丢掉,先存后还
SIDECAR="$DATASET_PATH/episode_success.json"
[ -f "$SIDECAR" ] && cp "$SIDECAR" "$DATASET_PATH.episode_success.stash.json"
python "$REPO_ROOT/scripts/convert_lerobot3.0_to_2.1.py" \
    --repo-id "$DATASET_ID" --root "$(dirname "$DATASET_PATH")"
[ -f "$DATASET_PATH.episode_success.stash.json" ] && mv "$DATASET_PATH.episode_success.stash.json" "$SIDECAR"

MODEL_PYTHON=${MODEL_PYTHON:-"$REPO_ROOT/policy/pi05/.venv/bin/python"}
if [ ! -x "$MODEL_PYTHON" ]; then
    echo "Target-model Python is not executable: $MODEL_PYTHON; queue stopped" >&2
    exit 1
fi

V21_REPORT="$REPORT_DIR/${DATASET_ID}_v2.1_pi05.json"
"$MODEL_PYTHON" "$REPO_ROOT/scripts/validate_lerobot_dataset.py" "$DATASET_PATH" \
    --expected-episodes "$EPISODES" --producer-exit-code 0 --report "$V21_REPORT"
echo "[gates 2-5] PASS: converted v2.1 dataset loaded by target training environment"
echo "VALIDATED_DATASET=$DATASET_PATH"
echo "VALIDATION_REPORT=$V21_REPORT"
