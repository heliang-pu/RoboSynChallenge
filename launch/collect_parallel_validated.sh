#!/usr/bin/env bash

# Collect a target episode count in independent parallel shards, validate every
# shard, merge into a fresh v3.0 dataset, convert to v2.1, and validate again
# with the target pi0.5 environment.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <task> <setting> <total_episodes> <workers> [output_id]" >&2
    exit 2
fi

TASK_NAME=$1
SETTING=$2
TOTAL_EPISODES=$3
WORKERS=$4
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ID=${5:-"cobotmagic_Sim_${TASK_NAME}_validated_${TOTAL_EPISODES}_${RUN_ID}"}

if ! [[ "$TOTAL_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "total_episodes must be a positive integer" >&2
    exit 2
fi
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || [ "$WORKERS" -gt "$TOTAL_EPISODES" ]; then
    echo "workers must be between 1 and total_episodes" >&2
    exit 2
fi

cd "$REPO_ROOT"
DATASET_ROOT="$REPO_ROOT/lerobot_dataset/$TASK_NAME"
RUN_DIR="$DATASET_ROOT/parallel_runs/$RUN_ID"
REPORT_DIR="$DATASET_ROOT/validation_reports"
mkdir -p "$RUN_DIR" "$REPORT_DIR"

declare -a PIDS LOGS
stop_children() {
    for pid in "${PIDS[@]:-}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
}
trap stop_children INT TERM

base=$((TOTAL_EPISODES / WORKERS))
remainder=$((TOTAL_EPISODES % WORKERS))

for ((worker = 0; worker < WORKERS; worker++)); do
    quota=$base
    if [ "$worker" -lt "$remainder" ]; then
        quota=$((quota + 1))
    fi
    log="$RUN_DIR/worker_${worker}_${quota}.log"
    LOGS+=("$log")
    echo "[parallel] worker $worker quota=$quota log=$log"
    "$SCRIPT_DIR/collect_until_valid.sh" "$TASK_NAME" "$SETTING" "$quota" 3_0 \
        >"$log" 2>&1 &
    PIDS+=("$!")

    # Dataset suffix selection is not atomic. Stagger workers until the prior
    # one has created its candidate directory, eliminating startup collisions.
    for _ in $(seq 1 120); do
        if grep -aq "Created LeRobot dataset at:" "$log"; then
            break
        fi
        if ! kill -0 "${PIDS[$worker]}" 2>/dev/null; then
            echo "worker $worker exited before creating a dataset; see $log" >&2
            exit 1
        fi
        sleep 1
    done
done

for index in "${!PIDS[@]}"; do
    wait "${PIDS[$index]}"
    echo "[parallel] worker $index passed its shard gates"
done
trap - INT TERM

declare -a SOURCE_PATHS SOURCE_IDS
for log in "${LOGS[@]}"; do
    path=$(grep -a '^VALIDATED_DATASET=' "$log" | tail -n 1 | cut -d= -f2-)
    if [ -z "$path" ] || [ ! -d "$path" ]; then
        echo "Cannot resolve validated shard from $log" >&2
        exit 1
    fi
    SOURCE_PATHS+=("$path")
    SOURCE_IDS+=("$(basename "$path")")
done

if [ -e "$DATASET_ROOT/$OUTPUT_ID" ]; then
    echo "Refusing to overwrite merge output: $DATASET_ROOT/$OUTPUT_ID" >&2
    exit 1
fi

repo_ids="["
for index in "${!SOURCE_IDS[@]}"; do
    if [ "$index" -gt 0 ]; then
        repo_ids+=", "
    fi
    repo_ids+="'${SOURCE_IDS[$index]}'"
done
repo_ids+="]"

echo "[parallel] merging validated v3.0 shards into $OUTPUT_ID"
python -m lerobot.scripts.lerobot_edit_dataset \
    --root "$DATASET_ROOT" \
    --repo_id "$OUTPUT_ID" \
    --push_to_hub false \
    --operation.type merge \
    --operation.repo_ids "$repo_ids"

V3_REPORT="$REPORT_DIR/${OUTPUT_ID}_merged_v3.0.json"
python "$REPO_ROOT/scripts/validate_lerobot_dataset.py" "$DATASET_ROOT/$OUTPUT_ID" \
    --expected-episodes "$TOTAL_EPISODES" --producer-exit-code 0 --report "$V3_REPORT"

python "$REPO_ROOT/scripts/convert_lerobot3.0_to_2.1.py" \
    --repo-id "$OUTPUT_ID" --root "$DATASET_ROOT"

MODEL_PYTHON=${MODEL_PYTHON:-"$REPO_ROOT/policy/pi05/.venv/bin/python"}
V21_REPORT="$REPORT_DIR/${OUTPUT_ID}_merged_v2.1_pi05.json"
"$MODEL_PYTHON" "$REPO_ROOT/scripts/validate_lerobot_dataset.py" "$DATASET_ROOT/$OUTPUT_ID" \
    --expected-episodes "$TOTAL_EPISODES" --producer-exit-code 0 --report "$V21_REPORT"

echo "VALIDATED_DATASET=$DATASET_ROOT/$OUTPUT_ID"
echo "VALIDATION_REPORT=$V21_REPORT"
echo "PARALLEL_RUN_DIR=$RUN_DIR"
