#!/usr/bin/env bash
# Resumable 1,000-episode collector: 10 deterministic batches of 100 episodes.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [task ...]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NAS_ROOT="${NAS_ROOT:-$HOME/FermiBotNas/dataset/RoboSynChallenge/Syn/seeded_1000_20260827}"
STAGING_ROOT="${STAGING_ROOT:-$REPO_ROOT/lerobot_dataset/seeded_1000_20260827}"
BATCH_EPISODES="${BATCH_EPISODES:-100}"
BATCH_COUNT="${BATCH_COUNT:-10}"
BATCH_IDS="${BATCH_IDS:-$(seq 0 $((BATCH_COUNT - 1)))}"
cd "$REPO_ROOT"

for task in "$@"; do
    action="configs/$task/action_config.json"
    base="configs/$task/random/gym_config.json"
    [ -f "$action" ] && [ -f "$base" ] || { echo "missing config for $task" >&2; exit 2; }
    task_root="$NAS_ROOT/$task"
    staging_task_root="$STAGING_ROOT/$task"
    settle_steps="${SUCCESS_SETTLE_STEPS:-25}"
    # The organizer's handle_basket predicate requires the final placement to
    # remain valid for 75 consecutive env steps.  A 25-step post-rollout hold
    # can therefore never turn a correct expert rollout into a saved episode.
    [ "$task" = "handle_basket" ] && settle_steps=75
    mkdir -p "$task_root/runtime_configs" "$task_root/batches" "$task_root/logs" \
        "$staging_task_root/batches" "$staging_task_root/failed_partial"

    for batch in $BATCH_IDS; do
        case "$batch" in
            ''|*[!0-9]*)
                echo "[$task] invalid batch id: $batch" >&2
                exit 2
                ;;
        esac
        if [ "$batch" -ge "$BATCH_COUNT" ]; then
            echo "[$task] batch id out of range: $batch (expected 0..$((BATCH_COUNT - 1)))" >&2
            exit 2
        fi
        batch_id=$(printf '%02d' "$batch")
        batch_root="$task_root/batches/batch_$batch_id"
        staging_batch_root="$staging_task_root/batches/batch_$batch_id"
        complete="$batch_root/.complete.json"
        [ -f "$complete" ] && { echo "[$task] batch $batch_id already complete"; continue; }
        # A producer killed in the middle of a batch leaves a valid-looking but
        # incomplete LeRobot directory.  Never append a restarted deterministic
        # seed stream to it: that would duplicate the beginning of the batch.
        # Preserve it for diagnosis and restart the whole 100-episode batch in
        # an empty directory with the same master seed.
        if [ -d "$batch_root" ] && find "$batch_root" -mindepth 1 -print -quit | grep -q .; then
            partial_root="$task_root/failed_partial"
            partial="$partial_root/batch_${batch_id}_$(date -u +%Y%m%dT%H%M%SZ)"
            mkdir -p "$partial_root"
            mv "$batch_root" "$partial"
            previous_log="$task_root/logs/batch_$batch_id.log"
            if [ -f "$previous_log" ]; then
                cp "$previous_log" "$partial/producer.log"
            fi
            printf '{"task":"%s","batch":%d,"status":"quarantined_partial","path":"%s"}\n' \
                "$task" "$batch" "$partial" >> "$task_root/MANIFEST.jsonl"
            echo "[$task] quarantined incomplete batch $batch_id -> $partial"
        fi
        if [ -d "$staging_batch_root" ] && find "$staging_batch_root" -mindepth 1 -print -quit | grep -q .; then
            staging_partial="$staging_task_root/failed_partial/batch_${batch_id}_$(date -u +%Y%m%dT%H%M%SZ)"
            mv "$staging_batch_root" "$staging_partial"
            previous_log="$task_root/logs/batch_$batch_id.log"
            if [ -f "$previous_log" ]; then
                cp "$previous_log" "$staging_partial/producer.log"
            fi
            echo "[$task] quarantined local staging batch $batch_id -> $staging_partial"
        fi
        seed=$(( 2026082700 + $(printf '%s' "$task" | cksum | awk '{print $1 % 100000}') + batch ))
        runtime="$task_root/runtime_configs/batch_$batch_id.json"
        "$PYTHON_BIN" - "$base" "$runtime" "$staging_batch_root" "$BATCH_EPISODES" "$task" "$seed" <<'PY'
import json, sys
source, target, save_path, episodes, task, seed = sys.argv[1:]
cfg=json.load(open(source))
cfg['max_episodes']=int(episodes)
params=cfg['env']['dataset']['lerobot']['params']
params['save_path']=save_path
extra=params.setdefault('extra', {})
extra.update({'collection_plan':'seeded_1000_20260827','task':task,'batch_seed':int(seed),'batch_episodes':int(episodes)})
json.dump(cfg, open(target,'w'), indent=2)
PY
        printf '{"task":"%s","batch":%d,"seed":%d,"status":"started"}\n' "$task" "$batch" "$seed" >> "$task_root/MANIFEST.jsonl"
        log="$task_root/logs/batch_$batch_id.log"
        echo "[$task] batch $batch_id seed=$seed -> $log"
        "$PYTHON_BIN" -m scripts.run_env --gym_config "$runtime" --action_config "$action" \
                --num_envs 1 --max_episodes "$BATCH_EPISODES" --headless --seed "$seed" \
                --report_task_success --save_only_success --success_settle_steps "$settle_steps" --max_generation_attempts 1000 \
                > "$log" 2>&1 || {
            status=$?
            printf '{"task":"%s","batch":%d,"seed":%d,"status":"producer_failed","exit_code":%d}\n' \
                "$task" "$batch" "$seed" "$status" >> "$task_root/MANIFEST.jsonl"
            echo "[$task] batch $batch_id producer failed (exit $status); rerun the same command to retry" >&2
            exit "$status"
        }
        staging_complete="$staging_batch_root/.complete.json"
        "$PYTHON_BIN" - "$staging_batch_root" "$staging_complete" "$task" "$batch" "$seed" "$BATCH_EPISODES" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); complete=Path(sys.argv[2])
task, batch, seed, expected = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
infos=list(root.glob('**/meta/info.json'))
episodes=sum(json.load(open(path)).get('total_episodes',0) for path in infos)
if episodes != expected:
    raise SystemExit(f'expected {expected} episodes, found {episodes}')
sidecars=list(root.glob('**/episode_success.json'))
if len(sidecars) != 1:
    raise SystemExit(f'expected one seed sidecar, found {len(sidecars)}')
payload=json.load(open(sidecars[0]))
if payload.get('master_seed') != seed:
    raise SystemExit(f'sidecar seed {payload.get("master_seed")} != {seed}')
if payload.get('saved_episode_count') != expected or len(payload.get('episodes', [])) != expected:
    raise SystemExit('seed sidecar episode count mismatch')
# This collector always runs with --save_only_success.  Older producer
# processes may still carry the pre-fix wrapper that sampled _task_success
# after reset and wrote false labels.  Normalize from the authoritative save
# decision before promotion, then make all-success a hard gate.
if not all(item.get('success') is True for item in payload['episodes']):
    for item in payload['episodes']:
        item['success'] = True
    payload['success_label_source'] = 'save_only_success'
    sidecars[0].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
if not all(item.get('success') is True for item in payload['episodes']):
    raise SystemExit('seed sidecar contains a non-success label')
complete.write_text(json.dumps({'task':task,'batch':batch,'seed':seed,'episodes':episodes}, indent=2)+'\n')
PY
        upload_root="$task_root/batches/.batch_${batch_id}.uploading_$(hostname)_$$"
        mkdir -p "$upload_root"
        rsync -a --delete "$staging_batch_root/" "$upload_root/"
        "$PYTHON_BIN" - "$upload_root" "$BATCH_EPISODES" "$seed" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); expected=int(sys.argv[2]); seed=int(sys.argv[3])
episodes=sum(json.load(open(p)).get('total_episodes', 0) for p in root.glob('**/meta/info.json'))
sidecars=list(root.glob('**/episode_success.json'))
if episodes != expected or len(sidecars) != 1:
    raise SystemExit(f'uploaded batch verification failed: episodes={episodes}, sidecars={len(sidecars)}')
payload=json.load(open(sidecars[0]))
if payload.get('master_seed') != seed or payload.get('saved_episode_count') != expected:
    raise SystemExit('uploaded seed sidecar verification failed')
if not all(item.get('success') is True for item in payload.get('episodes', [])):
    raise SystemExit('uploaded seed sidecar contains a non-success label')
PY
        mv "$upload_root" "$batch_root"
        complete="$batch_root/.complete.json"
        [ -f "$complete" ] || { echo "missing promoted completion marker: $complete" >&2; exit 1; }
        rm -rf -- "$staging_batch_root"
        printf '{"task":"%s","batch":%d,"seed":%d,"status":"complete","episodes":%d}\n' "$task" "$batch" "$seed" "$BATCH_EPISODES" >> "$task_root/MANIFEST.jsonl"
    done
done
