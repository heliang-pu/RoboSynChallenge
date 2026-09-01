#!/usr/bin/env bash
set -uo pipefail

repo=${ROBOSYN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
pi05="$repo/policy/pi05"
launch="$repo/.launch"
run=sample_loading_round1_vlm3500_baked_base_acp30
seed=29999
episodes=20
horizons=(25 30 40 50)

mkdir -p "$launch"

# The paired H=10 run is already active in its own tmux session. Do not start
# another model until it has released the single GPU.
while tmux has-session -t pi05-recap-eval-29999 2>/dev/null; do
    sleep 10
done

summary="$launch/pi05_recap_eval_29999_horizon_sweep.log"
printf '[%s] starting paired horizon sweep: %s\n' "$(date -Is)" "${horizons[*]}" >> "$summary"

for horizon in "${horizons[@]}"; do
    log="$launch/pi05_recap_eval_29999_h${horizon}.log"
    printf '[%s] H=%s start\n' "$(date -Is)" "$horizon" | tee -a "$summary"
    (
        cd "$pi05" || exit 1
        export PYTHONUNBUFFERED=1
        export PYTHONFAULTHANDLER=1
        export EMBODICHAIN_SIM_EXIT_PROCESS=0
        export MALLOC_ARENA_MAX=2
        export SIMRECAP_REPO_ID=RoboSynChallenge/simrecap_sample_loading_round1_vlm3500_baked_prompt
        export SIMRECAP_INDICATOR_KEY=complementary_info.acp_indicator_round1
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
        bash eval.sh sample_loading random pi05_sim_recap "$run" 0 \
            --checkpoint_id 29999 --pi0_step "$horizon" \
            --max_episodes "$episodes" --seed "$seed" \
            --headless true --eval_video_log true
    ) > >(tee "$log") 2>&1
    rc=${PIPESTATUS[0]}
    printf 'EVAL_EXIT_CODE=%s\n' "$rc" | tee -a "$log"
    printf '[%s] H=%s exit=%s\n' "$(date -Is)" "$horizon" "$rc" | tee -a "$summary"
done

printf '[%s] sweep complete\n' "$(date -Is)" | tee -a "$summary"
