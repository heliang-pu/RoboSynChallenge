#!/usr/bin/env bash
# 阶段 5:价值函数(pistar06)训练,脱离会话运行
# 用法: 04_value_train.sh <task> <tag> [steps=8000] [bs=64] [gpu=0]
# 参考: bs=64 ≈ 27.5GB 显存、5.4 s/步;350 集池 1 epoch ≈ 2500 步。不必训满——3000 步左右先跑 05_value_qc.sh 选档。
# 监控: tail -f outputs/value_train/value_<task>_<tag>/train.log ;wandb 链接见日志 "Track this run"
# 停止: 05 选好 checkpoint 后 kill <pid>(pid 见下方输出),checkpoint 每 500 步在 checkpoints/ 下
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}; STEPS=${3:-8000}; BS=${4:-64}; GPU=${5:-0}
WORK=$WORK_ROOT/${TASK}_${TAG}; need_file "$WORK/merged_v30/meta/info.json"
OUT=$REPO/outputs/value_train/value_${TASK}_${TAG}; [ -e "$OUT" ] && die "$OUT 已存在;删除后重跑,或续训: --resume=true --config_path=$OUT/checkpoints/last/pretrained_model/value_train_config.json"
mkdir -p "$OUT"; need_file "$MODEL_SIGLIP"; need_file "$MODEL_LM"
cat > "$OUT/train_cmd.sh" <<EOS
#!/usr/bin/env bash
cd "$REPO"; export HF_LEROBOT_HOME="$WORK" PYTHONPATH="$EVO_SRC" CUDA_VISIBLE_DEVICES=$GPU
exec "$PY_EVO" -m lerobot.scripts.lerobot_value_train \\
  --dataset.repo_id=local/merged_v30 --dataset.root="$WORK/merged_v30" \\
  --value.type=pistar06 --value.dtype=bfloat16 --batch_size=$BS --steps=$STEPS --save_freq=500 --log_freq=20 \\
  --value.use_gradient_checkpointing=true --value.freeze_vision_encoder=false --value.freeze_language_model=false \\
  --value.vision_repo_id="$MODEL_SIGLIP" --value.language_repo_id="$MODEL_LM" \\
  --output_dir="$OUT" --job_name="value_${TASK}_${TAG}" --wandb.enable=true
EOS
chmod +x "$OUT/train_cmd.sh"
pid=$(detach "$OUT/train.log" bash "$OUT/train_cmd.sh")
log "已脱离会话启动 pid=$pid  日志: $OUT/train.log"
for i in $(seq 1 30); do sleep 5; grep -a "Track this run" "$OUT/train.log" 2>/dev/null && break; done
log "约 $((STEPS*54/10/3600)) 小时训完;建议 ~3000 步时跑: 05_value_qc.sh $TASK $TAG 002000 003000 004000"
