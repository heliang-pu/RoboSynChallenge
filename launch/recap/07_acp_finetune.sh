#!/usr/bin/env bash
# 阶段 8:pi0.5 ACP 微调(prompt 注入 Advantage 标签),脱离会话运行
# 用法: 07_acp_finetune.sh <task> <tag> <exp_name> [gpu=0] [weights_params_dir]
# 例:   07_acp_finetune.sh sample_loading round1 sample_loading_round1 0 ./checkpoints/pi05_base_robosynchallenge_full/sample_loading/28000/params
# 说明: 训练配置 pi05_sim_recap 通过环境变量取 repo_id / indicator 列 / 初始权重(见 config.py),同一配置名服务所有轮次;
#       norm stats 按 repo_id 分目录,本脚本自行检查并计算(finetune.sh 的检查只看配置名,跨轮会误判"已存在")。
#       ACP 开关:SIMRECAP_ACP_ENABLE=1|0;dropout:SIMRECAP_ACP_DROPOUT=0.3;
#       保存间隔:SIMRECAP_SAVE_INTERVAL=2500;多卡:SIMRECAP_FSDP_DEVICES=8。
#       兼容旧写法 SIMRECAP_INDICATOR_KEY=none。
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}; EXP=${3:?exp_name}; GPU=${4:-0}
WEIGHTS=${5:-gs://openpi-assets/checkpoints/pi05_base/params}
NAME=simrecap_${TASK}_${TAG}; PI05=$REPO/policy/pi05
need_file "$PI05/training_data/RoboSynChallenge/$NAME/meta/info.json"
export SIMRECAP_REPO_ID="RoboSynChallenge/$NAME"
export SIMRECAP_INDICATOR_KEY="${SIMRECAP_INDICATOR_KEY:-complementary_info.acp_indicator_$TAG}"
export SIMRECAP_ACP_ENABLE="${SIMRECAP_ACP_ENABLE:-1}"
export SIMRECAP_ACP_DROPOUT="${SIMRECAP_ACP_DROPOUT:-0.3}"
export SIMRECAP_SAVE_INTERVAL="${SIMRECAP_SAVE_INTERVAL:-2500}"
export SIMRECAP_BATCH_SIZE="${SIMRECAP_BATCH_SIZE:-64}"
export SIMRECAP_FSDP_DEVICES="${SIMRECAP_FSDP_DEVICES:-1}"
export SIMRECAP_NUM_TRAIN_STEPS="${SIMRECAP_NUM_TRAIN_STEPS:-20000}"
export SIMRECAP_CHECKPOINT_BASE_DIR="${SIMRECAP_CHECKPOINT_BASE_DIR:-./checkpoints}"
export SIMRECAP_WEIGHTS="$WEIGHTS"
export HF_LEROBOT_HOME="$PI05/training_data" CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd "$PI05"
if [ ! -f "assets/pi05_sim_recap/$SIMRECAP_REPO_ID/norm_stats.json" ]; then
    log "计算 norm stats(按 repo_id 分目录)"; uv run --frozen scripts/compute_norm_stats.py --config-name pi05_sim_recap || die "norm stats 失败"
fi
OUT=${SIMRECAP_CHECKPOINT_BASE_DIR%/}/pi05_sim_recap/$EXP
[[ "$OUT" = /* ]] || OUT=$PI05/$OUT
[ -e "$OUT" ] && die "$OUT 已存在(会被训练器清空);换 exp_name 或手动删除"
# 启动脚本与日志放旁边的 .launch 目录:openpi 会清空/重建 checkpoint 目录
LAUNCH=${OUT}.launch; mkdir -p "$LAUNCH"
cat > "$LAUNCH/train_cmd.sh" <<EOS
#!/usr/bin/env bash
cd "$PI05"; export SIMRECAP_REPO_ID="$SIMRECAP_REPO_ID" SIMRECAP_INDICATOR_KEY="$SIMRECAP_INDICATOR_KEY" SIMRECAP_WEIGHTS="$SIMRECAP_WEIGHTS"
export SIMRECAP_ACP_ENABLE="$SIMRECAP_ACP_ENABLE" SIMRECAP_ACP_DROPOUT="$SIMRECAP_ACP_DROPOUT" SIMRECAP_SAVE_INTERVAL="$SIMRECAP_SAVE_INTERVAL"
export SIMRECAP_BATCH_SIZE="$SIMRECAP_BATCH_SIZE" SIMRECAP_FSDP_DEVICES="$SIMRECAP_FSDP_DEVICES" SIMRECAP_NUM_TRAIN_STEPS="$SIMRECAP_NUM_TRAIN_STEPS"
export SIMRECAP_CHECKPOINT_BASE_DIR="$SIMRECAP_CHECKPOINT_BASE_DIR"
export HF_LEROBOT_HOME="$HF_LEROBOT_HOME" CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
exec uv run --frozen scripts/train.py pi05_sim_recap --exp-name="$EXP"
EOS
chmod +x "$LAUNCH/train_cmd.sh"; pid=$(detach "$LAUNCH/train.log" bash "$LAUNCH/train_cmd.sh")
log "已脱离会话启动 pid=$pid 日志:$LAUNCH/train.log ACP=$SIMRECAP_ACP_ENABLE dropout=$SIMRECAP_ACP_DROPOUT indicator=$SIMRECAP_INDICATOR_KEY save=$SIMRECAP_SAVE_INTERVAL bs=$SIMRECAP_BATCH_SIZE fsdp=$SIMRECAP_FSDP_DEVICES steps=$SIMRECAP_NUM_TRAIN_STEPS"
log "训完评估: 08_eval.sh $TASK $TAG $EXP"
