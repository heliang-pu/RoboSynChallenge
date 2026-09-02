#!/usr/bin/env bash
# 价值函数质检:在合并池副本上用若干 checkpoint 做子集推理,比较成败分离度与 advantage 分布。
# 阶段 6a(质检):用若干 checkpoint 在合并池副本的 60 集子集上推理,一张表比较成败分离度与 advantage 分布
# 用法: 05_value_qc.sh <task> <tag> <step> [<step> ...]     step 形如 003000
# 子集 = 10 个 rollout 成功集 + 30 个 rollout 失败集 + 20 个专家集(按 rollout 边车与合并池布局自动推导)。
set -uo pipefail; source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TASK=${1:?task}; TAG=${2:?tag}; shift 2; STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || { echo "至少一个 step"; exit 2; }

WORK=$WORK_ROOT/${TASK}_${TAG}
QC=$WORK/qc_v30
CK=$REPO/outputs/value_train/value_${TASK}_${TAG}/checkpoints
PY=$PY_SIM
LOGDIR=${QC_LOGDIR:-$WORK}


for s in "${STEPS[@]}"; do [[ $s =~ ^[0-9]+$ && -d $CK/$s ]] || die "step 必须是数字目录名且存在: $CK/$s(如 003000)"; done
need_file "$WORK/merged_v30/meta/info.json"
[ -d "$QC" ] || { log "复制 merged_v30 -> qc_v30"; cp -r "$WORK/merged_v30" "$QC" || die "复制失败"; }
# 子集:专家集数 = 合并池集数 − rollout 集数(边车条数)
SIDECAR=$(find "$WORK/rollout_v30" -maxdepth 2 -name episode_success.json | head -1); [ -n "$SIDECAR" ] || die "找不到 rollout 边车"
EPS=$("$PY" - "$SIDECAR" "$QC" <<'PYEOF'
import sys, json
sc = json.load(open(sys.argv[1]))["episodes"]; info = json.load(open(f"{sys.argv[2]}/meta/info.json"))
n_expert = info["total_episodes"] - len(sc)
succ = [e["episode_index"] + n_expert for e in sc if e["success"]][:10]
fail = [e["episode_index"] + n_expert for e in sc if not e["success"]][:30]
print(json.dumps(sorted(list(range(min(20, n_expert))) + succ + fail)))
PYEOF
)
[ -n "$EPS" ] || die "子集推导失败"; log "子集 episodes: $EPS"
for step in "${STEPS[@]}"; do
    tag="qc$((10#$step))"
    log "推理 checkpoint $step -> 列后缀 _$tag"
    bash "$REPO/launch/run_value_infer.sh" "$QC" "$CK/$step" "$tag" 0 --runtime.batch_size 32 --dataset.episodes "$EPS" \
        > "$LOGDIR/value_qc_infer_$step.log" 2>&1 || { log "推理 $step 失败,见 $LOGDIR/value_qc_infer_$step.log"; exit 1; }
done
log "分析"
"$PY" - "$QC" "$SIDECAR" "${STEPS[@]}" <<'PYEOF'
import sys, glob, json
import numpy as np, pyarrow.parquet as pq
qc, sidecar, steps = sys.argv[1], sys.argv[2], sys.argv[3:]
labels = {}
for f in glob.glob(f"{qc}/meta/episodes/**/*.parquet", recursive=True):
    t = pq.read_table(f, columns=["episode_index", "episode_success"])
    labels.update(dict(zip(t.column(0).to_pylist(), t.column(1).to_pylist())))
n_expert = len(labels) - len(json.load(open(sidecar))["episodes"])
files = sorted(glob.glob(f"{qc}/data/**/*.parquet", recursive=True))
print(f"{'ckpt':>6} {'成功首帧V':>9} {'失败首帧V':>9} {'差':>6} {'A.std':>7} {'|A|<.01':>7} {'ind:专家':>7} {'ind:成功':>7} {'ind:失败':>7}")
for step in steps:
    tag = f"qc{int(step)}"
    cols = ["episode_index", "frame_index", f"complementary_info.value_{tag}", f"complementary_info.advantage_{tag}", f"complementary_info.acp_indicator_{tag}"]
    ep, fi, v, a, ind = [], [], [], [], []
    for f in files:
        t = pq.read_table(f, columns=cols)
        ep += t.column(0).to_pylist(); fi += t.column(1).to_pylist(); v += t.column(2).to_pylist(); a += t.column(3).to_pylist(); ind += t.column(4).to_pylist()
    ep, fi = np.asarray(ep), np.asarray(fi)
    v, a, ind = [np.asarray(x, dtype=float).reshape(len(ep)) for x in (v, a, ind)]
    keep = ~np.isnan(v); ep, fi, v, a, ind = ep[keep], fi[keep], v[keep], a[keep], ind[keep]
    succ = np.array([labels[e] == "success" for e in ep]); ro = ep >= n_expert; first = fi == 0
    sv, fv = v[first & succ & ro].mean(), v[first & ~succ & ro].mean()
    print(f"{int(step):>6} {sv:>9.3f} {fv:>9.3f} {sv-fv:>6.3f} {a.std():>7.4f} {(np.abs(a)<0.01).mean()*100:>6.1f}% {ind[~ro].mean()*100:>6.1f}% {ind[ro&succ].mean()*100:>6.1f}% {ind[ro&~succ].mean()*100:>6.1f}%")
print("规则: 差>=0.3 的档里选 A.std 最大 / |A|<.01 最小的。ind:* 三列是子集内的 top-30%,只看相对趋势;全量比例以 06 发布日志的 ACP stats 为准。")
PYEOF
log "完成"
