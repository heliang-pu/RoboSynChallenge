#!/usr/bin/env bash
# sim-RECAP 公共定义(被 launch/recap/*.sh source)。人手可改的机器相关路径都集中在这里。
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK_ROOT=$REPO/lerobot_dataset/.simrecap_work            # v3.0 中间产物(隐藏)
# NAS 挂载点因机器而异(本机 ~/FermiBotNas,fmc3-1 为 ~/workspace/FermiBotNas),取第一个存在的
_first_dir(){ for d in "$@"; do [ -d "$d" ] && { echo "$d"; return; }; done; echo "$1"; }
NAS=${RECAP_NAS:-$(_first_dir "$HOME/FermiBotNas/dataset/RoboSynChallenge" "$HOME/workspace/FermiBotNas/dataset/RoboSynChallenge")}
PY_SIM=${PY_SIM:-$HOME/miniconda3/envs/robosyn/bin/python}   # lerobot 0.4.4 + pyarrow(转换/打标/边车)
PY_EVO=${PY_EVO:-$HOME/miniconda3/envs/evo-rl/bin/python}    # pandas 2.x(合并/价值训练/价值推理)
EVO_SRC=$REPO/third_party/evo_rl/src
MODEL_SIGLIP=${MODEL_SIGLIP:-$HOME/workspace/models/google/siglip-so400m-patch14-384}
MODEL_LM=${MODEL_LM:-$HOME/workspace/models/google/gemma-3-270m}

log(){ echo "[$(basename "$0" .sh)] $(date +%T) $*"; }
die(){ log "错误: $*" >&2; exit 1; }
need_file(){ [ -e "$1" ] || die "缺少 $1"; }

# 记录器会在 save_path 下自建 <robot>_<scene>_<task>_NNN 子目录;返回真正的数据集目录(多于一个时报错)
nested_dataset_dir(){ local infos; infos=$(find "$1" -maxdepth 3 -path '*/meta/info.json' 2>/dev/null | sort)
    [ -n "$infos" ] || return 1; [ "$(echo "$infos" | wc -l)" -eq 1 ] || { echo "$1 下有多个数据集: $infos" >&2; return 2; }
    dirname "$(dirname "$infos")"; }
# 剥离 parquet 的 pandas/HF 扩展元数据(数值不变,幂等,原子替换)。merge/convert 前必做,否则 pandas 2/3 各有一种崩法。
# 拒绝对软链目录操作(会改到源数据)。
strip_meta(){ [ -L "$1" ] && { echo "strip_meta: $1 是软链,拒绝原地改写" >&2; return 1; }
"$PY_SIM" - "$1" <<'PYEOF'
import sys, glob, os
import pyarrow as pa, pyarrow.parquet as pq
d = sys.argv[1]; n = 0
for f in sorted(glob.glob(f"{d}/meta/**/*.parquet", recursive=True)) + sorted(glob.glob(f"{d}/data/**/*.parquet", recursive=True)):
    t = pq.read_table(f)
    if not (t.schema.metadata or any(fl.metadata for fl in t.schema)): continue
    t2 = pa.Table.from_arrays([t.column(i) for i in range(t.num_columns)], schema=pa.schema([pa.field(fl.name, fl.type, fl.nullable) for fl in t.schema]))
    pq.write_table(t2, f + ".tmp", compression="snappy"); os.replace(f + ".tmp", f); n += 1
print(f"  剥离元数据: {n} 个 parquet")
PYEOF
}
# 从 v3.0 meta/episodes 的 episode_success 列导出边车 json
export_sidecar(){
"$PY_SIM" - "$1" "$2" <<'PYEOF'
import sys, glob, json
import pyarrow.parquet as pq
src, out = sys.argv[1], sys.argv[2]; recs = []
for f in sorted(glob.glob(f"{src}/meta/episodes/**/*.parquet", recursive=True)):
    t = pq.read_table(f, columns=["episode_index", "episode_success"])
    recs += [{"episode_index": int(i), "success": s == "success"} for i, s in zip(t.column(0).to_pylist(), t.column(1).to_pylist())]
recs.sort(key=lambda r: r["episode_index"])
json.dump({"labels_field": "episode_success", "saved_episode_count": len(recs), "episodes": recs}, open(out, "w"), indent=2)
print(f"  边车: {len(recs)} 集, 成功 {sum(r['success'] for r in recs)}")
PYEOF
}
# v3.0 -> v2.1(原地转换 <parent>/<name>),自动剥元数据、保留边车、删除 <name>_v3.0 备份
to_v21(){ local parent=$1 name=$2 sidecar=$3
    strip_meta "$parent/$name"
    "$PY_SIM" "$REPO/scripts/convert_lerobot3.0_to_2.1.py" --repo-id "$name" --root "$parent" || return 1
    rm -rf "$parent/${name}_v3.0"; [ -n "$sidecar" ] && cp "$sidecar" "$parent/$name/episode_success.json"; return 0
}
# lerobot merge:root 参数是数据集目录本身,多数据集必须用 HF_LEROBOT_HOME;用 pandas 2 的 evo 环境
merge_v30(){ local home=$1 out=$2; shift 2; local ids="['$(IFS=,; echo "$*" | sed "s/,/', '/g")']"
    HF_LEROBOT_HOME="$home" PYTHONPATH="$EVO_SRC" "$PY_EVO" -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id "$out" --push_to_hub false --operation.type merge --operation.repo_ids "$ids"
}
# 按 cmdline 子串找进程:排除当前 shell 的祖先链(工具/终端包装 shell 也带同样的命令文本)、
# 与当前 shell 命令行完全相同的进程(子 shell/命令替换),只返回仍存在的 PID
pids_by_cmd(){ local pat=$1 own p a anc=" "; own=$(tr -d '\0' < /proc/$$/cmdline 2>/dev/null)
    a=$$; while [ "$a" -gt 1 ]; do anc="$anc$a "; a=$(ps -o ppid= -p "$a" 2>/dev/null | tr -d ' '); [ -n "$a" ] || break; done
    for p in $(pgrep -f -- "$pat"); do [ -d /proc/$p ] || continue
        case "$anc" in *" $p "*) continue;; esac
        [ "$(tr -d '\0' < /proc/$p/cmdline 2>/dev/null)" = "$own" ] && continue; echo "$p"; done; }
# 脱离会话启动长任务(会话重启也杀不到):detach <logfile> <cmd...>
detach(){ local logf=$1; shift; setsid nohup "$@" > "$logf" 2>&1 < /dev/null & disown; echo "$!"; }
