#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# G0.5 (GalaxeaVLA) 评估环境安装脚本
#
#   bash policy/g05/setup_env.sh
#
# 用 GalaxeaVLA 官方方式（uv sync）在 policy/g05/GalaxeaVLA/.venv 建虚拟环境。
# python 3.10.16 / torch 2.7.1+cu128，依赖版本完全由官方 uv.lock 锁定。
#
# 关键点（实测，勿改）：
#   1. 必须 unset 代理。download.pytorch.org 直连 65MB/s，走 127.0.0.1:7897
#      代理只有 2.3MB/s，torch 单个 wheel 就 1.1GB，走代理要几十分钟。
#   2. 用 --frozen，保证绝不重写官方 uv.lock（官方 lock 的默认 registry 已经是
#      mirrors.aliyun.com/pypi，本身就是国内源，不需要再换）。
#   3. --index-strategy unsafe-best-match 是官方 README 指定的参数，
#      torch 系列从 pytorch-cu128 索引取，其余从默认索引取。
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G05_ROOT="$SCRIPT_DIR/GalaxeaVLA"
VENV_DIR="$G05_ROOT/.venv"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -d "$G05_ROOT" ]]; then
    echo "Error: 找不到 GalaxeaVLA 源码目录: $G05_ROOT" >&2
    echo "请先执行: git clone https://github.com/OpenGalaxea/GalaxeaVLA.git $G05_ROOT" >&2
    exit 1
fi

if [[ ! -x "$UV_BIN" ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    else
        echo "Error: 找不到 uv。安装: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        exit 1
    fi
fi

# --- 1. 清掉残留代理 ---------------------------------------------------------
unset http_proxy https_proxy all_proxy ftp_proxy 2>/dev/null || true
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY 2>/dev/null || true

# 大 wheel 下载慢时不要被默认超时掐断
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
# uv 把 venv 建在项目内（官方约定）
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

echo "========================================="
echo "  G0.5 (GalaxeaVLA) 环境安装"
echo "  uv:      $UV_BIN ($("$UV_BIN" --version))"
echo "  项目:    $G05_ROOT"
echo "  venv:    $VENV_DIR"
echo "  代理:    已 unset（直连）"
echo "========================================="

cd "$G05_ROOT"

# --- 2. uv sync ---------------------------------------------------------------
# --frozen: 只用现有 uv.lock，绝不重新求解、绝不改写官方 lock 文件。
SYNC_ARGS=(sync --frozen --index-strategy unsafe-best-match)
if [[ "${G05_SETUP_EXTRA_DEV:-0}" == "1" ]]; then
    SYNC_ARGS+=(--extra dev)
fi

echo "[1/3] uv ${SYNC_ARGS[*]}"
# 刻意不做"失败就回退到无 --frozen 的 sync"：那样会重写官方 uv.lock，
# 而且被 Ctrl-C / kill 打断也会误触发。宁可报错停下，让人来判断。
if ! "$UV_BIN" "${SYNC_ARGS[@]}"; then
    cat >&2 <<'ERREOF'

[error] uv sync --frozen 失败。

常见原因与处理:
  1. 被中断（Ctrl-C / kill）——直接重跑本脚本即可，wheel 已缓存，会快很多。
  2. 同时有另一个 uv 在跑，抢 ~/.cache/uv/.lock 卡住
     —— 先 `pgrep -af "uv sync"` 确认只有一个再重跑。
  3. uv.lock 与 pyproject.toml 真的不一致
     —— 这是上游仓库的问题，需要人工决定是否重新 lock。
        重新 lock 会改写官方文件 uv.lock，脚本不会替你做这个决定。
        确认要做再手动执行:
          cd <GalaxeaVLA> && uv sync --index-strategy unsafe-best-match
ERREOF
    exit 1
fi

PY="$VENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "Error: venv 未建成，找不到 $PY" >&2
    exit 1
fi

# --- 2b. 补 torchcodec 缺的原生依赖 ---------------------------------------------
# torchcodec 的 FFmpeg 后端要 libnppicc.so.12（属于 nvidia-npp-cu12），
# 但这个包不在官方 uv.lock 里，不补的话训练读视频会炸：
#   RuntimeError: Could not load libtorchcodec ... libnppicc.so.12: cannot open shared object file
# 评估不受影响（观测来自仿真器，不解码视频），但训练必须要。
# 用 uv pip 装进 venv，不碰官方 uv.lock；索引显式指到 aliyun——
# uv pip 不读 uv.lock，默认会去 pypi.org，国内直连极慢。
NPP_LIB_DIR="$VENV_DIR/lib/python3.10/site-packages/nvidia/npp/lib"
if [[ ! -f "$NPP_LIB_DIR/libnppicc.so.12" ]]; then
    echo "[1b/3] 补装 nvidia-npp-cu12（torchcodec 依赖）"
    UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}" \
        "$UV_BIN" pip install --python "$PY" nvidia-npp-cu12
fi
# 这个目录不在动态库搜索路径上，必须显式加进 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$NPP_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- 3. 核心 import 自检（不碰权重、不占 GPU）-----------------------------------
echo "[2/3] 核心依赖自检"
"$PY" - <<'PYEOF'
import importlib
import sys

print(f"python  {sys.version.split()[0]}")

import torch
print(f"torch   {torch.__version__}  cuda_built={torch.version.cuda}  is_available={torch.cuda.is_available()}")

for mod in (
    "torchvision", "transformers", "hydra", "omegaconf", "numpy",
    "einops", "accelerate", "cv2", "PIL", "scipy", "zarr",
    "vector_quantize_pytorch", "fla",
):
    m = importlib.import_module(mod)
    print(f"  ok  {mod:26s} {getattr(m, '__version__', '(no __version__)')}")
PYEOF

# --- 2c. EmbodiChain 仿真栈 ------------------------------------------------------
# scripts/eval_policy.py 是**单进程**跑的：同一个解释器里既要有 G0.5 策略栈，
# 也要有 EmbodiChain 仿真栈。GalaxeaVLA 的 uv.lock 里没有这些，必须补装。
# 一律用 uv pip（不进 uv.lock）+ --no-deps（保住锁定的 torch 2.7.1 等核心版本）。
#
# 顺序很重要：`uv sync --frozen` 会**删掉**所有不在 uv.lock 里的包，
# 所以仿真栈必须装在 uv sync 之后。反过来说，装完之后别再单独跑 uv sync，
# 否则整套仿真栈会被清掉（重跑本脚本可恢复）。
REPO_ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$(cd "$REPO_ROOT_DIR/.." && pwd)/EmbodiChain}"
DEXSIM_INDEX="${DEXSIM_INDEX:-http://pyp.open3dv.site:2345/simple/}"

# uv pip 不读 uv.lock，默认索引是 pypi.org，国内直连极慢 —— 显式指到清华源。
UVPIP_INDEX="${UVPIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
uvpip() { UV_DEFAULT_INDEX="$UVPIP_INDEX" "$UV_BIN" pip install --python "$PY" "$@"; }

if [[ "${G05_SKIP_SIM_STACK:-0}" != "1" ]]; then
    echo "[1c/3] 安装 EmbodiChain 仿真栈"
    if [[ ! -d "$EMBODICHAIN_ROOT" ]]; then
        echo "Error: 找不到 EmbodiChain: $EMBODICHAIN_ROOT" >&2
        echo "       它必须与 RoboSynChallenge 同级；或用 EMBODICHAIN_ROOT 指定。" >&2
        exit 1
    fi

    # embodichain / embodichain_tasks：editable + --no-deps
    # （它们的依赖声明会拖动 torch，必须 --no-deps）
    uvpip --no-deps -e "$EMBODICHAIN_ROOT" -e "$EMBODICHAIN_ROOT/embodichain_tasks"

    # dexsim_engine 走私有源。它会顺带换掉 coacd/mujoco/open3d/trimesh —— 这几个
    # 是 GalaxeaVLA 自带 GalaxeaManipSim 用的，我们走 EmbodiChain，不受影响。
    # 注意必须把默认索引整个换成私有源：uvpip 平时把 UV_DEFAULT_INDEX 指向清华，
    # 那个环境变量会盖掉 --index-url，导致 "dexsim-engine was not found"。
    DEXSIM_HOST="$(echo "$DEXSIM_INDEX" | sed -E 's#https?://([^:/]+).*#\1#')"
    UVPIP_INDEX="$DEXSIM_INDEX" uvpip --trusted-host "$DEXSIM_HOST" "dexsim_engine==0.4.3"

    # eval_policy 导入链上缺的包，版本对齐已跑通端到端的 policy/{motus,xr1} venv：
    #   tensordict 0.13.0  —— 与 torch 2.7.1 配套（robosynchallenge.managers.actions 要）
    #   warp-lang  1.14.0  —— dexsim 0.4.3 只兼容这个版本，1.16 会段错误
    #   polars     1.31.0  —— embodichain 的 OPW solver 要；新版 1.43 装出来缺二进制
    uvpip --no-deps "tensordict==0.13.0" pyvers orjson cloudpickle \
        "polars==1.31.0" "warp-lang==1.14.0" janus

    # pytorch_kinematics：GalaxeaVLA 钉的是 pytorch_kinematics_ms==0.7.3，它提供的
    # 模块名同样叫 pytorch_kinematics，但**缺 forward_kinematics_tensor**，EmbodiChain
    # 会在运行时炸。PyPI 上的 0.10.0 也缺，必须用仓库根 .venv 里的补丁版。
    # GalaxeaVLA 自己的 src/ 从不 import pytorch_kinematics，替换是安全的。
    PK_SRC="$(ls -d "$REPO_ROOT_DIR"/.venv/lib/python3*/site-packages/pytorch_kinematics 2>/dev/null | head -1)"
    PK_DST_DIR="$VENV_DIR/lib/python3.10/site-packages"
    if [[ -d "$PK_SRC" ]]; then
        if ! "$PY" -c "import pytorch_kinematics as pk; import sys; sys.exit(0 if hasattr(pk.chain.Chain,'forward_kinematics_tensor') else 1)" 2>/dev/null; then
            echo "  替换为补丁版 pytorch_kinematics: $PK_SRC"
            rm -rf "$PK_DST_DIR/pytorch_kinematics" "$PK_DST_DIR"/pytorch_kinematics-0.10.0.dist-info
            cp -r "$PK_SRC" "$PK_DST_DIR/"
            PK_INFO="$(ls -d "$REPO_ROOT_DIR"/.venv/lib/python3*/site-packages/pytorch_kinematics-0.10.0.dist-info 2>/dev/null | head -1)"
            [[ -d "$PK_INFO" ]] && cp -r "$PK_INFO" "$PK_DST_DIR/"
        fi
    else
        echo "  [warn] 找不到仓库根 .venv 里的补丁版 pytorch_kinematics，" >&2
        echo "         EmbodiChain 可能在运行时报缺 forward_kinematics_tensor。" >&2
    fi
fi

echo "[2b/3] torchcodec 视频解码自检"
"$PY" - <<'PYEOF'
# 训练要靠 torchcodec 读 mp4；比赛数据是 av1 编码，这里确认原生库能加载。
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    print("  ok  torchcodec 原生库加载成功（av1 可解）")
except Exception as exc:
    print(f"  !!  torchcodec 加载失败: {str(exc).splitlines()[0]}")
    print("      评估不受影响，但训练读视频会失败。检查 nvidia-npp-cu12 与 LD_LIBRARY_PATH。")
PYEOF

if [[ "${G05_SKIP_SIM_STACK:-0}" != "1" ]]; then
    echo "[2c/3] 仿真栈自检（策略栈 + 仿真栈同进程共存）"
    PYTHONPATH="$EMBODICHAIN_ROOT:$REPO_ROOT_DIR:$REPO_ROOT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" - <<'PYEOF'
import torch
import pytorch_kinematics as pk
import polars, warp, tensordict
print(f"  ok  torch {torch.__version__}（核心版本未被仿真栈污染）")
print(f"  ok  tensordict {tensordict.__version__} / polars {polars.__version__} / warp {warp.config.version}")
assert hasattr(pk.chain.Chain, "forward_kinematics_tensor"), \
    "pytorch_kinematics 缺 forward_kinematics_tensor（装成 PyPI 版了，需要仓库根 .venv 的补丁版）"
print("  ok  pytorch_kinematics 补丁版（有 forward_kinematics_tensor）")
import eval_policy  # noqa: F401  —— 比赛评估脚本，导入成功即说明仿真栈齐了
print("  ok  scripts/eval_policy.py 可导入（embodichain / dexsim 就绪）")
PYEOF
fi

echo "[3/3] g05 包 import 自检"
"$PY" - <<'PYEOF'
# 这几个是 deploy_policy 进程内推理链路上真正会用到的符号，
# 能 import 成功就说明源码 + 依赖装对了（不需要权重）。
from g05.models.g05.inferencer import PolicyInferencer
from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.data_processor.transforms.action_filter import BaseActionFilter
from g05.utils.checkpoint.checkpoint_utils import load_state_dict_safely
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.data.processor_utils import build_processors

print("  ok  g05.models.g05.inferencer.PolicyInferencer")
print("  ok  g05.data_processor.processor.mixture_processor.MixtureProcessor")
print("  ok  g05.utils.data.processor_utils.build_processors")
PYEOF

cat <<EOF

=========================================
  安装完成
=========================================
  解释器: $PY

  冒烟测试（不需要权重）:
    $PY $SCRIPT_DIR/smoke_test.py

  权重（OpenGalaxea/G05 是 gated 仓库，必须先在 HF 网页同意协议，
        且只能走官方源 + token —— hf-mirror 拿不到 gated 仓库）:
    huggingface-cli download OpenGalaxea/G05 --repo-type model \\
        --local-dir /home/phl/workspace/models/g05 \\
        --include "g05-base/*" "qwen3_5_2b_base_processor/*" "action_tokenizer.pt"

  然后软链进 GalaxeaVLA（配置里的路径都是相对项目根的 checkpoints/...）:
    ln -sfn /home/phl/workspace/models/g05/action_tokenizer.pt \\
            $SCRIPT_DIR/GalaxeaVLA/checkpoints/action_tokenizer.pt
    ln -sfn /home/phl/workspace/models/g05/g05-base \\
            $SCRIPT_DIR/GalaxeaVLA/checkpoints/g05-base

  跑评估（注意 g05-base 是预训练权重，不能零样本部署，详见 README_INTEGRATION.md 第 8.1 节）:
    bash policy/g05/eval.sh <task_name> <setting> <ckpt_path> <gpu_id>
EOF
