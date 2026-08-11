#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# XR-1 (Xiaomi-Robotics-1) 评测环境安装
#
#   bash policy/xr1/setup_env.sh [--with-flash-attn] [--python 3.11]
#
# 在 policy/xr1/.venv 下建一个独立 venv，装 xr1(mibot) 包及其依赖。
# flash-attn 编译经常失败，默认不装；推理侧会自动退回 sdpa。
# ----------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
XR1_PKG_DIR="$SCRIPT_DIR/Xiaomi-Robotics-1/xr1"

PYTHON_VERSION="3.11"
WITH_FLASH_ATTN=0
SKIP_DEEPSPEED=0
SKIP_SIM=0
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
# 比赛仓库根部的 .venv：pytorch_kinematics 定制版的来源，见下方仿真栈段落
BASELINE_VENV="${BASELINE_VENV:-$REPO_ROOT/.venv}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-flash-attn) WITH_FLASH_ATTN=1; shift ;;
        --skip-deepspeed) SKIP_DEEPSPEED=1; shift ;;
        --skip-sim) SKIP_SIM=1; shift ;;
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# ssh 会话里残留的 127.0.0.1:7897 代理是坏的，会让所有 pip 请求超时。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-$UV_DEFAULT_INDEX}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
    UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" ]]; then
    echo "Error: 找不到 uv，请先安装 (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

if [[ ! -d "$XR1_PKG_DIR" ]]; then
    echo "Error: 找不到 XR-1 源码目录 $XR1_PKG_DIR" >&2
    echo "       请先 git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 到 policy/xr1/" >&2
    exit 1
fi

# deepspeed 的 setup.py 在构建期就要跑 nvcc 拿 CUDA 版本，没有 CUDA_HOME 直接报
# MissingCUDAException。这台机器没装系统级 CUDA，但 conda 环境里有可用的 toolkit。
detect_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        echo "$CUDA_HOME"; return
    fi
    local candidate
    for candidate in /usr/local/cuda /usr/local/cuda-*; do
        [[ -x "$candidate/bin/nvcc" ]] && { echo "$candidate"; return; }
    done
    for candidate in "$HOME"/miniconda3/envs/*/ "$HOME"/anaconda3/envs/*/; do
        [[ -x "${candidate}bin/nvcc" ]] && { echo "${candidate%/}"; return; }
    done
}

DETECTED_CUDA_HOME="$(detect_cuda_home)"
if [[ -n "$DETECTED_CUDA_HOME" ]]; then
    export CUDA_HOME="$DETECTED_CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
elif [[ $SKIP_DEEPSPEED -eq 0 ]]; then
    echo "!!! 找不到 CUDA toolkit (nvcc)，deepspeed 无法构建，自动转为 --skip-deepspeed" >&2
    echo "!!! 部署/推理不受影响；要做微调请先装 CUDA toolkit 再重跑本脚本" >&2
    SKIP_DEEPSPEED=1
fi

echo "=========================================="
echo "  XR-1 环境安装"
echo "  venv:      $VENV_DIR"
echo "  python:    $PYTHON_VERSION"
echo "  index:     $UV_DEFAULT_INDEX"
echo "  CUDA_HOME: ${CUDA_HOME:-<none>}"
echo "  仿真栈:    $([[ $SKIP_SIM -eq 1 ]] && echo 'skip (eval.sh 将不可用)' || echo "$EMBODICHAIN_ROOT")"
echo "  deepspeed: $([[ $SKIP_DEEPSPEED -eq 1 ]] && echo 'skip (仅推理可用)' || echo yes)"
echo "  flash-attn: $([[ $WITH_FLASH_ATTN -eq 1 ]] && echo yes || echo 'no (推理退回 sdpa)')"
echo "=========================================="

run_uv() { "$UV_BIN" pip install --python "$VENV_DIR/bin/python" "$@"; }

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR" || exit 1
fi

# 1) 构建后端 + torch 必须先落地：deepspeed 的 setup.py 在构建期就要 import torch，
#    所以后面所有安装都走 --no-build-isolation。
echo ">>> [1/4] 构建工具链"
run_uv setuptools wheel packaging ninja pybind11 || exit 1

echo ">>> [2/4] torch 2.8.0 / torchvision 0.23.0"
run_uv torch==2.8.0 torchvision==0.23.0 || exit 1

echo ">>> [3/4] xr1 (mibot) 包及其依赖"
if [[ $SKIP_DEEPSPEED -eq 1 ]]; then
    REQ_FILE="$(mktemp)"
    grep -v -i '^deepspeed' "$XR1_PKG_DIR/assets/requirements.txt" > "$REQ_FILE"
    run_uv --no-build-isolation -r "$REQ_FILE" || { rm -f "$REQ_FILE"; exit 1; }
    rm -f "$REQ_FILE"
    # 依赖已手动装好，装包体本身时跳过依赖解析（否则又会拉 deepspeed）
    run_uv --no-build-isolation --no-deps -e "$XR1_PKG_DIR" || exit 1
else
    run_uv --no-build-isolation -e "$XR1_PKG_DIR" || exit 1
fi
# 转换器需要读 LeRobot 的 parquet；上游 requirements 里没有。
run_uv --no-build-isolation pandas pyarrow || exit 1

echo ">>> [4/5] EmbodiChain 仿真栈"
# scripts/eval_policy.py 是在**同一个解释器**里同时 import 策略栈和仿真栈的，
# 所以这个 venv 必须两边都有，只装 XR-1 是跑不起来评测的。
if [[ $SKIP_SIM -eq 1 ]]; then
    echo "    跳过 (--skip-sim)；注意这样 eval.sh 会在 import gymnasium 处直接挂掉"
elif [[ ! -d "$EMBODICHAIN_ROOT" ]]; then
    echo "!!! 找不到 EmbodiChain ($EMBODICHAIN_ROOT)，跳过仿真栈" >&2
    echo "!!! eval.sh 将无法运行；可用 EMBODICHAIN_ROOT=<path> 指定后重跑" >&2
else
    # 约束文件：仿真栈的依赖解析不许动这几个包，否则 XR-1 那边直接废掉
    CONSTRAINTS="$(mktemp)"
    cat > "$CONSTRAINTS" <<'EOF'
torch==2.8.0
torchvision==0.23.0
transformers==4.57.1
numpy==2.1.3
warp-lang==1.14.0
EOF

    run_uv --no-deps --no-build-isolation -e "$EMBODICHAIN_ROOT" || exit 1
    run_uv --no-deps --no-build-isolation -e "$EMBODICHAIN_ROOT/embodichain_tasks" || exit 1

    # dexsim 只在私有源上。不同版本的 uv 对 HTTP(非 HTTPS) 源处理不一致
    # （pro6000 的 snap uv 会报 "not found in registry" 即使源可达），
    # 所以装不上就从基准 venv 整包复制——反正基准 venv 里那份版本必然与
    # 它自己的 EmbodiChain 配套。
    if ! "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
            --index-url http://pyp.open3dv.site:2345/simple/ \
            --allow-insecure-host pyp.open3dv.site --no-deps dexsim_engine==0.4.3 2>/dev/null \
       && ! "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
            --index-url http://pyp.open3dv.site:2345/simple/ --no-deps dexsim_engine==0.4.3 2>/dev/null; then
        echo "    私有源装 dexsim 失败，改从基准 venv 复制"
        DEXSIM_SP="$(echo "$BASELINE_VENV"/lib/python*/site-packages)"
        TARGET_SP="$(echo "$VENV_DIR"/lib/python*/site-packages)"
        if [[ -d "$DEXSIM_SP/dexsim" ]]; then
            rm -rf "$TARGET_SP/dexsim" "$TARGET_SP"/dexsim_engine-*.dist-info
            cp -r "$DEXSIM_SP/dexsim" "$TARGET_SP/" || exit 1
            cp -r "$DEXSIM_SP"/dexsim_engine-*.dist-info "$TARGET_SP/" 2>/dev/null
        else
            echo "!!! 基准 venv 里也没有 dexsim，无法继续" >&2
            exit 1
        fi
    fi

    # 这一组必须锁版本，理由见下方注释；一律 --no-deps 免得解析器乱动
    #   warp-lang 1.14.0  —— 1.16.0 会让 dexsim 的 _patched_findsource 无限递归，
    #                        表现为 import dexsim 直接 SIGSEGV(139)，且没有任何报错
    #   gymnasium 0.29.1  —— EmbodiChain 只写 >=0.29.1，但 1.x 有 API 变更，
    #                        对齐到 pi05 已跑通的版本
    #   toppra/polars     —— EmbodiChain pyproject 里的显式 pin，
    #                        因为我们 --no-deps 装的它，得手动落实
    run_uv --no-deps \
        warp-lang==1.14.0 newton==1.1.0 newton-actuators==0.1.1 \
        mujoco==3.6.0 mujoco-warp==3.6.0 \
        gymnasium==0.29.1 trimesh==4.12.2 toppra==0.6.3 polars==1.31.0 || exit 1

    # 其余运行时依赖，带依赖装但受上面的 constraint 约束
    "$UV_BIN" pip install --python "$VENV_DIR/bin/python" --constraint "$CONSTRAINTS" \
        open3d tensordict imageio coacd usd-core==26.5 h5py prettytable av || exit 1
    rm -f "$CONSTRAINTS"

    # pytorch_kinematics 必须从基准 venv 整包复制，**不能** pip 装。
    # EmbodiChain 的 solver 要调 Chain.forward_kinematics_tensor()，
    # 而这个方法只存在于基准 venv 里那份打过补丁的构建（chain.py:507）。
    # PyPI 上的 0.10.0 和 EmbodiChain 警告里提到的 0.7.6 都没有它，
    # 装了会在 solver 初始化时报 AttributeError。
    # 坑在于打补丁那份的版本号也写作 0.10.0，光看版本号分辨不出来。
    SITE_PACKAGES="$(echo "$VENV_DIR"/lib/python*/site-packages)"
    BASELINE_SP="$(echo "$BASELINE_VENV"/lib/python*/site-packages)"
    if [[ -d "$BASELINE_SP/pytorch_kinematics" ]]; then
        echo "    从基准 venv 复制 pytorch_kinematics (定制版)"
        rm -rf "$SITE_PACKAGES/pytorch_kinematics" "$SITE_PACKAGES"/pytorch_kinematics-*.dist-info
        cp -r "$BASELINE_SP/pytorch_kinematics" "$SITE_PACKAGES/" || exit 1
        cp -r "$BASELINE_SP"/pytorch_kinematics-*.dist-info "$SITE_PACKAGES/" 2>/dev/null
    else
        echo "!!! 基准 venv ($BASELINE_VENV) 里没有 pytorch_kinematics" >&2
        echo "!!! EmbodiChain 的 solver 会因缺 forward_kinematics_tensor 挂掉" >&2
        echo "!!! 可用 BASELINE_VENV=<path> 指定一个装好定制版的 venv 后重跑" >&2
    fi
fi

echo ">>> [5/5] flash-attn (可选)"
if [[ $WITH_FLASH_ATTN -eq 1 ]]; then
    if ! run_uv --no-build-isolation flash-attn; then
        echo "!!! flash-attn 安装失败，忽略；推理时用 attn_implementation=sdpa" >&2
    fi
else
    echo "    跳过 (--with-flash-attn 可开启)"
fi

echo ""
echo ">>> 校验"
"$VENV_DIR/bin/python" - <<'PY'
import importlib, sys

print("python      :", sys.version.split()[0])

import torch
print("torch       :", torch.__version__)

import transformers
print("transformers:", transformers.__version__)
assert transformers.__version__ == "4.57.1", "XR-1 要求 transformers==4.57.1"

import mibot
from mibot.utils.io import ACTION_DIM, STATE_DIM, ACTION_PARTS, compose_state
print("mibot       : ok  (ACTION_DIM=%d STATE_DIM=%d)" % (ACTION_DIM, STATE_DIM))

# 真正的推理导入路径（建模块要 liger-kernel + transformers，不需要 GPU）
from mibot.models import MIMODEL
import mibot.models.VLA.XR1  # noqa: F401
print("mibot.models.VLA.XR1: ok")

for name in ("pandas", "pyarrow", "decord", "mmengine", "deepspeed", "lightning"):
    try:
        m = importlib.import_module(name)
        print(f"{name:<12}: {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{name:<12}: FAILED ({exc})")

try:
    import flash_attn
    print("flash_attn  :", flash_attn.__version__)
except Exception:
    print("flash_attn  : 未安装 -> 推理使用 sdpa")

# 仿真栈：eval_policy 要在同一进程里 import 它们
print("--- 仿真栈 ---")
try:
    import warp
    print("warp-lang   :", warp.config.version)
    assert warp.config.version.startswith("1.14.0"), "warp 必须是 1.14.0，1.16.0 会让 dexsim 爆栈"
    for name in ("gymnasium", "dexsim", "embodichain", "newton", "mujoco"):
        module = importlib.import_module(name)
        print(f"{name:<12}: {getattr(module, '__version__', 'ok')}")

    # 定制版 pytorch_kinematics：版本号和 PyPI 版一样是 0.10.0，
    # 只能靠这个方法在不在来分辨
    import pytorch_kinematics
    has_fk = hasattr(pytorch_kinematics.chain.Chain, "forward_kinematics_tensor")
    print(f"{'pytorch_kin':<12}: {'定制版 ok' if has_fk else '**PyPI 版，缺 forward_kinematics_tensor**'}")
    assert has_fk, "pytorch_kinematics 不是定制版，EmbodiChain 的 solver 会挂"
except Exception as exc:
    print(f"仿真栈不可用: {exc}  -> eval.sh 跑不了（用 --skip-sim 时属预期）")
PY

echo ""
echo "完成。评测: bash policy/xr1/eval.sh <task> <setting> <train_config> <model_name> <gpu>"
