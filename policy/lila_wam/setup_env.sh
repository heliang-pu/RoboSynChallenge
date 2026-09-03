#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# LiLa-WAM 环境搭建(RoboSynChallenge)
#
#   bash policy/lila_wam/setup_env.sh                      # 训练环境
#   bash policy/lila_wam/setup_env.sh --download-encoder   # + 下载 DINOv3 权重
#   bash policy/lila_wam/setup_env.sh --with-sim           # + EmbodiChain 评测栈
#
# 在 policy/lila_wam/.venv 建独立环境(CPython 3.11)。上游 README 写的是
# conda + python 3.10,这里改用 uv + 3.11,理由:评测要和 EmbodiChain 跑在同一个
# 解释器里,而仿真栈的 pin(scikit-learn 1.9)要求 >= 3.11。LiLa-WAM 本身没有
# 3.10 专属依赖。
#
# DINOv3 权重在 HuggingFace 上是 gated 的:先在
# https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m 同意协议,
# 再用 `hf auth login` 或 HF_TOKEN 提供 token,--download-encoder 才能拉下来。
# ----------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
PY_VERSION="${LILA_PY_VERSION:-3.11}"
INDEX="${LILA_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX="${LILA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${LILA_TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${LILA_TORCHVISION_VERSION:-0.22.1}"
DEXSIM_INDEX="http://pyp.open3dv.site:2345/simple/"

ENCODER_REPO="${LILA_ENCODER_REPO:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
ENCODER_DIR="$SCRIPT_DIR/dinov3/$(basename "$ENCODER_REPO")"

WITH_SIM=0
DOWNLOAD_ENCODER=0
for arg in "$@"; do
    case "$arg" in
        --with-sim)          WITH_SIM=1 ;;
        --download-encoder)  DOWNLOAD_ENCODER=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# 继承下来的代理会让 pip 卡死,所以装包时一律不走代理;
# 但 huggingface.co 在这类机器上往往**只能**走代理,所以先存下来,
# 下权重的时候再放回去(--download-encoder 那一步)。
SAVED_HTTPS_PROXY="${https_proxy:-${HTTPS_PROXY:-}}"
SAVED_HTTP_PROXY="${http_proxy:-${HTTP_PROXY:-}}"
SAVED_ALL_PROXY="${all_proxy:-${ALL_PROXY:-}}"
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY

# huggingface_hub >= 1.0 走 httpx,而 httpx 只认 socks5:// / socks5h://,
# 不认某些代理客户端写出来的裸 socks://。统一改写一下。
normalize_proxy() { sed -E 's#^socks://#socks5://#' <<<"${1:-}"; }

with_proxy() {
    # 在子 shell 里恢复代理执行,不污染后续的 pip 调用
    (
        local https_p http_p all_p
        https_p="$(normalize_proxy "$SAVED_HTTPS_PROXY")"
        http_p="$(normalize_proxy "$SAVED_HTTP_PROXY")"
        all_p="$(normalize_proxy "$SAVED_ALL_PROXY")"
        [[ -n "$https_p" ]] && export https_proxy="$https_p" HTTPS_PROXY="$https_p"
        [[ -n "$http_p"  ]] && export http_proxy="$http_p"   HTTP_PROXY="$http_p"
        [[ -n "$all_p"   ]] && export all_proxy="$all_p"     ALL_PROXY="$all_p"
        "$@"
    )
}

UV="${UV_BIN:-$HOME/.local/bin/uv}"
command -v "$UV" >/dev/null 2>&1 || UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
    echo "Error: 找不到 uv(装一下:curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
    exit 1
fi

# --allow-existing 是必须的:本脚本的正常用法就是分多次跑(先装训练环境,
# 之后再补 --with-sim)。uv venv 默认会清空目标目录重建,那会在第二次调用时
# 把已装好的依赖全删掉——如果此时还有训练进程正用着这个 venv,会把它搞崩。
if [[ -x "$VENV_DIR/bin/python" ]]; then
    echo "=== [1/5] 复用已有 venv:$VENV_DIR ==="
    "$UV" venv --python "$PY_VERSION" --allow-existing "$VENV_DIR" || exit 1
else
    echo "=== [1/5] 建 venv(python $PY_VERSION):$VENV_DIR ==="
    "$UV" venv --python "$PY_VERSION" "$VENV_DIR" || exit 1
fi
PY="$VENV_DIR/bin/python"

pip_install() { "$UV" pip install --python "$PY" --index-url "$INDEX" "$@"; }

echo "=== [2/5] torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION (cu128) ==="
"$UV" pip install --python "$PY" --index-url "$TORCH_INDEX" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" || exit 1

echo "=== [3/5] LiLa-WAM 运行依赖 ==="
# transformers 是硬门槛:DINOv3 的建模代码从 4.56 才进 transformers。
# 上游 pin 的是 5.0.0rc0,拿不到就退到最后一个 4.x。
if ! pip_install --prerelease=allow "transformers==5.0.0rc0"; then
    echo "!!! transformers==5.0.0rc0 不可用,回退到 4.57.x"
    pip_install "transformers>=4.57,<5" || exit 1
fi
pip_install \
    "omegaconf==2.3.0" \
    "accelerate>=1.1.1" \
    "safetensors" \
    "h5py" \
    "opencv-python>=4.9.0.80" \
    "Pillow>=10.0.0" \
    "numpy" \
    "pyarrow" \
    "pandas" \
    "av" \
    "scipy>=1.11.0" \
    "matplotlib>=3.7.0" \
    "tqdm" \
    "huggingface-hub>=0.20.0" \
    "socksio" "PySocks" || exit 1
# socksio / PySocks 不是 LiLa-WAM 的依赖,是为了下权重:这类机器上 huggingface.co
# 常常只能走 socks 代理,而 httpx / requests 默认不认 socks:// scheme。

if [[ "$DOWNLOAD_ENCODER" == "1" ]]; then
    echo "=== [4/5] 下载视觉编码器 $ENCODER_REPO -> $ENCODER_DIR ==="
    mkdir -p "$(dirname "$ENCODER_DIR")"
    cat > "$VENV_DIR/.download_encoder.py" <<'PYEOF'
import sys

from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
)
print(f"downloaded to {path}")
PYEOF
    if ! with_proxy "$PY" "$VENV_DIR/.download_encoder.py" "$ENCODER_REPO" "$ENCODER_DIR"; then
        echo "!!! 下载失败。两个常见原因:" >&2
        echo "!!!  1) DINOv3 是 gated 仓库:先到 https://huggingface.co/$ENCODER_REPO" >&2
        echo "!!!     同意协议,再 hf auth login(或 export HF_TOKEN=...)。" >&2
        echo "!!!  2) 连不上 huggingface.co:本脚本装包时会禁用代理,下载这一步会把" >&2
        echo "!!!     调用前的 http(s)_proxy / all_proxy 放回来——如果你的机器要走代理," >&2
        echo "!!!     请确保调用 setup_env.sh 之前这些变量是设好的。" >&2
        exit 1
    fi
else
    echo "=== [4/5] 跳过编码器下载(需要时加 --download-encoder)==="
fi

if [[ "$WITH_SIM" == "1" ]]; then
    echo "=== [5/5] EmbodiChain 评测栈 ==="
    if [[ ! -d "$EMBODICHAIN_ROOT" ]]; then
        echo "!!! 没在 $EMBODICHAIN_ROOT 找到 EmbodiChain,跳过仿真栈。" >&2
        echo "!!! 设置 EMBODICHAIN_ROOT 后重跑。" >&2
    else
        "$UV" pip install --python "$PY" --index-url "$DEXSIM_INDEX" \
            --allow-insecure-host pyp.open3dv.site "dexsim_engine==0.4.3" || exit 1
        # --no-deps:不让它们改动上面钉好的 torch / transformers。
        "$UV" pip install --python "$PY" --index-url "$INDEX" \
            --no-deps --no-build-isolation \
            -e "$EMBODICHAIN_ROOT" -e "$EMBODICHAIN_ROOT/embodichain_tasks" || exit 1
        pip_install --no-deps \
            "gymnasium==0.29.1" "farama-notifications" \
            "tensordict" "cloudpickle" "prettytable" || exit 1
        pip_install --no-deps \
            "polars==1.31.0" "mujoco==3.6.0" "mujoco-warp==3.6.0" \
            "newton==1.1.0" "newton-actuators==0.1.1" "warp-lang==1.14.0" \
            "trimesh==4.12.2" "usd-core==26.5" "toppra==0.6.3" "pyvers==0.2.2" || exit 1

        # PyPI 上的 pytorch_kinematics 0.10.0 缺 Chain.forward_kinematics_tensor,
        # EmbodiChain 运行时要用。仓库根 .venv 里那份是改过的,纯 Python,可直接拷。
        BASELINE_VENV="${BASELINE_VENV:-$REPO_ROOT/.venv}"
        BASELINE_SP="$(ls -d "$BASELINE_VENV"/lib/python*/site-packages 2>/dev/null | head -1)"
        TARGET_SP="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
        if [[ -n "$BASELINE_SP" && -d "$BASELINE_SP/pytorch_kinematics" ]]; then
            rm -rf "$TARGET_SP/pytorch_kinematics" "$TARGET_SP"/pytorch_kinematics-*.dist-info
            cp -r "$BASELINE_SP/pytorch_kinematics" "$TARGET_SP/"
            cp -r "$BASELINE_SP"/pytorch_kinematics-*.dist-info "$TARGET_SP/" 2>/dev/null || true
            echo "--- 已从 $BASELINE_SP 拷贝定制版 pytorch_kinematics ---"
        else
            echo "!!! 没找到基线 venv($BASELINE_VENV),EmbodiChain 运行时可能会挂。" >&2
            pip_install --no-deps "pytorch_kinematics==0.10.0" || true
        fi
    fi
else
    echo "=== [5/5] 跳过仿真栈(评测时加 --with-sim)==="
fi

echo
echo "========================================="
echo "  环境就绪:$VENV_DIR"
echo "  编码器  :$ENCODER_DIR"
echo "  下一步  :bash policy/lila_wam/finetune.sh <dataset_root> <task_name> 0"
echo "========================================="
