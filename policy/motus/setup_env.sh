#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Motus policy environment for RoboSynChallenge
#
#   bash policy/motus/setup_env.sh                # inference + simulator stack
#   bash policy/motus/setup_env.sh --with-train   # + tensorboard/wandb (training box)
#   bash policy/motus/setup_env.sh --no-sim       # skip EmbodiChain (conversion only)
#
# Creates policy/motus/.venv (CPython 3.10, torch 2.7.1+cu128) following the
# upstream Motus install recipe, then adds the EmbodiChain simulation stack so
# that eval.sh can run scripts/eval_policy.py inside this same interpreter.
#
# Two non-obvious things this script handles (both discovered the hard way):
#   * deepspeed is an INFERENCE dependency, not just a training one — Motus'
#     utils/common.py does `import deepspeed.comm.comm` at module scope.
#   * deepspeed ships sdist-only and its setup.py *and* its runtime import both
#     shell out to `$CUDA_HOME/bin/nvcc -V` for a version string. This box has
#     no CUDA toolkit, so we install a version-only nvcc shim. With
#     DS_BUILD_OPS=0 no CUDA op is ever compiled, so the shim is sufficient.
# ----------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
PY_VERSION="3.10"
INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
DEXSIM_INDEX="http://pyp.open3dv.site:2345/simple/"

WITH_TRAIN=0
WITH_SIM=1
for arg in "$@"; do
    case "$arg" in
        --with-train) WITH_TRAIN=1 ;;
        --no-sim)     WITH_SIM=0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# The inherited proxy on this host is stale and breaks pip. Always drop it.
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY

UV="${UV_BIN:-$HOME/.local/bin/uv}"
if ! command -v "$UV" >/dev/null 2>&1; then
    UV="$(command -v uv || true)"
fi
if [[ -z "$UV" ]]; then
    echo "Error: uv not found (expected ~/.local/bin/uv)." >&2
    exit 1
fi

echo "=== [1/7] create venv (python $PY_VERSION) at $VENV_DIR ==="
"$UV" venv --python "$PY_VERSION" "$VENV_DIR" || exit 1
PY="$VENV_DIR/bin/python"

pip_install() {
    "$UV" pip install --python "$PY" --index-url "$INDEX" "$@"
}

echo "=== [2/7] torch 2.7.1 + torchvision 0.22.1 (cu128) ==="
"$UV" pip install --python "$PY" --index-url "$TORCH_INDEX" \
    torch==2.7.1 torchvision==0.22.1 || exit 1

echo "=== [3/7] Motus runtime dependencies ==="
# numpy pinned <2 per Motus inference requirements (bak/wan uses legacy aliases).
pip_install \
    "numpy>=1.23.5,<2" \
    "opencv-python>=4.9.0.80" \
    "Pillow>=10.0.0" \
    "diffusers>=0.31.0" \
    "accelerate>=1.1.1" \
    "safetensors" \
    "einops==0.8.1" \
    "easydict" \
    "ftfy>=6.3.1" \
    "omegaconf==2.3.0" \
    "PyYAML" \
    "tqdm" \
    "imageio" \
    "imageio-ffmpeg" \
    "matplotlib>=3.7.0" \
    "scipy>=1.11.0" \
    "huggingface-hub>=0.20.0" \
    "pyarrow" \
    "pandas" \
    "av" || exit 1

# Qwen3-VL needs transformers >= 4.57.  Upstream pins the 5.0 release
# candidate; fall back to the last 4.x that still ships Qwen3VL.
echo "--- transformers (Qwen3-VL capable) ---"
if ! pip_install --prerelease=allow "transformers==5.0.0rc0"; then
    echo "!!! transformers==5.0.0rc0 unavailable, falling back to 4.57.x"
    pip_install "transformers>=4.57,<5" || exit 1
fi
pip_install "tokenizers>=0.20.3" "qwen-vl-utils>=0.0.11" || true

echo "=== [4/7] deepspeed + nvcc shim (REQUIRED for inference) ==="
CUDA_VER="$("$PY" -c 'import torch;print(torch.version.cuda or "12.8")' 2>/dev/null || echo 12.8)"
SHIM_HOME="$VENV_DIR/.cuda-shim"
if command -v nvcc >/dev/null 2>&1; then
    echo "--- real nvcc found, no shim needed ---"
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
else
    mkdir -p "$SHIM_HOME/bin"
    cat > "$SHIM_HOME/bin/nvcc" <<EOF
#!/usr/bin/env bash
# Version-only nvcc shim for deepspeed (no compilation capability).
# Created by policy/motus/setup_env.sh; motus_model._ensure_cuda_home() selects it.
cat <<'BANNER'
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release ${CUDA_VER}, V${CUDA_VER}.0
BANNER
EOF
    chmod +x "$SHIM_HOME/bin/nvcc"
    export CUDA_HOME="$SHIM_HOME"
    echo "--- nvcc shim at $SHIM_HOME reporting CUDA $CUDA_VER ---"
fi
export DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1
"$UV" pip install --python "$PY" --index-url "$INDEX" --no-build-isolation \
    "deepspeed>=0.18.3" || exit 1

if [[ "$WITH_SIM" == "1" ]]; then
    echo "=== [5/7] EmbodiChain simulation stack ==="
    if [[ ! -d "$EMBODICHAIN_ROOT" ]]; then
        echo "!!! EmbodiChain not found at $EMBODICHAIN_ROOT; skipping sim stack." >&2
        echo "!!! Set EMBODICHAIN_ROOT or pass --no-sim." >&2
    else
        # dexsim_engine lives on a private index and drags in the sim deps.
        "$UV" pip install --python "$PY" --index-url "$DEXSIM_INDEX" \
            --allow-insecure-host pyp.open3dv.site "dexsim_engine==0.4.3" || exit 1

        # embodichain / embodichain_tasks editable, --no-deps so they cannot
        # move the torch / numpy / transformers pins set above.
        "$UV" pip install --python "$PY" --index-url "$INDEX" \
            --no-deps --no-build-isolation \
            -e "$EMBODICHAIN_ROOT" -e "$EMBODICHAIN_ROOT/embodichain_tasks" || exit 1

        # Modules eval_policy pulls in transitively that dexsim doesn't provide.
        pip_install --no-deps \
            "gymnasium==0.29.1" "farama-notifications" \
            "tensordict" "cloudpickle" "h5py" "prettytable" || exit 1

        # Pin the simulator core to the versions in policy/pi05/.venv, which is
        # the known-good reference. dexsim's resolver pulls newer ones that are
        # untested here (and its polars wheel arrives without its binary).
        pip_install --no-deps \
            "polars==1.31.0" "mujoco==3.6.0" "mujoco-warp==3.6.0" \
            "newton==1.1.0" "newton-actuators==0.1.1" "warp-lang==1.14.0" \
            "trimesh==4.12.2" "usd-core==26.5" "toppra==0.6.3" "pyvers==0.2.2" || exit 1
        # NOTE: pi05 pins scikit-learn 1.9.0, which requires Python >= 3.11.
        # Motus needs 3.10, so we stay on the 1.7.x line here.

        # pytorch_kinematics: the PyPI 0.10.0 wheel is MISSING
        # Chain.forward_kinematics_tensor, which EmbodiChain calls. The workspace
        # baseline venv ships a customised 0.10.0 that has it. The package is pure
        # Python (no .so), so copying it across 3.11 -> 3.10 is safe.
        BASELINE_VENV="${BASELINE_VENV:-$REPO_ROOT/.venv}"
        BASELINE_SP="$(ls -d "$BASELINE_VENV"/lib/python*/site-packages 2>/dev/null | head -1)"
        TARGET_SP="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
        if [[ -n "$BASELINE_SP" && -d "$BASELINE_SP/pytorch_kinematics" ]]; then
            echo "--- copying customised pytorch_kinematics from $BASELINE_SP ---"
            rm -rf "$TARGET_SP/pytorch_kinematics" "$TARGET_SP"/pytorch_kinematics-*.dist-info
            cp -r "$BASELINE_SP/pytorch_kinematics" "$TARGET_SP/"
            cp -r "$BASELINE_SP"/pytorch_kinematics-*.dist-info "$TARGET_SP/" 2>/dev/null || true
        else
            echo "!!! Baseline venv not found at $BASELINE_VENV; falling back to PyPI." >&2
            echo "!!! The PyPI build lacks Chain.forward_kinematics_tensor and WILL break" >&2
            echo "!!! EmbodiChain at runtime. Set BASELINE_VENV to a venv that has the" >&2
            echo "!!! customised build and re-run." >&2
            pip_install --no-deps "pytorch_kinematics==0.10.0" || true
        fi
        if ! "$PY" -c 'from pytorch_kinematics.chain import Chain; assert hasattr(Chain,"forward_kinematics_tensor")' 2>/dev/null; then
            echo "!!! pytorch_kinematics is MISSING Chain.forward_kinematics_tensor." >&2
            echo "!!! Copy it from a known-good venv: " >&2
            echo "!!!   cp -r <baseline>/lib/python*/site-packages/pytorch_kinematics $TARGET_SP/" >&2
        else
            echo "--- pytorch_kinematics: forward_kinematics_tensor present ---"
        fi
    fi
else
    echo "=== [5/7] simulation stack skipped (--no-sim) ==="
fi

if [[ "$WITH_TRAIN" == "1" ]]; then
    echo "=== [6/7] training extras ==="
    pip_install "tensorboard>=2.15.0" "wandb>=0.16.0" "seaborn>=0.12.0" || exit 1
else
    echo "=== [6/7] training extras skipped (pass --with-train) ==="
fi

echo "=== [7/7] flash-attn (optional, source build — failure is tolerated) ==="
if [[ "${MOTUS_SKIP_FLASH_ATTN:-0}" == "1" ]]; then
    echo "--- skipped (MOTUS_SKIP_FLASH_ATTN=1) ---"
elif pip_install --no-build-isolation "flash-attn"; then
    echo "--- flash-attn installed ---"
else
    echo "!!! flash-attn NOT installed."
    echo "!!! This is expected and NOT fatal: Motus falls back to"
    echo "!!! torch.nn.functional.scaled_dot_product_attention (slower, same results)."
fi

echo "=== verify ==="
EMBODICHAIN_ROOT="$EMBODICHAIN_ROOT" REPO_ROOT="$REPO_ROOT" WITH_SIM="$WITH_SIM" "$PY" - <<'PYEOF'
import importlib, os, sys
ok = True
print("python  :", sys.version.split()[0])

# Select the nvcc shim before anything can import torch.utils.cpp_extension,
# which caches CUDA_HOME at first import (deepspeed reads that cached value).
sys.path.insert(0, os.environ["REPO_ROOT"] + "/policy/motus")
from motus_model import _ensure_cuda_home
_ensure_cuda_home()
print("CUDA_HOME:", os.environ.get("CUDA_HOME", "<unset>"))

import torch
print("torch   :", torch.__version__, "cuda build:", torch.version.cuda)
import transformers
print("transformers:", transformers.__version__)
try:
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    print("Qwen3VL : available")
except Exception as e:
    ok = False
    print("Qwen3VL : MISSING ->", e)
for mod in ("cv2", "numpy", "diffusers", "einops", "easydict", "ftfy",
            "omegaconf", "safetensors", "PIL", "imageio", "matplotlib",
            "yaml", "pyarrow", "av", "deepspeed"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:12s}: {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        ok = False
        print(f"{mod:12s}: MISSING -> {e}")
try:
    import flash_attn
    print("flash_attn  :", flash_attn.__version__)
except Exception:
    print("flash_attn  : absent (sdpa fallback will be used)")

# Motus' own modules (this is what exercises the deepspeed/CUDA_HOME path).
try:
    from motus_model import _import_motus
    mm, _, _ = _import_motus()
    print("motus core  : ok (MotusConfig fields:",
          len(mm.MotusConfig.__dataclass_fields__), ")")
except Exception as e:
    ok = False
    print("motus core  : FAILED ->", type(e).__name__, e)

if os.environ.get("WITH_SIM") == "1":
    sys.path.insert(0, os.environ["EMBODICHAIN_ROOT"])
    sys.path.insert(0, os.environ["REPO_ROOT"])
    sys.path.insert(0, os.environ["REPO_ROOT"] + "/scripts")
    os.chdir(os.environ["REPO_ROOT"])
    for mod in ("gymnasium", "dexsim", "embodichain", "embodichain_tasks",
                "robosynchallenge", "polars", "mujoco", "trimesh"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod:16s}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            ok = False
            print(f"{mod:16s}: MISSING -> {e}")
    try:
        from pytorch_kinematics.chain import Chain
        if hasattr(Chain, "forward_kinematics_tensor"):
            print("pytorch_kinematics: ok (forward_kinematics_tensor present)")
        else:
            ok = False
            print("pytorch_kinematics: WRONG BUILD -> no forward_kinematics_tensor "
                  "(copy it from the baseline venv, see setup_env.sh)")
    except Exception as e:
        ok = False
        print("pytorch_kinematics: MISSING ->", e)
    try:
        import eval_policy  # noqa: F401
        print("eval_policy     : IMPORT OK")
    except Exception as e:
        ok = False
        print("eval_policy     : FAILED ->", type(e).__name__, e)

sys.exit(0 if ok else 1)
PYEOF
RC=$?

echo
if [[ $RC -eq 0 ]]; then
    echo "=== Motus venv ready: $VENV_DIR ==="
else
    echo "=== Motus venv INCOMPLETE (see MISSING/FAILED above) ===" >&2
fi
exit $RC
