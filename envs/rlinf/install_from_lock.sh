#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# 用锁定版本复现 RLinf(PPO/GRPO 后训练)的 uv 环境。
#
#   bash envs/rlinf/install_from_lock.sh
#
# 与 RLinf 官方 install.sh 的区别:官方脚本是"解析一次装什么算什么",本机装出来的
# 环境经历了 rlinf-openpi 把 torch 2.11 降到 2.7.1、lerobot 0.4.4 降到 0.3.3、
# 换掉 12 个包等一系列副作用,最终能跑的是 envs/rlinf/requirements.lock.txt 这个状态。
# 本脚本直接按 lock 装,一步到位,不再重演那些降级。
#
# 前置:uv、能访问 PyPI 与 DexForce 私有源(dexsim-engine 等只在那里)、
#       ~/workspace/RLinf 与 ~/workspace/EmbodiChain 已 clone(位置可用环境变量覆盖)。
#
# 覆盖点:RLINF_ROOT / EMBODICHAIN_ROOT / VENV_DIR / PYTHON_VERSION
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOSYN_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCK="$SCRIPT_DIR/requirements.lock.txt"

RLINF_ROOT="${RLINF_ROOT:-$HOME/workspace/RLinf}"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$HOME/workspace/EmbodiChain}"
VENV_DIR="${VENV_DIR:-$RLINF_ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.14}"
# dexsim-engine / embodichain 的依赖只在 DexForce 的私有源上
DEXFORCE_INDEX="${DEXFORCE_INDEX:-http://pyp.open3dv.site:2345/simple/}"

die() { echo "错误: $*" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || die "找不到 uv(https://docs.astral.sh/uv/)"
[[ -f "$LOCK" ]] || die "找不到 lock: $LOCK"
[[ -d "$RLINF_ROOT/rlinf" ]] || die "找不到 RLinf: $RLINF_ROOT(先 git clone https://github.com/RLinf/RLinf)"
[[ -d "$EMBODICHAIN_ROOT/embodichain" ]] || die "找不到 EmbodiChain: $EMBODICHAIN_ROOT"
[[ -d "$EMBODICHAIN_ROOT/embodichain_tasks" ]] || die "$EMBODICHAIN_ROOT 下没有 embodichain_tasks(需要 >=0.2.4 布局)"

echo "=== 1/6 建 venv(Python $PYTHON_VERSION)==="
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
PY="$VENV_DIR/bin/python"

echo
echo "=== 2/6 按 lock 装依赖(仅 PyPI)==="
# 主 lock 里全是 PyPI 包(torch 是 +cu126 的 PyPI 默认轮子),不给私有源:
# 否则 uv 会对 444 个包逐个去敲那个时常超时的私有源,整体卡死在第一个超时上。
#
# --no-deps 是必须的:freeze 本身就是完整闭包,不需要再解析;而且这个环境的元数据
# 自相矛盾——cmeel-boost==1.90.0 声明 numpy>=2.0,rlinf-openpi 却把 numpy 压到 1.26.4。
# 运行时没问题(冒烟 PPO 就是这个状态跑通的),但让 resolver 看一眼就会拒绝安装。
uv pip install -r "$LOCK" --no-deps

echo
echo "=== 2b/6 DexForce 私有源独有的包 ==="
# 只有 dexsim-engine(闭源仿真引擎)。依赖已在主 lock 里装好,--no-deps 避免再解析。
# 私有源不稳定,失败时给出明确提示而不是让整个脚本静默死在超时里。
DEXFORCE_HOST="$(echo "$DEXFORCE_INDEX" | sed -E 's#https?://([^/:]+).*#\1#')"
uv pip install -r "$SCRIPT_DIR/requirements.dexforce.txt" --no-deps \
    --index-url "$DEXFORCE_INDEX" --trusted-host "$DEXFORCE_HOST" \
    || die "从 DexForce 私有源装 dexsim-engine 失败(源不可达?)。确认能访问 $DEXFORCE_INDEX 后重跑;
其余包已装好,重跑会直接跳到这一步。"

echo
echo "=== 3/6 EmbodiChain 本地 editable(替代私有源 wheel)==="
# --no-deps:依赖已在 lock 里;embodichain_tasks 是同仓库里的独立包,必须单独装,
# 否则外层同名目录会作为命名空间包劫持 import。
uv pip install -e "$EMBODICHAIN_ROOT" --no-deps
uv pip install -e "$EMBODICHAIN_ROOT/embodichain_tasks" --no-deps

echo
echo "=== 4/6 EmbodiChain 并行 rollout 补丁 ==="
PATCH="$ROBOSYN_PATH/patches/embodichain_parallel_envs.patch"
if git -C "$EMBODICHAIN_ROOT" apply --check --reverse "$PATCH" >/dev/null 2>&1; then
    echo "已打过,跳过"
elif git -C "$EMBODICHAIN_ROOT" apply --check "$PATCH" >/dev/null 2>&1; then
    git -C "$EMBODICHAIN_ROOT" apply "$PATCH" && echo "已应用 $PATCH"
else
    die "补丁对不上当前 EmbodiChain(版本变了?)。见 patches/README.md 手工处理。"
fi

echo
echo "=== 5/6 robosynchallenge 可 import + RLinf 挂钩 ==="
SITE_PACKAGES="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROBOSYN_PATH" > "$SITE_PACKAGES/robosynchallenge.pth"
"$PY" "$ROBOSYN_PATH/scripts/patch_rlinf_env.py" --rlinf-root "$RLINF_ROOT"

echo
echo "=== 6/6 校验 ==="
RLINF_ROOT="$RLINF_ROOT" "$PY" - <<'PYEOF'
import importlib, os, sys
sys.path.insert(0, os.environ["RLINF_ROOT"])
bad = 0
for mod in ["ray", "torch", "openpi.models_pytorch.pi0_pytorch",
            "embodichain.lab.gym.utils.registration", "embodichain_tasks.tableware",
            "robosynchallenge", "robosynchallenge.rlinf_env", "rlinf.envs"]:
    try:
        importlib.import_module(mod); print(f"  OK   {mod}")
    except Exception as e:
        bad += 1; print(f"  FAIL {mod}: {type(e).__name__}: {e}")
try:
    from rlinf.envs import get_env_cls
    print("  OK   get_env_cls('robosynchallenge') ->", get_env_cls("robosynchallenge").__name__)
    from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
    assert "pi05_robosynchallenge" in _CONFIGS_DICT; print("  OK   pi05_robosynchallenge 已注册")
except Exception as e:
    bad += 1; print(f"  FAIL 挂钩: {type(e).__name__}: {e}")
import torch; print(f"  torch {torch.__version__} cuda {torch.version.cuda}")
sys.exit(1 if bad else 0)
PYEOF

echo
echo "环境就绪。接下来:"
echo "  bash launch/rlinf_train.sh ppo --dry-run"
