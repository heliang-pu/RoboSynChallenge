#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# 在 RLinf 官方 install.sh 之后,把环境补成"openpi 模型 + EmbodiChain 仿真"的组合。
#
# 先跑官方那步(建 venv + embodied 依赖 + embodichain wheel):
#
#   cd $RLINF_ROOT
#   bash requirements/install.sh embodied --env embodichain \
#        --venv $RLINF_ROOT/.venv --no-root
#
# 再跑这个脚本:
#
#   bash launch/rlinf_setup_env.sh
#
# 为什么需要它:RLinf 的 install_openpi_model 只覆盖 behavior / libero / metaworld /
# calvin / robocasa / robotwin / isaaclab / roboverse / polaris —— 没有 embodichain。
# 这不是疏漏,是因为 RLinf 的 EmbodiChain 支持本来就只到 CartPole + MLP,从没和 VLA
# 组合过。所以模型侧要照 install_openpi_model 里 metaworld 分支的做法手工补齐。
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOSYN_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROBOSYN_PATH/.." && pwd)"

RLINF_ROOT="${RLINF_ROOT:-$HOME/workspace/RLinf}"
VENV_DIR="${VENV_DIR:-$RLINF_ROOT/.venv}"
RLINF_OPENPI_VERSION="${RLINF_OPENPI_VERSION:-0.1.1}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"

if [[ -z "${EMBODICHAIN_ROOT:-}" ]]; then
    for cand in "$WORKSPACE_ROOT/EmbodiChain" "$HOME/workspace/EmbodiChain"; do
        [[ -d "$cand/embodichain" ]] && { EMBODICHAIN_ROOT="$cand"; break; }
    done
fi

die() { echo "错误: $*" >&2; exit 1; }

[[ -x "$VENV_DIR/bin/python" ]] || die "找不到 RLinf 的 venv: $VENV_DIR
先跑官方安装(见本文件头部注释)。"
[[ -n "${EMBODICHAIN_ROOT:-}" && -d "$EMBODICHAIN_ROOT/embodichain" ]] \
    || die "找不到 EmbodiChain 工作副本(用 EMBODICHAIN_ROOT= 指定)"

command -v uv >/dev/null 2>&1 || die "找不到 uv"

# uv pip 通过 VIRTUAL_ENV 认目标环境
export VIRTUAL_ENV="$VENV_DIR"
PY="$VENV_DIR/bin/python"

echo "=== 1/4 装 openpi 模型侧 (rlinf-openpi==$RLINF_OPENPI_VERSION) ==="
uv pip install "rlinf-openpi==$RLINF_OPENPI_VERSION"

echo
echo "=== 2/4 flash-attn ==="
if [[ "$SKIP_FLASH_ATTN" == "1" ]]; then
    echo "SKIP_FLASH_ATTN=1,跳过。"
else
    # 源码编译很慢且需要 nvcc;失败不阻断——没有 flash-attn 也能跑,只是慢些。
    uv pip install flash-attn --no-build-isolation || \
        echo "警告: flash-attn 装失败,继续。需要时设 SKIP_FLASH_ATTN=1 显式跳过。"
fi

echo
echo "=== 3/4 把 embodichain 换成本地 editable ==="
# 官方安装装的是 DexForce 私有源的 embodichain wheel,而 robosynchallenge 的任务代码
# 是针对本地这个工作副本写的(本地 __version__ 报 0.2.3 但已是 0.2.4 的布局:
# registration.py 里有 build_env / discover_task_packages / execute_init_hooks,
# 且带 embodichain_tasks 包)。不换的话 build_env 拿到的是 wheel 里的引擎,
# 任务代码和它可能对不上。
# --no-deps: 依赖已由 wheel 那步装齐,重新解析只会引入无谓的版本变动。
uv pip install -e "$EMBODICHAIN_ROOT" --no-deps

# embodichain_tasks 是同一个仓库里的**独立**包(EmbodiChain/embodichain_tasks/ 有自己的
# pyproject.toml),不会被上面那条带进去。不装它的话:PYTHONPATH 上的
# EmbodiChain/embodichain_tasks 这个外层目录会作为命名空间包劫持掉真正的
# embodichain_tasks/embodichain_tasks/,于是 `from embodichain_tasks.tableware.base_agent_env
# import BaseAgentEnv` 报 ModuleNotFoundError —— 而 robosynchallenge 的多个任务都 import 它。
uv pip install -e "$EMBODICHAIN_ROOT/embodichain_tasks" --no-deps

echo "embodichain       -> $("$PY" -c 'import embodichain; print(embodichain.__file__)')"
echo "embodichain_tasks -> $("$PY" -c 'import embodichain_tasks.tableware as m; print(m.__file__)')"

echo
echo "=== 4/4 让 robosynchallenge 可被 import ==="
# 刻意不用 pip install -e:本仓库的 requirements.txt 混进了 175 条 /opt/ros/jazzy
# 的条目,在没有那套 ROS 的机器上依赖解析会直接失败。任务只需要能 import
# (@register_env 在 import 时注册),所以用 .pth 把路径加进 site-packages——
# 这样 Ray 起的每个 worker 进程都能找到,不依赖 PYTHONPATH 传递。
SITE_PACKAGES="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROBOSYN_PATH" > "$SITE_PACKAGES/robosynchallenge.pth"
echo "已写入 $SITE_PACKAGES/robosynchallenge.pth -> $ROBOSYN_PATH"

echo
echo "=== 校验 ==="
# RLinf 本身不作为包安装 —— run_embodiment.sh 是靠 PYTHONPATH=$REPO_PATH 用源码的。
# 校验必须照做,否则 import rlinf 会失败,看起来像环境坏了。
RLINF_ROOT="$RLINF_ROOT" "$PY" - <<'PYEOF'
import importlib, os, sys

sys.path.insert(0, os.environ["RLINF_ROOT"])

failures = []
for mod, note in [
    ("ray", "Ray 编排"),
    ("torch", "训练后端"),
    ("openpi.models_pytorch.pi0_pytorch", "openpi 的 PyTorch pi0.5"),
    ("embodichain.lab.gym.utils.registration", "EmbodiChain 环境注册"),
    ("robosynchallenge", "RoboSynChallenge 任务"),
    ("robosynchallenge.rlinf_env", "本仓库的接入层"),
    ("rlinf.envs", "RLinf 环境注册表"),
]:
    try:
        importlib.import_module(mod)
        print(f"  OK   {mod:<48} {note}")
    except Exception as exc:
        failures.append((mod, exc))
        print(f"  FAIL {mod:<48} {type(exc).__name__}: {exc}")

try:
    from rlinf.envs import get_env_cls
    print(f"  OK   get_env_cls('robosynchallenge') -> {get_env_cls('robosynchallenge').__name__}")
except Exception as exc:
    failures.append(("get_env_cls", exc))
    print(f"  FAIL get_env_cls('robosynchallenge'): {type(exc).__name__}: {exc}")

try:
    from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
    ok = "pi05_robosynchallenge" in _CONFIGS_DICT
    print(f"  {'OK  ' if ok else 'FAIL'} openpi 配置 pi05_robosynchallenge 已注册: {ok}")
    if not ok:
        failures.append(("dataconfig", "未注册"))
except Exception as exc:
    failures.append(("dataconfig", exc))
    print(f"  FAIL openpi 配置表: {type(exc).__name__}: {exc}")

sys.exit(1 if failures else 0)
PYEOF

echo
echo "环境就绪。接下来:"
echo "  bash launch/rlinf_bench_envs.sh          # 实测吞吐,定 total_num_envs"
echo "  bash launch/rlinf_train.sh ppo --dry-run"
