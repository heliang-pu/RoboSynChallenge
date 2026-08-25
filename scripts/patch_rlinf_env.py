#!/usr/bin/env python
"""把 RoboSynChallenge 的 VLA 环境注册进 RLinf。

RLinf 用 ``rlinf/envs/__init__.py`` 里的 ``SupportedEnvType`` 枚举 + ``get_env_cls()``
的 if/elif 链来解析环境类,没有留插件注册口。所以要用自己的环境类就必须改它那个文件。

这个脚本做两件事,都是幂等的:

1. 往 ``SupportedEnvType`` 加一个 ``ROBOSYNCHALLENGE = "robosynchallenge"``
2. 往 ``get_env_cls()`` 加一个分支,返回
   ``robosynchallenge.rlinf_env.RoboSynChallengeVLAEnv``

真正的实现留在本仓库里(``robosynchallenge/rlinf_env/``),RLinf 那边只多两处几行的
挂钩,升级 RLinf 后重跑一次这个脚本即可。``--revert`` 可以撤销。

用法::

    python scripts/patch_rlinf_env.py --rlinf-root /home/phl/workspace/RLinf
    python scripts/patch_rlinf_env.py --check      # 只报告状态,不写
    python scripts/patch_rlinf_env.py --revert
"""

from __future__ import annotations

import argparse
import pathlib
import sys

DEFAULT_RLINF_ROOT = pathlib.Path("/home/phl/workspace/RLinf")
TARGET_RELPATH = "rlinf/envs/__init__.py"
DATACONFIG_RELPATH = "rlinf/models/embodiment/openpi/dataconfig/__init__.py"

MARKER = "# >>> robosynchallenge patch"
END_MARKER = "# <<< robosynchallenge patch"

# actor worker 解析 openpi 的 config_name 时不会 import robosynchallenge,
# 所以注册必须挂在 RLinf 自己的配置表模块里,而不是靠我们的包被顺带导入。
DATACONFIG_ANCHOR = "_CONFIGS_DICT = {config.name: config for config in _CONFIGS}\n"
DATACONFIG_PATCH = (
    DATACONFIG_ANCHOR
    + f"\n{MARKER} (dataconfig)\n"
    "# RoboSynChallenge 的 openpi 配置。它用 EmbodiChainInputs 而不是 AlohaInputs——\n"
    "# 后者会做 ALOHA 特有的关节翻转,和 RoboSynChallenge 的 SFT checkpoint 对不上。\n"
    "try:\n"
    "    from robosynchallenge.rlinf_env.dataconfig import register as _rsc_register\n"
    "\n"
    "    _rsc_register()\n"
    "except Exception:  # robosynchallenge 不在 PYTHONPATH 时静默跳过,不影响其他环境\n"
    "    pass\n"
    f"{END_MARKER} (dataconfig)\n"
)

ENUM_ANCHOR = '    EMBODICHAIN = "embodichain"\n'
ENUM_PATCH = (
    ENUM_ANCHOR
    + f"    {MARKER} (enum)\n"
    '    ROBOSYNCHALLENGE = "robosynchallenge"\n'
    f"    {END_MARKER} (enum)\n"
)

BRANCH_ANCHOR = """    elif env_type == SupportedEnvType.EMBODICHAIN:
        from rlinf.envs.embodichain.embodichain_env import EmbodiChainEnv

        return EmbodiChainEnv
"""
BRANCH_PATCH = (
    BRANCH_ANCHOR
    + f"    {MARKER} (branch)\n"
    "    elif env_type == SupportedEnvType.ROBOSYNCHALLENGE:\n"
    "        from robosynchallenge.rlinf_env import RoboSynChallengeVLAEnv\n"
    "\n"
    "        return RoboSynChallengeVLAEnv\n"
    f"    {END_MARKER} (branch)\n"
)


def read(path: pathlib.Path) -> str:
    if not path.is_file():
        raise SystemExit(f"找不到 RLinf 的环境注册文件: {path}\n用 --rlinf-root 指定 RLinf 仓库位置。")
    return path.read_text(encoding="utf-8")


def apply_patch(text: str) -> str:
    if MARKER in text:
        return text
    for anchor, patched, what in (
        (ENUM_ANCHOR, ENUM_PATCH, "SupportedEnvType 枚举"),
        (BRANCH_ANCHOR, BRANCH_PATCH, "get_env_cls 分支"),
    ):
        if anchor not in text:
            raise SystemExit(
                f"在 RLinf 里找不到{what}的锚点,大概是上游改了结构。\n"
                f"需要手工挂钩,锚点原文应为:\n{anchor}"
            )
        text = text.replace(anchor, patched, 1)
    return text


def apply_dataconfig_patch(text: str) -> str:
    if MARKER in text:
        return text
    if DATACONFIG_ANCHOR not in text:
        raise SystemExit(
            "在 RLinf 的 openpi 配置表里找不到 _CONFIGS_DICT 的构造语句,大概是上游改了结构。\n"
            f"锚点原文应为:\n{DATACONFIG_ANCHOR}"
        )
    return text.replace(DATACONFIG_ANCHOR, DATACONFIG_PATCH, 1)


def revert_patch(text: str) -> str:
    if MARKER not in text:
        return text
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        if MARKER in line:
            skipping = True
            continue
        if END_MARKER in line:
            skipping = False
            continue
        if not skipping:
            out.append(line)
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rlinf-root", default=str(DEFAULT_RLINF_ROOT))
    ap.add_argument("--check", action="store_true", help="只报告是否已打补丁")
    ap.add_argument("--revert", action="store_true", help="撤销补丁")
    args = ap.parse_args()

    root = pathlib.Path(args.rlinf_root)
    targets = [
        (root / TARGET_RELPATH, apply_patch, "env 类注册"),
        (root / DATACONFIG_RELPATH, apply_dataconfig_patch, "openpi 配置注册"),
    ]

    if args.check:
        all_patched = True
        for path, _, what in targets:
            ok = MARKER in read(path)
            all_patched &= ok
            print(f"[{what}] {path}: {'已打补丁' if ok else '未打补丁'}")
        sys.exit(0 if all_patched else 1)

    for path, patcher, what in targets:
        text = read(path)
        new_text = revert_patch(text) if args.revert else patcher(text)
        if new_text == text:
            print(f"[{what}] 无需改动({'本来就没打过' if args.revert else '已经打过了'})")
            continue
        path.write_text(new_text, encoding="utf-8")
        print(f"[{what}] {'已撤销' if args.revert else '已写入'}: {path}")

    if not args.revert:
        print('\n现在 yaml 里可以写:')
        print('  env.train.env_type: "robosynchallenge"')
        print('  actor.model.openpi.config_name: "pi05_robosynchallenge"')


if __name__ == "__main__":
    main()
