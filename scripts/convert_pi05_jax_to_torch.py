#!/usr/bin/env python
"""把 openpi 的 JAX orbax checkpoint 转成 RLinf 能吃的 PyTorch safetensors。

包装 RLinf 的 rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py，绕开两个坑:

1. **openpi 包被遮蔽**: 转换器同目录下有个也叫 ``openpi`` 的包
   (``rlinf/utils/ckpt_convertor/openpi/``)。直接 ``python <path>/convert_...py`` 运行时
   ``sys.path[0]`` 是脚本目录，那个包会把真正的 openpi 顶掉，报
   ``ModuleNotFoundError: No module named 'openpi.models'``。这里用 importlib 按文件路径
   加载模块，不往 sys.path 里塞脚本目录。

2. **assets 拷不到**: 转换器从 ``checkpoint_dir.parent/assets`` 找 norm_stats，
   但 openpi 训练产物的布局是 ``<step>/assets``(与 ``<step>/params`` 同级)，
   parent 是 ``<step>`` 的上一层，没有 assets。转换后这里补拷。

用法::

    policy/pi05/.venv/bin/python scripts/convert_pi05_jax_to_torch.py \\
        --checkpoint-dir policy/pi05/checkpoints/pi05_base_robosynchallenge_full/mixer_operating/28000 \\
        --config-name pi05_base_robosynchallenge_full \\
        --output-path /home/phl/workspace/models/pi05_pt/mixer_operating_28000

必须用 pi05 那个 venv 的解释器 —— 转换器要 import openpi。
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import shutil
import sys

DEFAULT_RLINF_ROOT = pathlib.Path("/home/phl/workspace/RLinf")
CONVERTER_RELPATH = "rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py"


def load_converter(rlinf_root: pathlib.Path):
    """按文件路径加载 RLinf 转换器，不把它的目录加进 sys.path。"""
    path = rlinf_root / CONVERTER_RELPATH
    if not path.is_file():
        raise SystemExit(f"找不到 RLinf 转换器: {path}\n用 --rlinf-root 指定 RLinf 仓库位置。")

    spec = importlib.util.spec_from_file_location("_rlinf_jax_to_torch", path)
    module = importlib.util.module_from_spec(spec)
    # exec_module 不改 sys.path，所以模块里的 `import openpi.models.gemma`
    # 走的是当前解释器的 openpi(policy/pi05 的 editable 安装)，不会被同名目录遮蔽。
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="指到 <step> 那一层(里面有 params/ 和 assets/)")
    ap.add_argument("--config-name", required=True, help="openpi TrainConfig 名，如 pi05_base_robosynchallenge_full")
    ap.add_argument("--output-path", help="输出目录；--inspect-only 时可省")
    ap.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--rlinf-root", default=str(DEFAULT_RLINF_ROOT))
    ap.add_argument("--inspect-only", action="store_true", help="只打印参数 key，不转换")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.checkpoint_dir).resolve()
    if not (ckpt / "params").is_dir():
        raise SystemExit(f"{ckpt} 下没有 params/ —— --checkpoint-dir 要指到 <step> 那一层，不是 params 本身。")

    # 转换器靠 checkpoint_dir 字符串里有没有 "pi05" 来决定走 adaptive-norm(Dense_0)
    # 还是标准 RMSNorm(scale)。这个判断很脆，路径不对会静默转出一个错的模型。
    if "pi05" not in str(ckpt):
        raise SystemExit(
            f"路径里不含 'pi05': {ckpt}\n"
            "转换器会误走 pi0 的 RMSNorm 分支，转出来的权重是错的。\n"
            "把 checkpoint 放到路径含 pi05 的位置，或改用 --checkpoint-dir 传一个含 pi05 的等价路径。"
        )

    conv = load_converter(pathlib.Path(args.rlinf_root))
    conv.main(
        checkpoint_dir=str(ckpt),
        config_name=args.config_name,
        output_path=args.output_path,
        precision=args.precision,
        inspect_only=args.inspect_only,
    )

    if args.inspect_only:
        return

    # 补拷 assets(norm_stats)——转换器找的是 parent/assets，对不上 openpi 的布局。
    src_assets = ckpt / "assets"
    dst_assets = pathlib.Path(args.output_path) / "assets"
    if not src_assets.is_dir():
        print(f"警告: {src_assets} 不存在，没有 norm_stats 可拷", file=sys.stderr)
    elif dst_assets.exists():
        print(f"assets 已存在，跳过: {dst_assets}")
    else:
        shutil.copytree(src_assets, dst_assets)
        print(f"补拷 assets: {src_assets} -> {dst_assets}")


if __name__ == "__main__":
    main()
