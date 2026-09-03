"""Convert a RoboSyn/OpenPI Pi0.5 Orbax checkpoint for realtime-vla."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

import transformers

from policy.pi05.realtime_vla.tokenizer_adapter import SentencePieceAutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REALTIME_VLA = REPO_ROOT.parent / "realtime-vla"
DEFAULT_TOKENIZER = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-path", type=Path, required=True, help="OpenPI checkpoint step directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination converted .pkl")
    parser.add_argument("--prompt", default="click the bell", help="Fallback prompt stored in the checkpoint")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(os.environ.get("OPENPI_TOKENIZER_PATH", DEFAULT_TOKENIZER)),
    )
    parser.add_argument(
        "--realtime-vla-dir",
        type=Path,
        default=Path(os.environ.get("REALTIME_VLA_DIR", DEFAULT_REALTIME_VLA)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = args.realtime_vla_dir.expanduser().resolve() / "convert_from_jax_pi05.py"
    if not converter.is_file():
        raise FileNotFoundError(
            f"realtime-vla converter not found at {converter}. Clone https://github.com/dexmal/realtime-vla first."
        )
    if not (args.jax_path / "params").is_dir():
        raise FileNotFoundError(f"Orbax params directory not found below {args.jax_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    original_auto_tokenizer = transformers.AutoTokenizer
    original_argv = sys.argv
    try:
        # The upstream converter only needs AutoTokenizer.from_pretrained.  Patch
        # that narrow surface so it can consume OpenPI's tokenizer.model directly.
        transformers.AutoTokenizer = SentencePieceAutoTokenizer
        sys.argv = [
            str(converter),
            "--jax_path",
            str(args.jax_path.resolve()),
            "--output",
            str(args.output.resolve()),
            "--prompt",
            args.prompt,
            "--tokenizer_path",
            str(args.tokenizer_path.expanduser().resolve()),
        ]
        runpy.run_path(str(converter), run_name="__main__")
    finally:
        transformers.AutoTokenizer = original_auto_tokenizer
        sys.argv = original_argv


if __name__ == "__main__":
    main()

