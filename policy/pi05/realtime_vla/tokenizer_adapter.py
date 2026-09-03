"""Use OpenPI's native SentencePiece tokenizer with realtime-vla.

realtime-vla currently calls ``AutoTokenizer.from_pretrained`` even though
OpenPI distributes the PaliGemma tokenizer as a single ``.model`` file.  The
small adapter below implements only the tokenizer surface used by
``convert_from_jax_pi05.py`` and ``pi05_infer.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sentencepiece
import torch


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self._processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool | str = False,
        **_: Any,
    ) -> dict[str, Any]:
        texts = [text] if isinstance(text, str) else list(text)
        token_ids = [self._processor.encode(item, add_bos=True) for item in texts]
        if truncation and max_length is not None:
            token_ids = [tokens[:max_length] for tokens in token_ids]
        if padding:
            target = max_length if padding == "max_length" and max_length else max(map(len, token_ids))
            token_ids = [tokens + [0] * (target - len(tokens)) for tokens in token_ids]
        if return_tensors == "pt":
            if len({len(tokens) for tokens in token_ids}) != 1:
                raise ValueError("return_tensors='pt' requires equally sized token sequences")
            return {"input_ids": torch.tensor(token_ids, dtype=torch.long)}
        return {"input_ids": token_ids}


class SentencePieceAutoTokenizer:
    """Drop-in replacement for the subset of AutoTokenizer used upstream."""

    @staticmethod
    def from_pretrained(model_path: str | Path, **_: Any) -> SentencePieceTokenizer:
        return SentencePieceTokenizer(model_path)

