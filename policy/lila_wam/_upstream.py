"""Import bridge to the pinned upstream LiLa-WAM checkout.

The upstream repository exposes ``models`` / ``dataloader`` / ``utils`` as
top-level packages.  Putting ``policy/lila_wam/LiLa-WAM`` on ``sys.path`` would
therefore shadow any module of the same (very generic) name in the evaluation
process, so instead we mount the checkout under a private ``lila_upstream``
namespace: ``lila_upstream.models.model_runner`` and friends.  Relative imports
inside upstream keep working because they resolve within that namespace.

Only ``models/`` is reachable this way. Upstream's top-level scripts
(``robotwin_infer.py``, ``train.py``, ...) import ``models.*`` absolutely, which
would resolve against ``sys.path`` rather than the namespace, so this adapter
reimplements the few helpers it needs from them instead.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parent / "LiLa-WAM"
NAMESPACE = "lila_upstream"


def ensure_upstream() -> Path:
    """Mount the upstream checkout under the ``lila_upstream`` namespace."""
    if NAMESPACE not in sys.modules:
        if not (UPSTREAM_ROOT / "models" / "vla_model_fm.py").exists():
            raise FileNotFoundError(
                f"upstream LiLa-WAM checkout is missing at {UPSTREAM_ROOT}.\n"
                f"Run: git submodule update --init policy/lila_wam/LiLa-WAM"
            )
        package = types.ModuleType(NAMESPACE)
        package.__path__ = [str(UPSTREAM_ROOT)]
        sys.modules[NAMESPACE] = package
    return UPSTREAM_ROOT


def upstream_models():
    """Return ``(ModelFactory, VLAWrapper, calc_flow_matching_loss)``."""
    ensure_upstream()
    from lila_upstream.models.model_runner import ModelFactory, VLAWrapper
    from lila_upstream.models.vla_model_fm import calc_flow_matching_loss

    return ModelFactory, VLAWrapper, calc_flow_matching_loss


def upstream_collate_fn():
    ensure_upstream()
    from lila_upstream.dataloader.dataset import collate_fn

    return collate_fn

