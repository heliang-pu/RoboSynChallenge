"""Put the RoboSynChallenge repo root on sys.path for standalone CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = REPO_ROOT / "policy"


def add_repo_root() -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def resolve_path(value) -> Path:
    """Resolve a config path: absolute stays put, relative is repo-root relative.

    Config files ship repo-relative paths so they work no matter which directory
    the training or evaluation command was launched from.
    """
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()
