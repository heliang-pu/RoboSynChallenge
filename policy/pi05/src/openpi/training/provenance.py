"""Capture the information needed to reproduce a training run.

The checkpoint formats used by JAX/Orbax and PyTorch primarily store tensors.
This module writes a small, framework-independent provenance bundle next to
those tensors so exported inference checkpoints retain their training context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import enum
import hashlib
import json
import math
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any, Literal

_PROVENANCE_FILES = (
    "train_config.json",
    "launch_command.txt",
    "git_commit.txt",
    "dataset_fingerprint.json",
)


def write_provenance(
    directory: pathlib.Path | str,
    config: Any,
    data_config: Any,
    *,
    checkpoint_step: int | None = None,
    preserve_existing: bool = False,
) -> pathlib.Path:
    """Write a reproducibility bundle and return the directory used.

    When ``preserve_existing`` is true, a differing bundle is written under a
    timestamped ``history`` directory. This keeps the original invocation intact
    when a run is resumed with overrides such as a larger ``num_train_steps``.
    """
    output_dir = pathlib.Path(directory)
    payloads = _build_payloads(config, data_config, checkpoint_step=checkpoint_step)

    if preserve_existing and any((output_dir / name).exists() for name in _PROVENANCE_FILES):
        if all(
            (output_dir / name).is_file() and (output_dir / name).read_text() == payload
            for name, payload in payloads.items()
        ):
            return output_dir
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_dir / "history" / f"{timestamp}_{os.getpid()}"

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _atomic_write_text(output_dir / name, payload)
    return output_dir


def _build_payloads(config: Any, data_config: Any, *, checkpoint_step: int | None) -> dict[str, str]:
    train_config = {
        "format_version": 1,
        "checkpoint_step": checkpoint_step,
        "resolved_paths": {
            "assets_dirs": _get_resolved_property(config, "assets_dirs"),
            "checkpoint_dir": _get_resolved_property(config, "checkpoint_dir"),
        },
        "config": _to_jsonable(config),
    }
    return {
        "train_config.json": _json_text(train_config),
        "launch_command.txt": _launch_command_text(),
        "git_commit.txt": _git_commit_text(),
        "dataset_fingerprint.json": _json_text(_dataset_fingerprint(data_config)),
    }


def _to_jsonable(value: Any) -> Any:
    """Convert nested config objects to deterministic, type-annotated JSON values."""
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, enum.Enum):
        return {"__type__": _qualified_type(value), "value": _to_jsonable(value.value)}
    if isinstance(value, pathlib.Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {"__type__": _qualified_type(value)}
        for field in dataclasses.fields(value):
            result[field.name] = _to_jsonable(getattr(value, field.name))
        return result
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, bytes | bytearray):
        return {"__type__": _qualified_type(value), "hex": bytes(value).hex()}

    # NumPy/JAX scalar and array types expose tolist without requiring either
    # framework as a dependency of this utility module.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except (TypeError, ValueError):
            pass

    return {"__type__": _qualified_type(value), "repr": repr(value)}


def _dataset_fingerprint(data_config: Any) -> dict[str, Any]:
    repo_id = _safe_getattr(data_config, "repo_id")
    asset_id = _safe_getattr(data_config, "asset_id")
    metadata_root = _find_dataset_root(repo_id, _safe_getattr(data_config, "rlds_data_dir"))
    metadata = _hash_metadata_tree(metadata_root / "meta") if metadata_root is not None else None

    norm_stats = _safe_getattr(data_config, "norm_stats")
    norm_stats_sha256 = None
    if norm_stats is not None:
        norm_stats_sha256 = hashlib.sha256(_canonical_json(_to_jsonable(norm_stats))).hexdigest()

    identity = {
        "repo_id": repo_id,
        "asset_id": asset_id,
        "action_sequence_keys": _to_jsonable(_safe_getattr(data_config, "action_sequence_keys")),
        "prompt_from_task": _safe_getattr(data_config, "prompt_from_task"),
        "rlds_data_dir": _safe_getattr(data_config, "rlds_data_dir"),
        "rlds_datasets": _to_jsonable(_safe_getattr(data_config, "datasets")),
        "metadata_sha256": metadata["sha256"] if metadata is not None else None,
        "norm_stats_sha256": norm_stats_sha256,
    }
    return {
        "format_version": 1,
        **identity,
        "dataset_root": str(metadata_root.resolve()) if metadata_root is not None else None,
        "metadata": metadata,
        "fingerprint_sha256": hashlib.sha256(_canonical_json(identity)).hexdigest(),
    }


def _find_dataset_root(repo_id: Any, rlds_data_dir: Any) -> pathlib.Path | None:
    candidates = [
        pathlib.Path(value).expanduser()
        for value in (rlds_data_dir, repo_id)
        if isinstance(value, str) and value and "://" not in value
    ]

    if isinstance(repo_id, str) and repo_id and "://" not in repo_id:
        lerobot_home = os.environ.get("HF_LEROBOT_HOME")
        if lerobot_home:
            candidates.append(pathlib.Path(lerobot_home).expanduser() / repo_id)
        hf_home = pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        candidates.append(hf_home / "lerobot" / repo_id)

    for candidate in candidates:
        if (candidate / "meta").is_dir():
            return candidate
    return None


def _hash_metadata_tree(meta_dir: pathlib.Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted((path for path in meta_dir.rglob("*") if path.is_file()), key=lambda path: path.as_posix()):
        relative = path.relative_to(meta_dir).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        file_count += 1
        total_bytes += size
    return {
        "path": str(meta_dir.resolve()),
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _launch_command_text() -> str:
    command = shlex.join([sys.executable, *sys.argv])
    lines = [f"cwd={pathlib.Path.cwd()}", f"command={command}"]
    parent_command = _read_process_command(os.getppid())
    if parent_command:
        lines.append(f"parent_command={parent_command}")
    return "\n".join(lines) + "\n"


def _read_process_command(pid: int) -> str | None:
    try:
        parts = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        args = [part.decode(errors="replace") for part in parts if part]
        return shlex.join(args) if args else None
    except OSError:
        return None


def _git_commit_text() -> str:
    repo_root = _git_output("rev-parse", "--show-toplevel")
    if repo_root is None:
        return "commit=unknown\nbranch=unknown\ndirty=unknown\n"

    commit = _git_output("-C", repo_root, "rev-parse", "HEAD") or "unknown"
    branch = _git_output("-C", repo_root, "branch", "--show-current") or "detached"
    status = _git_output("-C", repo_root, "status", "--porcelain", "--untracked-files=normal")
    dirty: bool | Literal["unknown"] = "unknown" if status is None else bool(status)
    return f"commit={commit}\nbranch={branch}\ndirty={str(dirty).lower()}\n"


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            cwd=pathlib.Path.cwd(),
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _get_resolved_property(config: Any, name: str) -> str | None:
    try:
        return str(getattr(config, name))
    except (AttributeError, TypeError, ValueError):
        return None


def _safe_getattr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name, None)
    except (AttributeError, TypeError, ValueError):
        return None


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(content)
    os.replace(tmp_path, path)
