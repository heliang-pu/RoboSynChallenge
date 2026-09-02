#!/usr/bin/env python3
"""Atomically replace identical recap video copies with hardlinks."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def videos(root: Path) -> dict[str, Path]:
    base = root / "videos"
    result = {
        path.relative_to(base).as_posix(): path
        for path in sorted(base.glob("**/*.mp4"))
    }
    other_files = [path for path in base.glob("**/*") if path.is_file() and path.suffix != ".mp4"]
    if other_files:
        raise RuntimeError(f"unexpected non-MP4 files under {base}: {other_files[:10]}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--duplicate", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    roots = [args.canonical.resolve(), *(path.resolve() for path in args.duplicate)]
    if len(set(roots)) != len(roots):
        raise ValueError("canonical and duplicate roots must be distinct")
    collections = [videos(root) for root in roots]
    expected = set(collections[0])
    if len(expected) != 1050:
        raise RuntimeError(f"expected 1050 canonical MP4s, found {len(expected)}")
    for root, collection in zip(roots[1:], collections[1:]):
        if set(collection) != expected:
            raise RuntimeError(f"video relative-path set differs: {root}")
    for relative in sorted(expected):
        sizes = [collection[relative].stat().st_size for collection in collections]
        if len(set(sizes)) != 1 or sizes[0] <= 0:
            raise RuntimeError(f"video size mismatch: {relative}: {sizes}")

    hash_jobs = [path for collection in collections for path in collection.values()]
    hashes: dict[Path, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(sha256, path): path for path in hash_jobs}
        for index, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            path = future_map[future]
            hashes[path] = future.result()
            if index % 150 == 0 or index == len(future_map):
                print(f"SHA256 {index}/{len(future_map)}", flush=True)
    for relative in sorted(expected):
        values = [hashes[collection[relative]] for collection in collections]
        if len(set(values)) != 1:
            raise RuntimeError(f"video SHA256 mismatch: {relative}: {values}")

    bytes_per_copy = sum(collections[0][relative].stat().st_size for relative in expected)
    replaced = 0
    if args.execute:
        for duplicate_collection in collections[1:]:
            for relative in sorted(expected):
                source = collections[0][relative]
                destination = duplicate_collection[relative]
                temporary = destination.with_name(f".{destination.name}.hardlink_tmp")
                if temporary.exists():
                    raise FileExistsError(temporary)
                os.link(source, temporary)
                os.replace(temporary, destination)
                replaced += 1
        for relative in sorted(expected):
            link_counts = [collection[relative].stat().st_nlink for collection in collections]
            if min(link_counts) < 3:
                raise RuntimeError(f"unexpected hardlink count after replace: {relative}: {link_counts}")

    report = {
        "passed": True,
        "executed": bool(args.execute),
        "canonical": str(roots[0]),
        "duplicates": [str(root) for root in roots[1:]],
        "video_files_per_dataset": len(expected),
        "verified_files": len(hash_jobs),
        "sha256_all_identical": True,
        "bytes_per_video_copy": bytes_per_copy,
        "duplicate_copies_relinked": len(roots) - 1 if args.execute else 0,
        "files_atomically_replaced": replaced,
        "estimated_bytes_reclaimed": bytes_per_copy * (len(roots) - 1) if args.execute else 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
