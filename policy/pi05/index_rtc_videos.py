#!/usr/bin/env python3
"""Index a matrix run's videos by configuration.

`eval_policy` files videos under a timestamped directory that says nothing about
which cell produced them, so the raw tree cannot be browsed by configuration.
This builds a symlink tree keyed by task and cell instead -- no copying, and
re-running it just refreshes the links as more cells finish.

    python policy/pi05/index_rtc_videos.py report/rtc_matrix_final/<timestamp>
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, nargs="+")
    p.add_argument("--name", default="videos_by_cell")
    args = p.parse_args()

    linked = skipped = 0
    for raw_dir in args.run_dir:
        run_dir = raw_dir if raw_dir.is_absolute() else REPO_ROOT / raw_dir
        results = json.loads((run_dir / "results.json").read_text())["results"]
        index_root = run_dir / args.name
        index_root.mkdir(exist_ok=True)

        for r in results:
            if not r.get("metrics_path"):
                skipped += 1
                continue
            videos = (REPO_ROOT / r["metrics_path"]).parent / "videos"
            if not videos.is_dir():
                skipped += 1
                continue
            cell_dir = index_root / f"{r['task']}__{r['cell']['name']}"
            cell_dir.mkdir(exist_ok=True)
            for video in sorted(videos.glob("*.mp4")):
                link = cell_dir / video.name
                if link.is_symlink() or link.exists():
                    continue
                link.symlink_to(video.resolve())
                linked += 1

        # A plain-text map from cell to the run directory it came from, so the
        # provenance survives even if the symlinks are copied somewhere flat.
        manifest = [
            f"{r['task']}__{r['cell']['name']}\t{r.get('metrics_path') or '(no run dir)'}"
            for r in results
        ]
        (index_root / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n")
        print(f"{run_dir.name}: {len(results)} cells indexed -> {index_root}")

    print(f"linked {linked} videos, skipped {skipped} cells with no videos")


if __name__ == "__main__":
    sys.exit(main())
