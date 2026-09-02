#!/usr/bin/env python
"""Decode a LeRobot v2.1 dataset into the LiLa-WAM JPEG frame cache.

    python policy/lila_wam/build_frame_cache.py --config policy/lila_wam/configs/robosyn_3cam.yaml

Training samples frames in random order, and random access into AV1 video is far
too slow for that; this converts each episode once into HDF5-stored JPEG buffers
at the training resolution. Re-running is a no-op unless the dataset, camera set
or image size changed (then pass --overwrite).
"""

from __future__ import annotations

import argparse
import logging

from _bootstrap import add_repo_root

add_repo_root()

from omegaconf import OmegaConf  # noqa: E402

from policy.lila_wam.lila_dataset import (  # noqa: E402
    ensure_frame_cache,
    load_metas,
    resolve_cache_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="LiLa-WAM config yaml")
    parser.add_argument("--overwrite", action="store_true", help="rebuild even if a cache exists")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    metas = load_metas(config)
    for meta in metas:
        logger.info(
            "%s: %d episodes / %d frames @ %g fps",
            meta.root,
            len(meta.episodes),
            sum(ep.length for ep in meta.episodes),
            meta.fps,
        )
        meta.require_cameras([str(c) for c in config.dataset.camera_names])

    logger.info("cache dir: %s", resolve_cache_dir(config))
    path = ensure_frame_cache(config, overwrite=args.overwrite)
    logger.info("frame cache ready: %s (%.2f GB)", path, path.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
