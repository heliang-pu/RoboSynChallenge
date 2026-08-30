#!/usr/bin/env python3
"""Compute exact state/action stats without decoding unused videos."""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch
import tqdm

from lerobot.common.datasets import lerobot_dataset
import openpi.shared.normalize as normalize
import openpi.training.config as config_lib
import openpi.training.data_loader as data_loader
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class StateActionOnly(transforms.DataTransformFn):
    def __call__(self, item: dict) -> dict:
        state = item.get("observation.state")
        if state is None:
            state = item["observation.qpos"]
        return {
            "state": np.asarray(state),
            "actions": np.asarray(item["action"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    config = config_lib.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("config has no LeRobot repo_id")

    delta_mask = transforms.make_bool_mask(6, -1, 6, -1)
    source_repo_ids = tuple(data_config.lerobot_repo_ids) or (data_config.repo_id,)
    datasets = []
    for source_repo_id in source_repo_ids:
        metadata = lerobot_dataset.LeRobotDatasetMetadata(source_repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            source_repo_id,
            delta_timestamps={
                key: [step / metadata.fps for step in range(config.model.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        dataset.meta.info["features"] = {
            key: feature
            for key, feature in dataset.meta.info["features"].items()
            if feature.get("dtype") not in {"video", "image"}
        }
        datasets.append(
            data_loader.TransformedDataset(
                dataset,
                [StateActionOnly(), transforms.DeltaActions(delta_mask)],
            )
        )
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    num_batches = len(dataset) // config.batch_size
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        num_batches=num_batches,
        framework="pytorch",
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing state/action stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    output = config.assets_dirs / data_config.repo_id
    normalize.save(output, {key: value.get_statistics() for key, value in stats.items()})
    print(f"Wrote exact stats for {len(dataset)} frames to {output}")


if __name__ == "__main__":
    main()
