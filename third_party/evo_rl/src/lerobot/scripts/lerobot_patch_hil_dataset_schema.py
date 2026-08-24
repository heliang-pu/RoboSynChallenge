#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Patch legacy human-in-loop datasets so they can merge with newer rollout datasets."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_record import _ensure_human_inloop_compatible_features
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging


MISSING_HIL_FEATURES = (
    "complementary_info.policy_action",
    "complementary_info.is_intervention",
    "complementary_info.state",
)


@dataclass
class PatchHilDatasetSchemaConfig:
    repo_id: str
    root: str | None = None
    output_repo_id: str | None = None
    output_dir: str | None = None


def _build_missing_hil_features(dataset: LeRobotDataset) -> dict[str, tuple[np.ndarray, dict]]:
    action_feature = dataset.features[ACTION]
    action_names = action_feature["names"]
    if action_names is None:
        action_names = [f"action_{idx}" for idx in range(action_feature["shape"][0])]
    else:
        action_names = list(action_names)

    feature_defs: dict[str, dict] = {}
    _ensure_human_inloop_compatible_features(feature_defs, action_feature_names=action_names)

    num_frames = dataset.num_frames
    num_action_dims = len(action_names)
    values_by_feature = {
        "complementary_info.policy_action": np.zeros((num_frames, num_action_dims), dtype=np.float32),
        "complementary_info.is_intervention": np.zeros((num_frames, 1), dtype=np.float32),
        "complementary_info.state": np.zeros((num_frames, 1), dtype=np.float32),
    }

    missing_features = {}
    for feature_name in MISSING_HIL_FEATURES:
        if feature_name not in dataset.features:
            missing_features[feature_name] = (values_by_feature[feature_name], feature_defs[feature_name])

    return missing_features


@parser.wrap()
def patch_hil_dataset_schema(cfg: PatchHilDatasetSchemaConfig) -> LeRobotDataset:
    init_logging()

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    missing_features = _build_missing_hil_features(dataset)
    if not missing_features:
        logging.info("Dataset already has a merge-compatible human-in-loop schema. Nothing to patch.")
        return dataset

    output_repo_id = cfg.output_repo_id or f"{cfg.repo_id}_hil_schema_patched"
    output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else None
    if output_dir is None and cfg.root is not None:
        src_root = Path(cfg.root)
        output_dir = src_root.parent / f"{src_root.name}_hil_schema_patched"

    logging.info("Adding missing human-in-loop schema features: %s", sorted(missing_features))
    patched_dataset = add_features(
        dataset=dataset,
        features=missing_features,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )
    logging.info("Patched dataset saved to %s", patched_dataset.root)
    return patched_dataset


def main():
    patch_hil_dataset_schema()


if __name__ == "__main__":
    main()
