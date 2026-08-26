# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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
# ----------------------------------------------------------------------------

"""Dataset functors for collecting and saving episode data."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import gymnasium as gym
import torch
import tqdm

from tensordict import TensorDict

from embodichain.utils import logger
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATASET_ROOT
from embodichain.lab.gym.utils.misc import is_stereocam
from embodichain.lab.sim.sensors import Camera, ContactSensor
from embodichain.lab.gym.envs.managers.manager_base import Functor
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
from robosynchallenge.data.constants import ROBOSYNCHALLENGE_ROOT

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

def _load_recorder_lerobot_dataset():
    """Load the modern LeRobot dataset API without replacing openpi's legacy API.

    The pi0.5 environment intentionally pins an older LeRobot tree under the
    ``lerobot.common`` namespace, while EmbodiChain's recorder uses the newer
    ``lerobot.datasets`` namespace.  When ``LEROBOT_RECORDER_PACKAGE_ROOT`` is
    set, extend the installed package namespace with that modern LeRobot tree
    and make its sibling dependencies available as a last-resort import path
    (plus any dirs listed in ``LEROBOT_RECORDER_EXTRA_SITE_PACKAGES``).
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        package_root = os.environ.get("LEROBOT_RECORDER_PACKAGE_ROOT")
        if not package_root:
            return None

    package_root = Path(package_root).expanduser().resolve()
    if package_root.name != "lerobot":
        package_root = package_root / "lerobot"
    if not (package_root / "datasets" / "lerobot_dataset.py").is_file():
        return None

    import lerobot

    package_path = str(package_root)
    if package_path not in lerobot.__path__:
        lerobot.__path__.append(package_path)
    site_packages = str(package_root.parent)
    if site_packages not in sys.path:
        sys.path.append(site_packages)
    # Editable installs (e.g. third_party/evo_rl/src/lerobot) keep their
    # dependencies elsewhere; let the launcher hand us those site-packages
    # dirs as a last-resort import path too.
    for extra in os.environ.get("LEROBOT_RECORDER_EXTRA_SITE_PACKAGES", "").split(os.pathsep):
        if extra and Path(extra).is_dir() and extra not in sys.path:
            sys.path.append(extra)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        return None


LeRobotDataset = _load_recorder_lerobot_dataset()
LEROBOT_AVAILABLE = LeRobotDataset is not None

# EmbodiChain may have imported its dataset module before this compatibility
# bridge ran.  Refresh its module globals before subclassing the recorder.
import embodichain.lab.gym.envs.managers.datasets as _embodichain_datasets

if LEROBOT_AVAILABLE:
    _embodichain_datasets.LeRobotDataset = LeRobotDataset
    _embodichain_datasets.LEROBOT_AVAILABLE = True

EmbodiChainLeRobotRecorder = _embodichain_datasets.LeRobotRecorder

__all__ = ["LeRobotRecorder", "install_lerobot_recorder_override"]


class LeRobotRecorder(EmbodiChainLeRobotRecorder):
    """LeRobot recorder with save_path resolved relative to RoboSynChallenge."""

    def __init__(self, cfg: DatasetFunctorCfg, env: EmbodiedEnv):
        save_path = cfg.params.get("save_path", None)
        if save_path:
            save_path = Path(str(save_path)).expanduser()
            if not save_path.is_absolute():
                params = dict(cfg.params)
                params["save_path"] = str(ROBOSYNCHALLENGE_ROOT / save_path)
                cfg.params = params

        super().__init__(cfg, env)


def install_lerobot_recorder_override() -> None:
    """Make EmbodiChain config lookup resolve LeRobotRecorder to this subclass."""
    import embodichain.lab.gym.envs.managers as manager_module
    import embodichain.lab.gym.envs.managers.datasets as dataset_module

    dataset_module.LeRobotRecorder = LeRobotRecorder
    manager_module.LeRobotRecorder = LeRobotRecorder
