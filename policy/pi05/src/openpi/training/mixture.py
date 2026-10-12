"""Weighted mixing of several LeRobot datasets for sim + real co-training.

The stock openpi data loader reads exactly one LeRobot ``repo_id``. For sim2real
co-training we usually have one large simulated dataset and one much smaller
real-robot dataset, and we want to control how often each one is sampled
instead of letting the larger dataset dominate.

``WeightedMixtureDataset`` exposes several map-style datasets as a single one.
Each source owns a block of virtual indices whose size is proportional to its
weight, and the block is wrapped around the source so that every sample of
every source appears at least once per virtual epoch. The training loader then
shuffles the virtual indices as usual, so each batch draws from the sources
with the requested proportions.

``GripperRescale`` converts gripper values between two normalizations. The
simulated CobotMagic V100 gripper stores ``1.0`` for a 100 mm opening, while a
PiPER with the standard 70 mm gripper stores ``1.0`` for 70 mm. Scaling the
simulated values by ``100 / 70`` and clipping to ``[0, 1]`` puts both datasets
in the same physical unit, so the same commanded value means the same finger
opening on both robots.

This module only depends on numpy so that it can be unit tested without the
full training stack.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import logging
from typing import Any, Protocol, SupportsIndex

import numpy as np

# Active-joint layout of the RoboSynChallenge CobotMagic robot:
# left arm (0-5), left gripper (6), right arm (7-12), right gripper (13).
COBOTMAGIC_GRIPPER_DIMS: tuple[int, ...] = (6, 13)

# Full opening of the simulated CobotMagic V100 gripper: two fingers x 0.05 m.
SIM_V100_GRIPPER_FULL_MM = 100.0
# Full opening of the standard PiPER gripper.
PIPER_GRIPPER_FULL_MM = 70.0
# Factor that maps simulated V100 gripper values to PiPER 70 mm gripper values.
SIM_V100_TO_PIPER_70MM = SIM_V100_GRIPPER_FULL_MM / PIPER_GRIPPER_FULL_MM


class _MapDataset(Protocol):
    def __getitem__(self, index: SupportsIndex) -> Any: ...

    def __len__(self) -> int: ...


@dataclasses.dataclass(frozen=True)
class DatasetSource:
    """One LeRobot dataset that takes part in a weighted mixture."""

    # LeRobot repo id, resolved under ``HF_LEROBOT_HOME``.
    repo_id: str
    # Relative sampling weight. Weights are normalized over all sources.
    weight: float = 1.0
    # Multiplier applied to the gripper dimensions before mixing. Use
    # ``SIM_V100_TO_PIPER_70MM`` for simulated data when the real robot has
    # the 70 mm PiPER gripper, and ``1.0`` for data already in that unit.
    gripper_scale: float = 1.0
    # Indices of the gripper dimensions inside the state and action vectors.
    gripper_dims: tuple[int, ...] = COBOTMAGIC_GRIPPER_DIMS
    # Raw dataset keys whose gripper dimensions are rescaled.
    rescale_keys: tuple[str, ...] = ("observation.qpos", "action")


@dataclasses.dataclass(frozen=True)
class GripperRescale:
    """Scale and clip the gripper dimensions of selected keys in a sample."""

    scale: float
    dims: tuple[int, ...] = COBOTMAGIC_GRIPPER_DIMS
    keys: tuple[str, ...] = ("observation.qpos", "action")
    low: float = 0.0
    high: float = 1.0

    def __call__(self, data: dict) -> dict:
        if self.scale == 1.0:
            return data
        data = dict(data)
        dims = list(self.dims)
        for key in self.keys:
            if key not in data:
                continue
            value = np.array(data[key], dtype=np.float32, copy=True)
            if value.shape[-1] <= max(dims):
                raise ValueError(
                    f"Key {key!r} has last dimension {value.shape[-1]}, which does not contain gripper dims {dims}."
                )
            value[..., dims] = np.clip(value[..., dims] * self.scale, self.low, self.high)
            data[key] = value
        return data


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("At least one mixture weight is required.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(f"Mixture weights must be finite and positive, got {weights.tolist()}.")
    return weights / weights.sum()


def mixture_block_sizes(lengths: Sequence[int], weights: Sequence[float]) -> list[int]:
    """Return how many virtual indices each source gets.

    The virtual length is the smallest one for which every source is covered
    at least once at the requested proportion. Smaller sources are therefore
    repeated, and no sample of a larger source is ever dropped.
    """
    lengths = [int(n) for n in lengths]
    if len(lengths) != len(weights):
        raise ValueError("lengths and weights must have the same number of entries.")
    if any(n <= 0 for n in lengths):
        raise ValueError(f"Every mixture source needs at least one sample, got lengths {lengths}.")
    probs = normalize_weights(weights)
    total = int(np.ceil(max(n / p for n, p in zip(lengths, probs, strict=True))))
    return [max(n, round(float(p) * total)) for n, p in zip(lengths, probs, strict=True)]


class WeightedMixtureDataset:
    """Map-style dataset that samples several datasets with fixed proportions."""

    def __init__(self, datasets: Sequence[_MapDataset], weights: Sequence[float], names: Sequence[str] | None = None):
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same number of entries.")
        self._datasets = list(datasets)
        self._lengths = [len(d) for d in self._datasets]
        self._sizes = mixture_block_sizes(self._lengths, weights)
        self._offsets = np.cumsum([0, *self._sizes])
        self._names = list(names) if names is not None else [f"source_{i}" for i in range(len(datasets))]

        total = int(self._offsets[-1])
        for name, n, size in zip(self._names, self._lengths, self._sizes, strict=True):
            logging.info(
                "Mixture source %s: %d samples, %d virtual slots (%.1f%% of batches, %.2fx repeat).",
                name,
                n,
                size,
                100.0 * size / total,
                size / n,
            )

    @property
    def block_sizes(self) -> list[int]:
        return list(self._sizes)

    def locate(self, index: SupportsIndex) -> tuple[int, int]:
        """Map a virtual index to ``(source index, sample index in that source)``."""
        index = index.__index__()
        total = int(self._offsets[-1])
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"Index {index} is out of range for a mixture of length {total}.")
        source = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local = index - int(self._offsets[source])
        return source, local % self._lengths[source]

    def __getitem__(self, index: SupportsIndex) -> Any:
        source, local = self.locate(index)
        return self._datasets[source][local]

    def __len__(self) -> int:
        return int(self._offsets[-1])
