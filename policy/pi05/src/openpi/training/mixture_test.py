import numpy as np
import pytest

from openpi.training import mixture


class _ListDataset:
    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, index):
        return self._items[index.__index__()]

    def __len__(self):
        return len(self._items)


def test_block_sizes_cover_every_source_with_requested_ratio():
    sizes = mixture.mixture_block_sizes([1000, 100], [0.5, 0.5])
    assert sizes == [1000, 1000]

    sizes = mixture.mixture_block_sizes([1000, 100], [0.8, 0.2])
    assert sizes[0] >= 1000
    assert sizes[1] >= 100
    assert sizes[0] / sum(sizes) == pytest.approx(0.8, abs=1e-3)


def test_mixture_samples_sources_in_proportion_and_covers_all_items():
    sim = _ListDataset([("sim", i) for i in range(30)])
    real = _ListDataset([("real", i) for i in range(4)])
    ds = mixture.WeightedMixtureDataset([sim, real], [3.0, 1.0])

    items = [ds[i] for i in range(len(ds))]
    sources = [s for s, _ in items]
    assert sources.count("sim") / len(items) == pytest.approx(0.75, abs=0.01)
    assert {i for s, i in items if s == "sim"} == set(range(30))
    assert {i for s, i in items if s == "real"} == set(range(4))
    assert ds[-1] == items[-1]
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_mixture_rejects_bad_weights():
    with pytest.raises(ValueError, match="finite and positive"):
        mixture.WeightedMixtureDataset([_ListDataset([1])], [0.0])
    with pytest.raises(ValueError, match="same number"):
        mixture.WeightedMixtureDataset([_ListDataset([1]), _ListDataset([2])], [1.0])


def test_gripper_rescale_maps_sim_v100_to_piper_70mm():
    state = np.zeros(14, dtype=np.float32)
    state[6] = 0.35  # 35 mm opening in simulation
    state[13] = 1.0  # fully open, 100 mm
    action = np.tile(state, (5, 1))
    sample = {"observation.qpos": state, "action": action, "prompt": "x"}

    out = mixture.GripperRescale(scale=mixture.SIM_V100_TO_PIPER_70MM)(sample)

    assert out["observation.qpos"][6] == pytest.approx(0.5)  # 35 mm on a 70 mm gripper
    assert out["observation.qpos"][13] == pytest.approx(1.0)  # clipped to fully open
    np.testing.assert_allclose(out["action"][:, 6], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out["observation.qpos"][:6], 0.0)
    assert out["prompt"] == "x"
    # The input sample is left untouched.
    assert state[6] == pytest.approx(0.35)


def test_gripper_rescale_identity_and_shape_check():
    sample = {"observation.qpos": np.ones(14)}
    assert mixture.GripperRescale(scale=1.0)(sample) is sample
    with pytest.raises(ValueError, match="gripper dims"):
        mixture.GripperRescale(scale=2.0)({"observation.qpos": np.ones(7)})
