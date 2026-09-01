from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import robosynchallenge.managers.events as events


class AssetStub:
    def __init__(self, uid: str, init_pos: list[float]):
        self.cfg = SimpleNamespace(uid=uid, init_pos=init_pos, init_rot=[0.0, 0.0, 0.0])
        self.pose = None
        self.clear_count = 0

    def set_local_pose(self, pose, env_ids=None):
        self.pose = pose.clone()

    def clear_dynamics(self, env_ids=None):
        self.clear_count += 1


class SimStub:
    def __init__(self):
        self.assets = {
            "table": AssetStub("table", [0.725, 0.0, 0.775]),
            "guijiao1": AssetStub("guijiao1", [0.6, -0.156, 0.8]),
            "guijiao2": AssetStub("guijiao2", [0.5, 0.16, 0.8]),
        }
        self.update_steps = []

    def get_rigid_object(self, uid):
        return self.assets.get(uid)

    def update(self, step):
        self.update_steps.append(step)


class EnvStub:
    def __init__(self):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.sim = SimStub()


def test_linked_randomization_preserves_object_height_relative_to_table(monkeypatch):
    env = EnvStub()
    shared_z = torch.tensor([-0.04, 0.05])

    def deterministic_sample(*, lower, upper, size, device):
        if tuple(size) == (2,):
            return shared_z.to(device)
        midpoint = (lower + upper) / 2
        return midpoint.expand(*size).clone()

    monkeypatch.setattr(events, "sample_uniform", deterministic_sample)
    events.randomize_linked_table_and_object_poses(
        env,
        env_ids=torch.tensor([0, 1]),
        table_entity_cfg={"uid": "table"},
        first_entity_cfg={"uid": "guijiao1"},
        second_entity_cfg={"uid": "guijiao2"},
        shared_z_range=[-0.05, 0.05],
        first_position_range=[[-0.01, -0.006, 0.0], [0.0, 0.004, 0.0]],
        second_position_range=[[0.002, -0.01, 0.0], [0.008, 0.0, 0.0]],
        physics_update_step=3,
    )

    table = env.sim.assets["table"].pose
    first = env.sim.assets["guijiao1"].pose
    second = env.sim.assets["guijiao2"].pose
    torch.testing.assert_close(first[:, 2, 3] - table[:, 2, 3], torch.full((2,), 0.025))
    torch.testing.assert_close(second[:, 2, 3] - table[:, 2, 3], torch.full((2,), 0.025))
    torch.testing.assert_close(table[:, 2, 3], torch.tensor([0.735, 0.825]))
    torch.testing.assert_close(first[:, :2, 3], torch.tensor([[0.595, -0.157], [0.595, -0.157]]))
    torch.testing.assert_close(second[:, :2, 3], torch.tensor([[0.505, 0.155], [0.505, 0.155]]))
    assert env.sim.update_steps == [3]
    assert all(asset.clear_count == 1 for asset in env.sim.assets.values())


def test_linked_randomization_rejects_independent_object_z_offsets():
    env = EnvStub()
    with pytest.raises(ValueError, match="Z bounds must be zero"):
        events.randomize_linked_table_and_object_poses(
            env,
            env_ids=torch.tensor([0]),
            table_entity_cfg={"uid": "table"},
            first_entity_cfg={"uid": "guijiao1"},
            second_entity_cfg={"uid": "guijiao2"},
            shared_z_range=[-0.05, 0.05],
            first_position_range=[[-0.01, -0.01, 0.01], [0.01, 0.01, 0.01]],
            second_position_range=[[-0.01, -0.01, 0.0], [0.01, 0.01, 0.0]],
        )
