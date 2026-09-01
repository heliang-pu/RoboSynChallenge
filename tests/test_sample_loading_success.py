from types import SimpleNamespace

import torch

from robosynchallenge.tasks.sample_loading.sample_loading import SampleLoadingEnv


class _RigidObject:
    def __init__(self, poses: torch.Tensor, vertices: torch.Tensor):
        self._poses = poses
        self._vertices = vertices.unsqueeze(0).expand(len(poses), -1, -1)

    def get_local_pose(self, to_matrix=True):
        assert to_matrix
        return self._poses

    def get_vertices(self, scale=False):
        assert scale
        return self._vertices


class _Simulation:
    def __init__(self, cube: _RigidObject, rack: _RigidObject):
        self._objects = {"cube": cube, "rack": rack}

    def get_rigid_object(self, name):
        return self._objects[name]


def _pose(*, translation=(0.0, 0.0, 0.0), y_degrees=0.0):
    angle = torch.deg2rad(torch.tensor(y_degrees))
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )
    pose[:3, 3] = torch.tensor(translation)
    return pose


def _evaluate(cube_poses, rack_poses):
    # Only the extrema matter to the evaluator. These match the approximate
    # scaled dimensions of the production tube and rack assets.
    cube_vertices = torch.tensor(
        [[-0.008, -0.008, -0.10], [0.008, 0.008, 0.10]]
    )
    rack_vertices = torch.tensor(
        [[-0.14, -0.06, -0.04], [0.14, 0.06, 0.04]]
    )
    stacked_cube_poses = torch.stack(cube_poses)
    initial_cube_poses = stacked_cube_poses.clone()
    initial_cube_poses[:, 2, 3] -= 0.10
    env = SimpleNamespace(
        num_envs=len(cube_poses),
        metadata={},
        _sample_initial_tube_pose=initial_cube_poses,
        _sample_tube_lifted=torch.zeros(len(cube_poses), dtype=torch.bool),
        _place_stable_count=torch.zeros(len(cube_poses), dtype=torch.long),
        success_stable_steps=1,
    )
    env.sim = _Simulation(
        _RigidObject(stacked_cube_poses, cube_vertices),
        _RigidObject(torch.stack(rack_poses), rack_vertices),
    )
    return SampleLoadingEnv._evaluate_task_state(env)


def test_sample_loading_success_requires_an_actual_insertion():
    rack_upright = _pose()
    rack_tipped = _pose(y_degrees=15.0)
    hole = (0.0123, -0.0016, 0.10)
    cube_relative_to_tipped_rack = rack_tipped @ _pose(translation=hole)

    cube_poses = [
        _pose(translation=hole),  # inserted
        _pose(translation=(0.1232, 0.0402, 0.10)),  # valid edge hole
        _pose(translation=(0.0123, -0.0016, 0.16)),  # hovering above rack
        _pose(translation=hole, y_degrees=90.0),  # lying across rack
        _pose(translation=(0.20, 0.0, 0.10)),  # outside rack footprint
        _pose(translation=(0.0, 0.0, -0.02)),  # passed below rack bottom
        cube_relative_to_tipped_rack,  # insertion in a fallen rack
        _pose(translation=(0.0234, -0.0016, 0.10)),  # on a bar between holes
    ]
    rack_poses = [rack_upright] * 6 + [rack_tipped, rack_upright]

    success, failure, metrics = _evaluate(cube_poses, rack_poses)

    assert success.tolist() == [True, True, False, False, False, False, False, False]
    assert failure.tolist() == [False, False, False, False, False, False, True, False]
    assert metrics["insertion_ok_single_frame"].tolist() == success.tolist()
