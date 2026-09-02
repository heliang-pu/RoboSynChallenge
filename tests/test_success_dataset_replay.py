"""Replay recorded task state through the production success predicates.

These tests deliberately call the task evaluators themselves instead of a
second, test-only reimplementation.  They are skipped on machines that do not
have the local RoboSynChallenge example datasets.
"""

import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

pq = pytest.importorskip("pyarrow.parquet")

from robosynchallenge.tasks.item_assembly.item_assembly import ItemAssemblyEnv
from robosynchallenge.tasks.items_handover.items_handover import ItemsHandoverEnv
from robosynchallenge.tasks.mixer_operating.mixer_operating import MixerOperatingEnv
from robosynchallenge.tasks.sample_loading.sample_loading import SampleLoadingEnv
from robosynchallenge.tasks.water_pouring.water_pouring import WaterPouringEnv


# 本地示例数据集路径，缺失时这些用例会自动 skip；用环境变量指到本机位置。
EXAMPLE_ROOT = Path(
    os.environ.get(
        "ROBOSYN_EXAMPLE_DATASET_ROOT",
        str(Path.home() / "Datacollect_T/RoboSynChallenge_ws/datasets_example/Sim"),
    )
)
FULL_SAMPLE_ROOT = Path(
    os.environ.get(
        "ROBOSYN_RAW_SAMPLE_LOADING",
        str(Path.home() / "workspace/dataset/cobotmagic_Sim_sample_loading"),
    )
)
ROLLOUT_SAMPLE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "lerobot_dataset/.simrecap_work/sample_loading_round1/shards"
)


class ReplayEntity:
    def __init__(self, pose=None, *, user_id=1, vertices=None):
        self.pose = pose
        self.user_id = user_id
        self.vertices = vertices

    def get_local_pose(self, to_matrix=True):
        assert to_matrix
        return self.pose

    def get_user_ids(self):
        return torch.tensor([self.user_id], dtype=torch.int32)

    def get_vertices(self, scale=False):
        assert scale
        return self.vertices.unsqueeze(0)


class ReplaySensor:
    def __init__(self):
        self.data = None

    def get_data(self):
        return self.data


class ReplaySim:
    def __init__(self, rigid, sensors=None):
        self.rigid = rigid
        self.sensors = sensors or {}

    def get_rigid_object(self, name):
        return self.rigid[name]

    def get_sensor(self, name):
        return self.sensors.get(name)


def bind(env, owner, *names):
    for name in names:
        setattr(env, name, MethodType(getattr(owner, name), env))


def pose_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)


def episode_files(task):
    root = EXAMPLE_ROOT / f"cobotmagic_Sim_{task}" / "data"
    if not root.exists():
        pytest.skip(f"success dataset is unavailable: {root}")
    return sorted(root.glob("**/*.parquet"))


def test_item_assembly_success_dataset_replay():
    passed = 0
    files = episode_files("item_assembly")
    for path in files:
        table = pq.read_table(path)
        pose_1 = np.asarray(table["guijiao1_pose"].to_pylist())
        pose_2 = np.asarray(table["guijiao2_pose"].to_pylist())
        sensor = ReplaySensor()
        entity_1 = ReplayEntity(user_id=100)
        entity_2 = ReplayEntity(user_id=96)
        env = SimpleNamespace(
            sim=ReplaySim(
                {"guijiao1": entity_1, "guijiao2": entity_2},
                {"guijiao_contact": sensor},
            ),
            num_envs=1,
            device=torch.device("cpu"),
            _elapsed_steps=torch.zeros(1, dtype=torch.long),
            _assembly_initial_pose_1=pose_tensor(pose_1[0]),
            _assembly_initial_pose_2=pose_tensor(pose_2[0]),
            _assembly_tube_1_lifted=torch.zeros(1, dtype=torch.bool),
            _assembly_tube_2_lifted=torch.zeros(1, dtype=torch.bool),
            _assembly_stable_count=torch.zeros(1, dtype=torch.long),
        )
        start_success = None
        for frame in range(len(pose_1)):
            entity_1.pose = pose_tensor(pose_1[frame])
            entity_2.pose = pose_tensor(pose_2[frame])
            sensor.data = {
                "distance": torch.as_tensor(
                    table["guijiao_contact.distance"][frame].as_py(),
                    dtype=torch.float32,
                ).unsqueeze(0),
                "is_valid": torch.as_tensor(
                    table["guijiao_contact.is_valid"][frame].as_py(),
                    dtype=torch.bool,
                ).unsqueeze(0),
                "user_ids": torch.as_tensor(
                    table["guijiao_contact.user_ids"][frame].as_py(),
                    dtype=torch.int32,
                ).unsqueeze(0),
            }
            env._elapsed_steps[:] = frame
            success, _, _ = ItemAssemblyEnv._evaluate_task_state(env)
            if frame == 0:
                start_success = bool(success.item())
        assert not start_success
        passed += int(success.item())
    assert passed == len(files) == 5


def test_items_handover_success_dataset_replay():
    passed = 0
    files = episode_files("items_handover")
    for path in files:
        table = pq.read_table(path)
        pen_poses = np.asarray(table["pen_pose"].to_pylist())
        holder_poses = np.asarray(table["holder_pose"].to_pylist())
        pen = ReplayEntity()
        holder = ReplayEntity()
        env = SimpleNamespace(
            sim=ReplaySim({"pen": pen, "holder": holder}),
            device=torch.device("cpu"),
            _handover_initial_pen_pose=pose_tensor(pen_poses[0]),
            _handover_pen_lifted=torch.zeros(1, dtype=torch.bool),
            _handover_pen_moved=torch.zeros(1, dtype=torch.bool),
            _handover_stable_count=torch.zeros(1, dtype=torch.long),
        )
        bind(env, ItemsHandoverEnv, "_is_fall_y")
        start_success = None
        for frame in range(len(pen_poses)):
            pen.pose = pose_tensor(pen_poses[frame])
            holder.pose = pose_tensor(holder_poses[frame])
            success, _, _ = ItemsHandoverEnv._evaluate_task_state(env)
            if frame == 0:
                start_success = bool(success.item())
        assert not start_success
        passed += int(success.item())
    assert passed == len(files) == 5


def test_mixer_operating_success_dataset_replay():
    passed = 0
    files = episode_files("mixer_operating")
    for path in files:
        table = pq.read_table(path)
        beaker_poses = np.asarray(table["beaker_pose"].to_pylist())
        mixer_poses = np.asarray(table["beaker_mixer_pose"].to_pylist())
        beaker = ReplayEntity()
        mixer = ReplayEntity(user_id=99)
        sensor = ReplaySensor()
        env = SimpleNamespace(
            sim=ReplaySim(
                {"beaker": beaker, "beaker_mixer": mixer},
                {"beaker_mixer_button_contact": sensor},
            ),
            num_envs=1,
            device=torch.device("cpu"),
            affordance_datas={
                "beaker_mixer_button_offset_x": -0.1176,
                "beaker_mixer_button_offset_y": 0.0,
                "beaker_mixer_button_offset_z": 0.056,
            },
            _button_contact_sensor=sensor,
            _button_region_radius=0.015,
            _button_impulse_threshold=0.005,
            _mixer_user_ids=torch.tensor([99], dtype=torch.int32),
            _arm_link_user_ids=torch.tensor([16, 17], dtype=torch.int32),
            _button_contact_happened=torch.zeros(1, dtype=torch.bool),
            _button_pressed_after_placement=torch.zeros(1, dtype=torch.bool),
            _mixer_initial_beaker_pose=pose_tensor(beaker_poses[0]),
            _mixer_beaker_lifted=torch.zeros(1, dtype=torch.bool),
            _mixer_beaker_moved=torch.zeros(1, dtype=torch.bool),
            _mixer_placement_count=torch.zeros(1, dtype=torch.long),
        )
        bind(
            env,
            MixerOperatingEnv,
            "_get_scalar_from_affordance",
            "_get_button_pose",
            "_get_button_position",
            "_update_button_contact_history",
            "_is_fall",
        )
        start_success = None
        for frame in range(len(beaker_poses)):
            beaker.pose = pose_tensor(beaker_poses[frame])
            mixer.pose = pose_tensor(mixer_poses[frame])
            sensor.data = {
                key: torch.as_tensor(
                    table[f"beaker_mixer_button_contact.{key}"][frame].as_py(),
                    dtype=torch.bool if key == "is_valid" else (
                        torch.int32 if key == "user_ids" else torch.float32
                    ),
                ).unsqueeze(0)
                for key in ("position", "impulse", "user_ids", "is_valid")
            }
            success, _, _ = MixerOperatingEnv._evaluate_task_state(env)
            if frame == 0:
                start_success = bool(success.item())
        assert not start_success
        passed += int(success.item())
    assert passed == len(files) == 5


def test_water_pouring_success_dataset_replay():
    passed = 0
    files = episode_files("water_pouring")
    for path in files:
        table = pq.read_table(path)
        bottle_poses = np.asarray(table["bottle_pose"].to_pylist())
        cup_poses = np.asarray(table["cup_pose"].to_pylist())
        bottle = ReplayEntity()
        cup = ReplayEntity()
        env = SimpleNamespace(
            sim=ReplaySim({"bottle": bottle, "cup": cup}),
            num_envs=1,
            device=torch.device("cpu"),
            _initial_bottle_pose=pose_tensor(bottle_poses[0]),
            _initial_cup_pose=pose_tensor(cup_poses[0]),
            _bottle_repositioned=torch.zeros(1, dtype=torch.bool),
            _cup_repositioned=torch.zeros(1, dtype=torch.bool),
            _pouring_started=torch.zeros(1, dtype=torch.bool),
            _pour_stable_count=torch.zeros(1, dtype=torch.long),
            _return_stable_count=torch.zeros(1, dtype=torch.long),
        )
        bind(env, WaterPouringEnv, "_is_fall_z")
        start_success = None
        for frame in range(len(bottle_poses)):
            bottle.pose = pose_tensor(bottle_poses[frame])
            cup.pose = pose_tensor(cup_poses[frame])
            success, _, _ = WaterPouringEnv._evaluate_task_state(env)
            if frame == 0:
                start_success = bool(success.item())
        assert not start_success
        passed += int(success.item())
    assert passed == len(files) == 5


def sample_episode_groups(dataset_root):
    for path in sorted((dataset_root / "data").glob("**/*.parquet")):
        table = pq.read_table(
            path, columns=["episode_index", "cube_pose", "rack_pose"]
        )
        episode_ids = np.asarray(table["episode_index"])
        cube_poses = np.asarray(table["cube_pose"].to_pylist())
        rack_poses = np.asarray(table["rack_pose"].to_pylist())
        for episode_id in np.unique(episode_ids):
            mask = episode_ids == episode_id
            yield int(episode_id), cube_poses[mask], rack_poses[mask]


def replay_sample_episode(cube_poses, rack_poses):
    cube_vertices = torch.tensor(
        [[-0.00838975, -0.00829567, -0.09750985],
         [0.00838975, 0.00829567, 0.09750985]],
        dtype=torch.float32,
    )
    rack_vertices = torch.tensor(
        [[-0.13761741, -0.05807605, -0.03964396],
         [0.13761741, 0.05807605, 0.03964396]],
        dtype=torch.float32,
    )
    cube = ReplayEntity(vertices=cube_vertices)
    rack = ReplayEntity(vertices=rack_vertices)
    lifted = bool(
        np.max(cube_poses[:, 2, 3] - cube_poses[0, 2, 3]) >= 0.05
    )
    env = SimpleNamespace(
        sim=ReplaySim({"cube": cube, "rack": rack}),
        num_envs=1,
        metadata={},
        _sample_initial_tube_pose=pose_tensor(cube_poses[0]),
        _sample_tube_lifted=torch.tensor([lifted], dtype=torch.bool),
        _place_stable_count=torch.zeros(1, dtype=torch.long),
    )
    cube.pose = pose_tensor(cube_poses[0])
    rack.pose = pose_tensor(rack_poses[0])
    start_success, _, _ = SampleLoadingEnv._evaluate_task_state(env)
    env._sample_tube_lifted[:] = lifted
    env._place_stable_count.zero_()
    cube.pose = pose_tensor(cube_poses[-1])
    rack.pose = pose_tensor(rack_poses[-1])
    final_success, _, _ = SampleLoadingEnv._evaluate_task_state(env)
    return bool(start_success.item()), bool(final_success.item())


def test_sample_loading_1000_success_dataset_replay():
    if not FULL_SAMPLE_ROOT.exists():
        pytest.skip(f"full success dataset is unavailable: {FULL_SAMPLE_ROOT}")
    outcomes = [
        (episode_id, *replay_sample_episode(cube_poses, rack_poses))
        for episode_id, cube_poses, rack_poses in sample_episode_groups(FULL_SAMPLE_ROOT)
    ]
    assert len(outcomes) == 1000
    assert sum(start for _, start, _ in outcomes) == 0
    # Episode 734 is an outlier created by the former permissive predicate: its
    # tube ends outside the rack and between hole centres, so a precise replay
    # must reject it even though it lives in the nominal "success" dataset.
    rejected = [episode_id for episode_id, _, final in outcomes if not final]
    assert rejected == [734]


def test_sample_loading_legacy_rollout_labels_are_tightened():
    if not ROLLOUT_SAMPLE_ROOT.exists():
        pytest.skip(f"labelled rollout dataset is unavailable: {ROLLOUT_SAMPLE_ROOT}")

    labels = {}
    episodes = []
    for shard in ("s0", "s1"):
        dataset = (
            ROLLOUT_SAMPLE_ROOT / shard / "cobotmagic_Sim_sample_loading_000"
        )
        label_path = dataset / "episode_success.json"
        if not label_path.exists():
            pytest.skip(f"rollout labels are unavailable: {label_path}")
        import json

        shard_labels = {
            int(item["episode_index"]): bool(item["success"])
            for item in json.loads(label_path.read_text())["episodes"]
        }
        for episode_id, cube_poses, rack_poses in sample_episode_groups(dataset):
            key = (shard, episode_id)
            labels[key] = shard_labels[episode_id]
            episodes.append((key, cube_poses, rack_poses))

    predictions = {
        key: replay_sample_episode(cube_poses, rack_poses)[1]
        for key, cube_poses, rack_poses in episodes
    }
    true_positive = sum(labels[key] and predictions[key] for key in labels)
    false_positive = sum(not labels[key] and predictions[key] for key in labels)
    false_negative = sum(labels[key] and not predictions[key] for key in labels)
    true_negative = sum(not labels[key] and not predictions[key] for key in labels)
    # These sidecar labels were produced by the old outer-bounding-box
    # predicate, not by independent human annotation.  The five disagreements
    # are shallow/misaligned placements on bars or at the rack edge; retaining
    # them would preserve the exact false-positive mode this review fixes.
    disagreements = sorted(
        key for key in labels if labels[key] and not predictions[key]
    )
    assert disagreements == [
        ("s0", 25),
        ("s1", 16),
        ("s1", 17),
        ("s1", 38),
        ("s1", 69),
    ]
    assert (true_positive, false_positive, false_negative, true_negative) == (
        6,
        0,
        5,
        139,
    )
