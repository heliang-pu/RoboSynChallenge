from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from robosynchallenge.tasks.click_bell.click_bell import ClickBellEnv
from robosynchallenge.tasks.drawer_open_place.drawer_open_place import DrawerOpenPlaceEnv
from robosynchallenge.tasks.handle_basket.handle_basket import HandleBasketEnv
from robosynchallenge.tasks.item_assembly.item_assembly import ItemAssemblyEnv
from robosynchallenge.tasks.items_handover.items_handover import ItemsHandoverEnv
from robosynchallenge.tasks.manipulate_pipette.manipulate_pipette import ManipulatePipetteEnv
from robosynchallenge.tasks.mixer_operating.mixer_operating import MixerOperatingEnv
from robosynchallenge.tasks.table_rearrangement.table_rearrangement import TableRearrangementEnv
from robosynchallenge.tasks.water_pouring.water_pouring import WaterPouringEnv


def pose(position=(0.0, 0.0, 0.0), rotation=None):
    value = torch.eye(4, dtype=torch.float32)
    if rotation is not None:
        value[:3, :3] = torch.as_tensor(rotation, dtype=torch.float32)
    value[:3, 3] = torch.tensor(position, dtype=torch.float32)
    return value.unsqueeze(0)


def rot_x(degrees):
    angle = torch.deg2rad(torch.tensor(float(degrees)))
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(degrees):
    angle = torch.deg2rad(torch.tensor(float(degrees)))
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(degrees):
    angle = torch.deg2rad(torch.tensor(float(degrees)))
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Entity:
    def __init__(self, entity_pose=None, qpos=None, limits=None, links=None):
        self.pose = entity_pose
        self.qpos = qpos
        self.qpos_limits = limits
        self.links = links or {}

    def get_local_pose(self, to_matrix=True):
        return self.pose

    def get_qpos(self):
        return self.qpos

    def get_link_pose(self, name, to_matrix=True):
        return self.links[name]

    def get_user_ids(self):
        return torch.tensor([1], dtype=torch.int32)


class Sim:
    def __init__(self, rigid=None, articulation=None, sensors=None):
        self.rigid = rigid or {}
        self.articulation = articulation or {}
        self.sensors = sensors or {}

    def get_rigid_object(self, name):
        return self.rigid[name]

    def get_articulation(self, name):
        return self.articulation[name]

    def get_sensor(self, name):
        return self.sensors.get(name)


def test_handle_basket_keeps_action_config_from_launcher():
    action_config = {"scope": {}, "node": {}, "edge": {}}
    with patch(
        "robosynchallenge.tasks.handle_basket.handle_basket.EmbodiedEnv.__init__",
        return_value=None,
    ):
        env = HandleBasketEnv(action_config=action_config)
    assert env.action_config is action_config
    env.affordance_datas = {
        "milk_pose": pose((0.21, -0.12, 0.03)).squeeze(0).numpy(),
        "basket_pose": pose((0.63, 0.18, 0.04)).squeeze(0).numpy(),
    }
    env._sync_carry_basket_runtime_attrs()
    assert env.milk_xy_random_center.tolist() == pytest.approx([0.21, -0.12])
    assert env.basket_xy_random_center.tolist() == pytest.approx([0.63, 0.18])
    env.sim = Sim(rigid={"basket": Entity(pose((0.63, 0.18, 0.94)))})
    env._capture_basket_success_baseline()
    assert env._hb_orig_basket_x == pytest.approx(0.63)
    assert env._hb_orig_basket_y == pytest.approx(0.18)
    assert env._hb_orig_basket_z == pytest.approx(0.94)


def test_click_bell_requires_real_press():
    env = SimpleNamespace(
        sim=Sim(articulation={"button": Entity(qpos=torch.tensor([[-0.0041]]))}),
        _button_pressed=torch.zeros(1, dtype=torch.bool),
    )
    success, _ = ClickBellEnv._evaluate_task_state(env)
    assert success.tolist() == [True]

    env._button_pressed.zero_()
    env.sim.articulation["button"].qpos[:] = -0.001
    success, _ = ClickBellEnv._evaluate_task_state(env)
    assert success.tolist() == [False]


def test_drawer_requires_open_drawer_and_object_inside():
    drawer = Entity(
        qpos=torch.tensor([[0.12]]),
        links={"inner_box": pose()},
    )
    env = SimpleNamespace(
        sim=Sim(
            rigid={"duck": Entity(pose((0.02, 0.0, 0.03)))},
            articulation={"drawer": drawer},
        ),
        _drawer_initial_qpos=torch.tensor([0.0]),
        _drawer_initial_object_pose=pose((0.30, 0.0, -0.10)),
        _drawer_object_lifted=torch.zeros(1, dtype=torch.bool),
        _drawer_object_moved=torch.zeros(1, dtype=torch.bool),
        _drawer_success_count=torch.zeros(1, dtype=torch.long),
    )
    for _ in range(3):
        success, _, _ = DrawerOpenPlaceEnv._evaluate_task_state(env)
    assert success.tolist() == [True]
    drawer.qpos[:] = 0.01
    success, _, _ = DrawerOpenPlaceEnv._evaluate_task_state(env)
    assert success.tolist() == [False]


def test_handle_basket_requires_place_then_carry():
    basket = Entity(pose(rotation=rot_x(90)))
    milk = Entity(pose((0.20, 0.0, 0.0)))
    env = SimpleNamespace(
        sim=Sim(rigid={"basket": basket, "milk": milk}),
        device=torch.device("cpu"),
        num_envs=1,
        _hb_initial_basket_pose=basket.pose.clone(),
        _hb_initial_milk_pose=milk.pose.clone(),
        _hb_milk_lifted=torch.zeros(1, dtype=torch.bool),
        _hb_milk_placed=torch.zeros(1, dtype=torch.bool),
        _hb_basket_carried=torch.zeros(1, dtype=torch.bool),
        _hb_success_count=torch.zeros(1, dtype=torch.long),
    )
    env._reset_success_state = lambda reset_ids=None: HandleBasketEnv._reset_success_state(env, reset_ids)
    milk.pose = pose((0.20, 0.0, 0.08))
    HandleBasketEnv._evaluate_task_state(env)
    milk.pose = pose((0.0, 0.0, 0.08))
    HandleBasketEnv._evaluate_task_state(env)
    basket.pose = pose((0.10, 0.0, 0.10), rot_x(90))
    milk.pose = pose((0.10, 0.0, 0.16))
    success, _, _ = HandleBasketEnv._evaluate_task_state(env)
    assert success.tolist() == [False]
    assert env._hb_basket_carried.tolist() == [True]

    # Carrying is only an intermediate state; the basket must be set down at
    # the new location before the task can complete.
    basket.pose = pose((0.10, 0.0, 0.0), rot_x(90))
    milk.pose = pose((0.10, 0.0, 0.06))
    for _ in range(3):
        success, _, _ = HandleBasketEnv._evaluate_task_state(env)
    assert success.tolist() == [True]


def test_item_assembly_rejects_parallel_but_separated_tubes():
    g1 = Entity(pose())
    g2 = Entity(pose((0.20, 0.01, 0.0)))
    env = SimpleNamespace(
        sim=Sim(rigid={"guijiao1": g1, "guijiao2": g2}),
        num_envs=1,
        device=torch.device("cpu"),
        _elapsed_steps=torch.tensor([50]),
        _assembly_initial_pose_1=pose((0.0, 0.0, -0.10)),
        _assembly_initial_pose_2=pose((0.20, 0.01, -0.10)),
        _assembly_tube_1_lifted=torch.zeros(1, dtype=torch.bool),
        _assembly_tube_2_lifted=torch.zeros(1, dtype=torch.bool),
        _assembly_stable_count=torch.zeros(1, dtype=torch.long),
    )
    for _ in range(3):
        success, _, _ = ItemAssemblyEnv._evaluate_task_state(env)
    assert success.tolist() == [True]
    g2.pose = pose((0.20, 0.08, 0.0))
    success, _, _ = ItemAssemblyEnv._evaluate_task_state(env)
    assert success.tolist() == [False]


def test_item_assembly_reports_success_metrics_without_changing_predicate():
    env = SimpleNamespace(
        sim=Sim(
            rigid={
                "guijiao1": Entity(pose()),
                "guijiao2": Entity(pose((0.10, 0.0, 0.0))),
            }
        ),
        num_envs=1,
        device=torch.device("cpu"),
        _elapsed_steps=torch.tensor([10]),
        success_lateral_tol=0.02,
    )
    success = ItemAssemblyEnv.is_task_success(env)
    assert success.tolist() == [True]
    assert env._last_success_metrics["success"].tolist() == [True]
    assert env._last_success_metrics["angle_deg"].tolist() == pytest.approx([0.0])
    assert env._last_success_metrics["lateral_offset"].tolist() == pytest.approx([0.0])


def test_items_handover_uses_holder_frame_insertion():
    holder_pose = pose(rotation=rot_x(90))
    pen_rel = pose((0.01, 0.075, 0.01), rot_z(90))
    pen_pose = torch.bmm(holder_pose, pen_rel)
    env = SimpleNamespace(
        sim=Sim(rigid={"holder": Entity(holder_pose), "pen": Entity(pen_pose)}),
        device=torch.device("cpu"),
        _handover_initial_pen_pose=pose((0.20, 0.0, -0.10)),
        _handover_pen_lifted=torch.zeros(1, dtype=torch.bool),
        _handover_pen_moved=torch.zeros(1, dtype=torch.bool),
        _handover_stable_count=torch.zeros(1, dtype=torch.long),
    )
    env._is_fall_y = lambda value: ItemsHandoverEnv._is_fall_y(env, value)
    for _ in range(3):
        success, _, _ = ItemsHandoverEnv._evaluate_task_state(env)
    assert success.tolist() == [True]
    env.sim.rigid["pen"].pose = torch.bmm(holder_pose, pose((0.10, 0.075, 0.0), rot_z(90)))
    success, _, _ = ItemsHandoverEnv._evaluate_task_state(env)
    assert success.tolist() == [False]

    # A pen inserted tail-first points opposite the holder axis and must not
    # receive credit merely because the two axes are parallel.
    env.sim.rigid["pen"].pose = torch.bmm(
        holder_pose, pose((0.0, 0.075, 0.0), rot_z(-90))
    )
    success, _, _ = ItemsHandoverEnv._evaluate_task_state(env)
    assert success.tolist() == [False]


def test_pipette_press_only_counts_over_beaker():
    beaker = Entity(pose())
    pipette_pose = pose((0.0, 0.0, 0.234), rot_y(-90))
    pipette = Entity(
        pipette_pose,
        qpos=torch.tensor([[-0.002]]),
        limits=torch.tensor([[[-0.002, 0.00467]]]),
    )
    env = SimpleNamespace(
        sim=Sim(rigid={"beaker1": beaker}, articulation={"pipette": pipette}),
        device=torch.device("cpu"),
        _pipette_min_tolerance=1e-4,
        _pipette_min_reach_count=torch.zeros(1, dtype=torch.int32),
        _pipette_was_at_min=torch.zeros(1, dtype=torch.bool),
        _pipette_pressed_over_beaker=torch.zeros(1, dtype=torch.bool),
        _pipette_initial_pose=pose((0.20, 0.0, 0.0), rot_y(-90)),
        _pipette_was_lifted=torch.zeros(1, dtype=torch.bool),
        _pipette_was_moved=torch.zeros(1, dtype=torch.bool),
    )
    env._is_fall_z = lambda value: ManipulatePipetteEnv._is_fall_z(env, value)
    env._is_fall_x = lambda value: ManipulatePipetteEnv._is_fall_x(env, value)
    env._get_pipette_slide_limits = lambda entity: ManipulatePipetteEnv._get_pipette_slide_limits(env, entity)
    success, _, _ = ManipulatePipetteEnv._evaluate_task_state(env)
    assert success.tolist() == [True]

    env._pipette_pressed_over_beaker.zero_()
    env._pipette_was_at_min.zero_()
    pipette.pose = pose((0.20, 0.0, 0.234), rot_y(-90))
    success, _, _ = ManipulatePipetteEnv._evaluate_task_state(env)
    assert success.tolist() == [False]

    # A horizontal pipette is a normal intermediate pickup pose, not an
    # episode-ending failure; it simply cannot satisfy success.
    pipette.qpos[:] = 0.00467
    pipette.pose = pose((0.20, 0.0, 0.10))
    success, failure, _ = ManipulatePipetteEnv._evaluate_task_state(env)
    assert success.tolist() == [False]
    assert failure.tolist() == [False]


def test_mixer_requires_beaker_placement_and_switch_history():
    env = SimpleNamespace(
        sim=Sim(
            rigid={
                "beaker": Entity(pose((0.01, 0.044, -0.028))),
                "beaker_mixer": Entity(pose()),
            }
        ),
        _button_contact_sensor=None,
        _button_contact_happened=torch.zeros(1, dtype=torch.bool),
        _button_pressed_after_placement=torch.zeros(1, dtype=torch.bool),
        _mixer_initial_beaker_pose=pose((0.20, 0.20, -0.10)),
        _mixer_beaker_lifted=torch.zeros(1, dtype=torch.bool),
        _mixer_beaker_moved=torch.zeros(1, dtype=torch.bool),
        _mixer_placement_count=torch.zeros(1, dtype=torch.long),
        press_now=torch.tensor([True]),
    )
    def update_button():
        env._button_contact_happened |= env.press_now
        return env.press_now

    env._update_button_contact_history = update_button
    env._is_fall = lambda value: MixerOperatingEnv._is_fall(env, value)

    # A switch press before the beaker has been stable for three frames does
    # not count, even if the beaker settles afterward.
    success, _, _ = MixerOperatingEnv._evaluate_task_state(env)
    env.press_now[:] = False
    MixerOperatingEnv._evaluate_task_state(env)
    success, _, _ = MixerOperatingEnv._evaluate_task_state(env)
    assert success.tolist() == [False]
    env.press_now[:] = True
    success, _, _ = MixerOperatingEnv._evaluate_task_state(env)
    assert success.tolist() == [True]


def test_table_rearrangement_uses_plate_frame_and_bounded_height():
    plate_pose = pose(rotation=rot_x(180))
    spoon_pose = torch.bmm(plate_pose, pose((0.0, 0.16, 0.0)))
    fork_pose = torch.bmm(plate_pose, pose((0.0, -0.16, 0.0)))
    env = SimpleNamespace(
        sim=Sim(
            rigid={
                "plate": Entity(plate_pose),
                "spoon": Entity(spoon_pose),
                "fork": Entity(fork_pose),
            }
        ),
        metadata={},
        device=torch.device("cpu"),
        _table_initial_fork_pose=pose((0.20, 0.20, -0.10)),
        _table_initial_spoon_pose=pose((-0.20, -0.20, -0.10)),
        _table_fork_lifted=torch.zeros(1, dtype=torch.bool),
        _table_spoon_lifted=torch.zeros(1, dtype=torch.bool),
        _table_fork_moved=torch.zeros(1, dtype=torch.bool),
        _table_spoon_moved=torch.zeros(1, dtype=torch.bool),
        _table_success_count=torch.zeros(1, dtype=torch.long),
    )
    for _ in range(3):
        success, _, _ = TableRearrangementEnv._evaluate_task_state(env)
    assert success.tolist() == [True]
    env.sim.rigid["fork"].pose = torch.bmm(plate_pose, pose((0.0, -0.16, 0.20)))
    success, _, _ = TableRearrangementEnv._evaluate_task_state(env)
    assert success.tolist() == [False]

    upside_down_plate = pose()
    env.sim.rigid["plate"].pose = upside_down_plate
    env.sim.rigid["spoon"].pose = torch.bmm(
        upside_down_plate, pose((0.0, 0.16, 0.0))
    )
    env.sim.rigid["fork"].pose = torch.bmm(
        upside_down_plate, pose((0.0, -0.16, 0.0))
    )
    success, _, _ = TableRearrangementEnv._evaluate_task_state(env)
    assert success.tolist() == [False]


def test_water_pouring_requires_pour_then_return_upright():
    upright = rot_x(90)
    bottle = Entity(pose((0.0, 0.0, -0.10), upright))
    cup = Entity(pose())
    env = SimpleNamespace(
        sim=Sim(rigid={"bottle": bottle, "cup": cup}),
        num_envs=1,
        device=torch.device("cpu"),
        _initial_bottle_pose=bottle.pose.clone(),
        _initial_cup_pose=cup.pose.clone(),
        _bottle_repositioned=torch.ones(1, dtype=torch.bool),
        _cup_repositioned=torch.ones(1, dtype=torch.bool),
        _pouring_started=torch.zeros(1, dtype=torch.bool),
        _pour_stable_count=torch.zeros(1, dtype=torch.long),
        _return_stable_count=torch.zeros(1, dtype=torch.long),
    )
    env._is_fall_z = lambda value: WaterPouringEnv._is_fall_z(env, value)

    pouring_rotation = rot_x(30)
    bottle.pose = pose((0.0, -0.204, -0.018), pouring_rotation)
    for _ in range(3):
        success, _, _ = WaterPouringEnv._evaluate_task_state(env)
    assert success.tolist() == [False]
    bottle.pose = pose((0.0, 0.0, 0.0), upright)
    for _ in range(3):
        success, _, _ = WaterPouringEnv._evaluate_task_state(env)
    assert success.tolist() == [True]
