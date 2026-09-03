"""ParallelEvalProxy 与并行种子计划的单元测试(stub 环境,不起仿真)。

官方口径的关键语义在这里钉死:
  * 成功 = 某步 step 后 is_task_success 为 True 且该 env 未截断;
  * 截断当步的成功不计(真实环境里该步 auto-reset 已清锁存,代理侧同样屏蔽);
  * 成功一经锁存不回退;
  * 种子按 episode 序号同序抽取,分片/并行怎么组合都不改变 episode k 的种子。
"""

import numpy as np
import pytest
import torch

from scripts.eval_policy_parallel import ParallelEvalProxy, build_episode_plan


class StubBatchEnv:
    """Batched env stub: success per a schedule fn, truncation at a fixed step."""

    def __init__(self, num_envs, success_at=None, truncate_at=10):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.unwrapped = self
        self.step_count = 0
        self._success_at = success_at or {}  # slot -> first success step
        self._truncate_at = truncate_at

    def _is_task_success(self):
        flags = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.step_count >= self._truncate_at:
            # 真实环境在截断步 auto-reset 后锁存已清零。
            return flags
        for slot, at_step in self._success_at.items():
            if self.step_count >= at_step:
                flags[slot] = True
        return flags

    def get_wrapper_attr(self, name):
        assert name == "is_task_success"
        return self._is_task_success

    def step(self, action):
        self.step_count += 1
        truncated = torch.full(
            (self.num_envs,), self.step_count >= self._truncate_at, dtype=torch.bool
        )
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        return None, torch.zeros(self.num_envs), terminated, truncated, {}


def _batched_action(num_envs, dof=14):
    return torch.zeros(num_envs, dof)


def test_success_latched_with_step_count():
    env = StubBatchEnv(num_envs=2, success_at={1: 3}, truncate_at=10)
    proxy = ParallelEvalProxy(env, num_envs=2)
    proxy.begin_wave(active_count=2)

    for _ in range(4):
        proxy.step(_batched_action(2))

    assert not proxy.all_done()  # slot 0 还没结束
    success, env_steps, _ = proxy.episode_result(1, max_env_steps=10)
    assert success and env_steps == 3
    success, env_steps, _ = proxy.episode_result(0, max_env_steps=10)
    assert not success and env_steps == 10


def test_wave_finishes_on_truncation():
    env = StubBatchEnv(num_envs=2, success_at={1: 3}, truncate_at=5)
    proxy = ParallelEvalProxy(env, num_envs=2)
    proxy.begin_wave(active_count=2)

    steps = 0
    while not proxy.all_done():
        proxy.step(_batched_action(2))
        steps += 1
        assert steps <= 5

    assert steps == 5
    assert proxy.episode_result(1, max_env_steps=5) == (True, 3, None)
    assert proxy.episode_result(0, max_env_steps=5) == (False, 5, None)


def test_truncation_step_success_does_not_count():
    # 成功信号首次出现在截断当步:官方口径不算成功。
    env = StubBatchEnv(num_envs=1, success_at={0: 4}, truncate_at=4)

    # 绕过 stub 的「截断步清零」,显式让截断当步 is_task_success 为 True,
    # 验证的是代理自身的屏蔽逻辑而非 stub 的行为。
    env._is_task_success = lambda: torch.tensor(
        [env.step_count >= 4], dtype=torch.bool
    )

    proxy = ParallelEvalProxy(env, num_envs=1)
    proxy.begin_wave(active_count=1)
    for _ in range(4):
        proxy.step(_batched_action(1))

    assert proxy.all_done()
    assert proxy.episode_result(0, max_env_steps=4) == (False, 4, None)


def test_success_latch_survives_signal_dropping():
    # 非锁存型任务:成功位在后续步骤回落,已锁存的成功不回退。
    env = StubBatchEnv(num_envs=1, truncate_at=6)
    env._is_task_success = lambda: torch.tensor(
        [env.step_count == 2], dtype=torch.bool
    )
    proxy = ParallelEvalProxy(env, num_envs=1)
    proxy.begin_wave(active_count=1)

    for _ in range(3):
        proxy.step(_batched_action(1))

    assert proxy.all_done()
    assert proxy.episode_result(0, max_env_steps=6) == (True, 2, None)


def test_inactive_slots_do_not_block_wave():
    env = StubBatchEnv(num_envs=3, success_at={0: 1, 1: 1}, truncate_at=10)
    proxy = ParallelEvalProxy(env, num_envs=3)
    proxy.begin_wave(active_count=2)  # 槽位 2 是填充位

    proxy.step(_batched_action(3))
    assert proxy.all_done()


def test_aggregated_is_task_success_feeds_adapters():
    env = StubBatchEnv(num_envs=2, success_at={0: 1, 1: 2}, truncate_at=10)
    proxy = ParallelEvalProxy(env, num_envs=2)
    proxy.begin_wave(active_count=2)

    is_done = proxy.get_wrapper_attr("is_task_success")
    proxy.step(_batched_action(2))
    assert is_done() is False  # 适配器视角:还有 env 在跑,不能 break
    proxy.step(_batched_action(2))
    assert is_done() is True


def test_single_env_action_rejected():
    env = StubBatchEnv(num_envs=2, truncate_at=10)
    proxy = ParallelEvalProxy(env, num_envs=2)
    proxy.begin_wave(active_count=2)

    with pytest.raises(ValueError, match="not.*batch-ready|batch"):
        proxy.step(torch.zeros(1, 14))


def test_begin_wave_clears_previous_state():
    env = StubBatchEnv(num_envs=2, success_at={0: 1}, truncate_at=10)
    proxy = ParallelEvalProxy(env, num_envs=2)
    proxy.begin_wave(active_count=2)
    proxy.step(_batched_action(2))
    assert proxy.episode_result(0, max_env_steps=10)[0]

    proxy.begin_wave(active_count=1)
    assert proxy.wave_steps == 0
    assert not proxy.episode_result(0, max_env_steps=10)[0]
    assert not proxy.all_done()


def test_episode_plan_matches_sequential_draw():
    reference = np.random.RandomState(7)
    expected = [int(reference.randint(0, 2**31 - 1)) for _ in range(10)]

    plan = build_episode_plan(
        np.random.RandomState(7),
        max_episodes=10,
        num_shards=1,
        shard_index=0,
        fixed_episode_seed=None,
    )
    assert plan == list(enumerate(expected))


def test_episode_plan_sharding_preserves_seed_mapping():
    reference = np.random.RandomState(7)
    expected = [int(reference.randint(0, 2**31 - 1)) for _ in range(10)]

    merged = {}
    for shard_index in range(3):
        for episode, seed in build_episode_plan(
            np.random.RandomState(7), 10, 3, shard_index, None
        ):
            assert episode % 3 == shard_index
            merged[episode] = seed

    assert merged == dict(enumerate(expected))


def test_episode_plan_fixed_seed():
    plan = build_episode_plan(np.random.RandomState(0), 4, 1, 0, 123)
    assert [seed for _, seed in plan] == [123, 123, 123, 123]
