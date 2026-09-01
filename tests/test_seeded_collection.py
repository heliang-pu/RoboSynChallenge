from types import MethodType

from scripts.seeded_collection import SeededCollection


def test_saved_episode_uses_explicit_save_decision_not_reset_verdict():
    wrapper = SeededCollection.__new__(SeededCollection)
    wrapper._env = type("Env", (), {"reset": lambda self, **kwargs: (None, {})})()
    wrapper._current_scene_seed = 123
    wrapper._rng = __import__("numpy").random.RandomState(1)
    wrapper._records = []
    wrapper._last_saved_count = 0
    wrapper._sidecar_written = False
    wrapper._elapsed_steps_snapshot = MethodType(lambda self: 42, wrapper)
    wrapper._saved_episode_count = MethodType(lambda self: 1, wrapper)
    wrapper._official_success_verdict = MethodType(lambda self: False, wrapper)

    wrapper.reset(options={"save_data": True})

    assert wrapper._records == [
        {"episode_index": 0, "seed": 123, "success": True, "env_steps": 42}
    ]
