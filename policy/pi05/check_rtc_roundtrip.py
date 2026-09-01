"""The decisive check: model-space <-> env-space must round-trip.

The RTC guidance target is an old plan expressed as absolute environment
actions. If `to_model_action_space` is not the exact inverse of the policy's
output chain, the target lands in the wrong frame and RTC steers the arm
somewhere it was never meant to go.
"""
import sys
import numpy as np

sys.path.insert(0, "policy/pi05")
sys.path.insert(0, "policy/pi05/src")
sys.path.insert(0, "policy/pi05/packages/openpi-client/src")

from pi_model import PI0

model = PI0(train_config_name="pi05_base_robosynchallenge_full", model_name="mixer_operating",
            checkpoint_id=28000, pi0_step=10, pytorch_device="cpu")

rng = np.random.default_rng(0)
def make_obs(state):
    return {
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": state.astype(np.float32),
        "prompt": "Pick the beaker, place it on the mixer, then flip the toggle switch.",
    }

state_a = rng.normal(size=14).astype(np.float32)
model.set_language("Pick the beaker, place it on the mixer, then flip the toggle switch.")
model.observation_window = make_obs(state_a)

noise = rng.normal(size=(50, 32)).astype(np.float32)
env_actions = model.policy.infer(model.observation_window, noise=noise)["actions"]
print("env actions", env_actions.shape, "range", env_actions.min(), env_actions.max())

# Round-trip at the SAME state: to_model_action_space must invert the output chain.
back = model.to_model_action_space(env_actions)
regen = model.policy.infer(model.observation_window, noise=noise, prev_chunk=back,
                           prefix_weights=np.ones(50, np.float32),
                           max_guidance_weight=10.0, rtc_correction="identity")["actions"]
err_same = np.abs(regen - env_actions).max()
print(f"\nA. hard-pin to its own output, same state: max |regen - env| = {err_same:.4f}")

# Now the case that was broken: the plan was made at state_a, we are now at state_b.
state_b = state_a + rng.normal(scale=0.1, size=14).astype(np.float32)
model.observation_window = make_obs(state_b)
rebased = model.to_model_action_space(env_actions)      # rebased onto state_b

regen_b = model.policy.infer(model.observation_window, noise=noise, prev_chunk=rebased,
                             prefix_weights=np.ones(50, np.float32),
                             max_guidance_weight=10.0, rtc_correction="identity")["actions"]
err_rebased = np.abs(regen_b - env_actions).max()
print(f"B. hard-pin to a plan from another state, REBASED:  max |regen - target| = {err_rebased:.4f}")

# What the buggy version did: feed the model-space chunk computed at state_a.

model.observation_window = make_obs(state_a)
stale_target = model.to_model_action_space(env_actions)
model.observation_window = make_obs(state_b)
regen_stale = model.policy.infer(model.observation_window, noise=noise, prev_chunk=stale_target,
                                 prefix_weights=np.ones(50, np.float32),
                                 max_guidance_weight=10.0, rtc_correction="identity")["actions"]
err_stale = np.abs(regen_stale - env_actions).max()
print(f"C. same but target built against the STALE state:   max |regen - target| = {err_stale:.4f}")
print(f"\n   state shift magnitude: {np.abs(state_b - state_a).max():.4f}")
print(f"   rebasing reduces the pinning error by {err_stale / max(err_rebased,1e-9):.1f}x")
