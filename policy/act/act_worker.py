#!/usr/bin/env python
"""ACT inference worker pinned to the training-time LeRobot environment."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def load_policy(checkpoint_dir, device):
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    policy.to(device)
    policy.eval()
    return policy


def load_batch(obs_path, policy, device):
    batch = {}
    with np.load(obs_path, allow_pickle=False) as data:
        state = torch.from_numpy(data["observation.state"]).to(
            device=device, dtype=torch.float32
        )
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch["observation.state"] = state

        for image_key in policy.config.image_features:
            image = torch.from_numpy(data[image_key])
            if image.ndim == 3:
                if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                    image = image[:3]
                else:
                    image = image[..., :3].permute(2, 0, 1)
                image = image.unsqueeze(0)
            elif image.ndim == 4 and image.shape[-1] in (1, 3, 4):
                image = image[..., :3].permute(0, 3, 1, 2)
            image = image.contiguous().to(device=device, dtype=torch.float32)
            if torch.max(image) > 1.5:
                image = image / 255.0
            batch[image_key] = image
    return batch


def main():
    args = parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy = load_policy(args.checkpoint_dir, args.device)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                policy.reset()
                send({"ok": True})
                continue
            if command != "infer":
                raise ValueError(f"Unsupported command: {command}")

            batch = load_batch(request["obs_path"], policy, args.device)
            num_actions = int(request.get("n_action_steps", 1))
            if not 0 < num_actions <= int(policy.config.chunk_size):
                raise ValueError(
                    f"n_action_steps must be in [1, {policy.config.chunk_size}], "
                    f"got {num_actions}"
                )
            with torch.inference_mode():
                # Predict a fresh chunk from the current observation on every
                # RPC. Taking only the first H actions implements a genuine
                # receding-horizon controller (H=10 replans after 10 env steps
                # instead of silently consuming the remaining 40 cached
                # actions from a 50-step chunk).
                policy.reset()
                action_chunk = policy.predict_action_chunk(batch)[0, :num_actions]
                actions = action_chunk.detach().cpu().numpy().astype(np.float32).tolist()
            send({"ok": True, "actions": actions})
        except Exception as exc:  # noqa: BLE001
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
