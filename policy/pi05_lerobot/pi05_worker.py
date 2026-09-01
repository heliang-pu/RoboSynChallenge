#!/usr/bin/env python
# ----------------------------------------------------------------------------
# LeRobot PI0.5 (PyTorch) inference worker.
#
# Runs in its own interpreter so the simulator env does not have to satisfy
# LeRobot's transformers/torch pins. Speaks newline-delimited JSON on stdio:
#   {"cmd": "info"}                              -> checkpoint capabilities
#   {"cmd": "reset"}                             -> clear action + MEM queues
#   {"cmd": "select_action", "obs_path": ...}    -> one action, MEM-aligned
#   {"cmd": "predict_chunk", "obs_path": ..., "n_action_steps": N} -> N actions
# ----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--lerobot-root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tokenizer",
        default="",
        help=(
            "Override the tokenizer the preprocessor was saved with. Checkpoints "
            "trained elsewhere often store an absolute path that does not exist here; "
            "'google/paligemma-3b-pt-224' is the stock PI0.5 tokenizer."
        ),
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=0,
        help=(
            "Override PI05Config.n_action_steps, i.e. how many actions are executed "
            "before the model replans. 0 keeps the checkpoint value."
        ),
    )
    parser.add_argument(
        "--memory-stride",
        type=int,
        default=0,
        help=(
            "Override PI05Config.memory_stride, in dataset frames. 0 keeps the "
            "checkpoint value. Only meaningful when the checkpoint enables MEM."
        ),
    )
    return parser.parse_args()


def _load_runtime(args: argparse.Namespace):
    if args.lerobot_root:
        lerobot_src = Path(args.lerobot_root).expanduser().resolve() / "src"
        if lerobot_src.exists():
            sys.path.insert(0, str(lerobot_src))
        else:
            raise FileNotFoundError(
                f"LeRobot source path does not exist: {lerobot_src}. "
                "Install lerobot in this Python environment or set PI05_LEROBOT_ROOT."
            )

    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import lerobot.policies.pi05. Install LeRobot (>= the commit that "
            "added MEM, see policy/pi05_lerobot/README.md) in the worker Python, or run "
            "policy/pi05_lerobot/setup_lerobot.sh and set PI05_LEROBOT_ROOT."
        ) from exc

    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    policy = PI05Policy.from_pretrained(checkpoint_dir)

    if args.n_action_steps > 0:
        if args.n_action_steps > policy.config.chunk_size:
            raise ValueError(
                f"n_action_steps ({args.n_action_steps}) cannot exceed the checkpoint "
                f"chunk_size ({policy.config.chunk_size})"
            )
        policy.config.n_action_steps = args.n_action_steps

    if args.memory_stride > 0:
        # MEM 的 stride 以「数据集帧」计。训练集是 25fps 而 MEM 默认 30,
        # 两者对不上时这里直接改配置比重导出 checkpoint 便宜。改完必须
        # reset(),因为环形缓冲的长度是由 stride 算出来的。
        policy.config.memory_stride = args.memory_stride

    policy.to(args.device)
    policy.eval()
    policy.reset()

    overrides = {"device_processor": {"device": args.device}}
    if args.tokenizer:
        overrides["tokenizer_processor"] = {"tokenizer_name": args.tokenizer}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides=overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def _describe(policy) -> dict:
    config = policy.config
    use_visual_memory = bool(getattr(config, "use_visual_memory", False))
    use_proprioceptive_memory = bool(getattr(config, "use_proprioceptive_memory", False))
    return {
        "chunk_size": int(config.chunk_size),
        "n_action_steps": int(config.n_action_steps),
        "image_features": list(config.image_features),
        "use_visual_memory": use_visual_memory,
        "use_proprioceptive_memory": use_proprioceptive_memory,
        "memory_enabled": use_visual_memory or use_proprioceptive_memory,
        "memory_frames": int(getattr(config, "memory_frames", 0)),
        "memory_stride": int(getattr(config, "memory_stride", 0)),
        "rtc_enabled": bool(policy._rtc_enabled()),
        # Observations pushed into the MEM ring buffer since the last reset. One
        # per env step is what training assumed; anything less means the history
        # is stretched.
        "memory_steps_seen": int(getattr(policy, "_memory_steps_seen", 0)),
        "queued_actions": len(getattr(policy, "_action_queue", ())),
        "use_relative_actions": bool(getattr(config, "use_relative_actions", False)),
    }


def _load_obs(obs_path: str) -> tuple[dict[str, np.ndarray], str]:
    with np.load(obs_path, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files if key != "task"}
        task = str(data["task"].item()) if "task" in data else ""
    return obs, task


def _to_batch(obs: dict[str, np.ndarray], task: str) -> dict:
    batch = {}
    for key, value in obs.items():
        tensor = torch.from_numpy(value)
        if tensor.dtype == torch.uint8:
            # PI05 `_preprocess_images` assumes [0, 1] floats before mapping to
            # [-1, 1] for SigLIP, and the resize op has no Byte kernel anyway.
            tensor = tensor.to(torch.float32).div_(255.0)
        batch[key] = tensor
    batch["task"] = task
    return batch


def _action_to_list(action: torch.Tensor) -> list[float]:
    return action.squeeze(0).cpu().numpy().astype(np.float32).tolist()


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    # from_pretrained prints an OpenPI disclaimer on stdout; stdout is the RPC
    # channel, so every library print is diverted to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        policy, preprocessor, postprocessor = _load_runtime(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        cmd = request.get("cmd")

        try:
            if cmd == "info":
                _send({"ok": True, "info": _describe(policy)})
                continue

            if cmd == "reset":
                policy.reset()
                _send({"ok": True})
                continue

            if cmd not in ("select_action", "predict_chunk"):
                raise ValueError(f"Unsupported command: {cmd}")

            with contextlib.redirect_stdout(sys.stderr):
                obs, task = _load_obs(request["obs_path"])
                proc_obs = preprocessor(_to_batch(obs, task))

                if cmd == "select_action":
                    # An empty action queue means this call runs the model; a
                    # non-empty one only pops a cached step. The caller needs the
                    # distinction to keep inference-time stats honest.
                    planned = len(policy._action_queue) == 0
                    action = postprocessor(policy.select_action(proc_obs))
                    actions = [_action_to_list(action)]
                else:
                    planned = True
                    num_actions = int(
                        request.get("n_action_steps", policy.config.n_action_steps)
                    )
                    num_actions = max(1, min(num_actions, policy.config.chunk_size))
                    chunk = policy.predict_action_chunk(proc_obs)[:, :num_actions]
                    actions = [
                        _action_to_list(postprocessor(chunk[:, i]))
                        for i in range(chunk.shape[1])
                    ]

            _send({"ok": True, "actions": actions, "planned": planned})
        except Exception as exc:  # noqa: BLE001
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
