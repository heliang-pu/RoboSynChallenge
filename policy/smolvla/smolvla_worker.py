#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--lerobot-root", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_runtime(args: argparse.Namespace):
    if args.lerobot_root:
        lerobot_src = Path(args.lerobot_root).expanduser().resolve() / "src"
        if lerobot_src.exists():
            sys.path.insert(0, str(lerobot_src))
        else:
            raise FileNotFoundError(
                f"LeRobot source path does not exist: {lerobot_src}. "
                "Install lerobot in this Python environment or set SMOLVLA_LEROBOT_ROOT."
            )

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import lerobot. Install LeRobot in the SmolVLA Python env, "
            "or run policy/smolvla/setup_lerobot.sh and set SMOLVLA_LEROBOT_ROOT."
        ) from exc

    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir)
    policy.to(args.device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
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


def _load_obs(obs_path: str) -> tuple[dict[str, np.ndarray], str]:
    with np.load(obs_path, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files if key != "task"}
        task = str(data["task"].item()) if "task" in data else ""
    return obs, task


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy, preprocessor, postprocessor = _load_runtime(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        cmd = request.get("cmd")

        try:
            if cmd == "reset":
                policy.reset()
                _send({"ok": True})
                continue

            if cmd != "infer":
                raise ValueError(f"Unsupported command: {cmd}")

            with contextlib.redirect_stdout(sys.stderr):
                # 实验开关:每次请求都清空 lerobot 的动作队列,让 n_action_steps 个
                # 动作全部来自对「本次观测」的重新推理(真闭环),而不是弹旧 chunk。
                if os.environ.get("SMOLVLA_REINFER_EACH_REQUEST") == "1":
                    policy.reset()
                obs, task = _load_obs(request["obs_path"])
                # 仿真侧传来的图像是 uint8。新版 lerobot 的预处理管线会先做
                # resize(upsample_bilinear2d),该算子不支持 Byte,需先转 float
                # 并归一到 [0,1](等价于旧版归一化器对 uint8 的处理)。
                _tensors = {}
                for key, value in obs.items():
                    t = torch.from_numpy(value)
                    if t.dtype == torch.uint8:
                        t = t.to(torch.float32).div_(255.0)
                    _tensors[key] = t
                obs = _tensors
                obs["task"] = task
                proc_obs = preprocessor(obs)

                num_actions = int(request.get("n_action_steps", 1))
                actions = []
                for _ in range(num_actions):
                    action = policy.select_action(proc_obs)
                    action = postprocessor(action)
                    actions.append(action.squeeze(0).cpu().numpy().astype(np.float32).tolist())

            _send({"ok": True, "actions": actions})
        except Exception as exc:  # noqa: BLE001
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
