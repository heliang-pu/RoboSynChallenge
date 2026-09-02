#!/usr/bin/env python3
"""Load one transformed PI0.5 training batch from every clean task."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


CHILD_CODE = r'''
import dataclasses
import json
import numpy as np
from openpi.training import config as c
from openpi.training import data_loader as d

cfg = dataclasses.replace(
    c.get_config("pi05_base_robosynchallenge_full"),
    batch_size=2,
    num_workers=0,
)
loader = d.create_data_loader(
    cfg,
    skip_norm_stats=True,
    num_batches=1,
    shuffle=False,
    framework="pytorch",
)
observation, action = next(iter(loader))
action_array = np.asarray(action)
state_array = np.asarray(observation.state)
assert action_array.shape == (2, 50, 32), action_array.shape
assert state_array.shape == (2, 32), state_array.shape
assert np.isfinite(action_array).all() and np.isfinite(state_array).all()
assert set(observation.images) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
image_shapes = {key: list(np.asarray(value).shape) for key, value in observation.images.items()}
assert all(shape == [2, 3, 224, 224] for shape in image_shapes.values()), image_shapes
assert tuple(observation.tokenized_prompt.shape) == (2, 200)
print("RESULT=" + json.dumps({
    "action": list(action_array.shape),
    "state": list(state_array.shape),
    "images": image_shapes,
    "prompt": list(observation.tokenized_prompt.shape),
}))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tasks = sorted(
        path.name
        for path in args.clean_root.iterdir()
        if (path / "meta" / "info.json").is_file()
    )
    temporary = Path(tempfile.mkdtemp(prefix="robosyn-clean-pi05-"))
    output = {"passed": False, "tasks": {}}
    try:
        hub_root = temporary / "RoboSynChallenge"
        hub_root.mkdir()
        for task in tasks:
            (hub_root / task).symlink_to((args.clean_root / task).resolve(), target_is_directory=True)
        for task in tasks:
            print(f"PI0.5 transformed batch: {task}", flush=True)
            environment = os.environ.copy()
            environment.update(
                HF_LEROBOT_HOME=str(temporary),
                HF_HOME=str(temporary / "hf" / task),
                HF_DATASETS_CACHE=str(temporary / "hf" / task / "datasets"),
                ROBOSYN_REPO_ID=f"RoboSynChallenge/{task}",
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
            )
            result = subprocess.run(
                [str(args.runtime_python), "-c", CHILD_CODE],
                cwd=args.policy_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
            )
            lines = result.stdout.splitlines()
            payload_lines = [line for line in lines if line.startswith("RESULT=")]
            if result.returncode != 0 or len(payload_lines) != 1:
                raise RuntimeError(
                    f"PI0.5 loader failed for {task}, exit={result.returncode}:\n"
                    + "\n".join(lines[-40:])
                )
            output["tasks"][task] = json.loads(payload_lines[0].removeprefix("RESULT="))
        output["passed"] = True
        return 0
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        shutil.rmtree(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
