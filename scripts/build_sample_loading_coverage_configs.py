#!/usr/bin/env python3
"""Build collision-constrained sample_loading coverage collection configs."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

DEFAULT_BASE_CONFIG = Path("configs/sample_loading/random/gym_config.json")
DEFAULT_COVERAGE_SUMMARY = Path(
    "report/sample_loading_random_coverage/coverage_summary.json"
)
DEFAULT_CONFIG_ROOT = Path("configs/sample_loading")
DEFAULT_SAVE_ROOT = Path(
    os.environ.get(
        "ROBOSYN_SAVE_ROOT",
        str(Path.home() / "FermiBotNas/dataset/RoboSynChallenge/Syn"),
    )
) / "sample_loading_coverage"

# Conservative local-frame half extents. These are slightly larger than the
# scaled mesh AABBs so that the 3 cm requested clearance is never optimistic.
TUBE_HALF_EXTENTS = [0.00923, 0.00913, 0.12677]
RACK_HALF_EXTENTS = [0.15139, 0.06390, 0.04362]


def constrained_pair_event(recommendation: dict) -> dict:
    return {
        "func": "randomize_rigid_object_pair_pose_constrained",
        "mode": "reset",
        "params": {
            "first_entity_cfg": {"uid": "cube"},
            "second_entity_cfg": {"uid": "rack"},
            "first_position_range": recommendation["tube_position_range"],
            "second_position_range": recommendation["rack_position_range"],
            "first_rotation_range": recommendation["tube_rotation_range"],
            "second_rotation_range": recommendation["rack_rotation_range"],
            "first_half_extents": TUBE_HALF_EXTENTS,
            "second_half_extents": RACK_HALF_EXTENTS,
            "first_relative_position": False,
            "second_relative_position": False,
            "first_relative_rotation": True,
            "second_relative_rotation": True,
            "min_xy_clearance": 0.03,
            "max_xy_center_distance": 0.38,
            "max_resample_attempts": 256,
            "physics_update_step": 1,
        },
    }


def replace_spatial_events(events: dict, recommendation: dict) -> dict:
    output = {}
    inserted = False
    for name, event in events.items():
        if name in {"random_cube_pose", "random_rack_pose"}:
            if not inserted:
                output["randomize_tube_rack_pose_constrained"] = constrained_pair_event(
                    recommendation
                )
                inserted = True
            continue
        output[name] = event
    if not inserted:
        raise KeyError("Base config has no random_cube_pose/random_rack_pose events")
    return output


def build_config(base: dict, recommendation: dict, save_root: Path) -> dict:
    config = copy.deepcopy(base)
    name = recommendation["name"]
    config["max_episodes"] = int(recommendation["episodes"])
    config["env"]["events"] = replace_spatial_events(
        config["env"]["events"], recommendation
    )

    recorder = config["env"]["dataset"]["lerobot"]["params"]
    recorder["save_path"] = str(save_root / name)
    recorder.setdefault("extra", {})["coverage_group"] = name
    recorder["extra"]["coverage_reason"] = recommendation["reason"]
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--coverage-summary", type=Path, default=DEFAULT_COVERAGE_SUMMARY
    )
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    args = parser.parse_args()

    base = json.loads(args.base_config.read_text())
    summary = json.loads(args.coverage_summary.read_text())
    recommendations = summary["recommendations"]
    if not recommendations:
        raise ValueError("Coverage summary contains no recommendations")

    for recommendation in recommendations:
        name = recommendation["name"]
        output_path = args.config_root / name / "gym_config.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config = build_config(base, recommendation, args.save_root)
        output_path.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n")
        print(f"wrote {output_path} ({recommendation['episodes']} episodes)")


if __name__ == "__main__":
    main()
