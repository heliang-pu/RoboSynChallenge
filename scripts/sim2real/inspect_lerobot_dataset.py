"""Print what a RoboSynChallenge-style LeRobot dataset actually contains.

Use it on your simulated and real datasets before mixing them. It reports the
frame rate, image features and per-dimension ranges of the state and action
vectors, and it flags the two mistakes that break sim + real co-training most
often: a gripper stored in a different unit, and joints stored in degrees.

    python scripts/sim2real/inspect_lerobot_dataset.py \
        lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell \
        lerobot_dataset/piper_real/click_bell

Only numpy, pandas and pyarrow are needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

DEFAULT_KEYS = ("observation.qpos", "action")
DEFAULT_GRIPPER_DIMS = (6, 13)
# Largest PiPER joint range in radians is about 3.14, so anything well above it is not radians.
MAX_PLAUSIBLE_RAD = 3.3


def load_info(root: pathlib.Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found. Is {root} a LeRobot dataset root?"
        )
    return json.loads(info_path.read_text())


def load_columns(
    root: pathlib.Path, keys: tuple[str, ...], max_files: int | None
) -> dict[str, np.ndarray]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}.")
    if max_files is not None:
        files = files[:max_files]
    frames = [
        pd.read_parquet(path, columns=[k for k in keys if _has_column(path, k)])
        for path in files
    ]
    table = pd.concat(frames, ignore_index=True)
    columns = {}
    for key in keys:
        if key in table:
            columns[key] = np.stack(
                [np.asarray(v, dtype=np.float64) for v in table[key].to_numpy()]
            )
    return columns


def _has_column(path: pathlib.Path, key: str) -> bool:
    import pyarrow.parquet as pq

    return key in pq.read_schema(path).names


def describe_gripper(values: np.ndarray) -> str:
    high = float(np.percentile(values, 99))
    low = float(np.percentile(values, 1))
    if low >= -0.01 and high <= 0.12:
        # Normalized data whose gripper never opens past 12 % would also land here.
        return "probably meters of opening"
    if low >= -0.05 and high <= 1.05:
        return "normalized 0-1"
    if high <= 120:
        return "probably millimeters"
    return "probably 0.001 mm or another raw unit"


def inspect(
    root: pathlib.Path,
    keys: tuple[str, ...],
    gripper_dims: tuple[int, ...],
    max_files: int | None,
) -> list[str]:
    """Print a report for one dataset and return the warnings."""
    info = load_info(root)
    warnings = []
    print(f"\n=== {root}")
    print(
        f"codebase {info.get('codebase_version')} | robot {info.get('robot_type')} | fps {info.get('fps')} | "
        f"episodes {info.get('total_episodes')} | frames {info.get('total_frames')}"
    )
    features = info.get("features", {})
    for name, feature in features.items():
        if feature.get("dtype") in ("video", "image"):
            print(f"image  {name}: {feature.get('shape')} ({feature.get('dtype')})")

    columns = load_columns(root, keys, max_files)
    for key in keys:
        if key not in columns:
            warnings.append(f"{root.name}: key {key!r} is missing.")
            continue
        data = columns[key]
        names = features.get(key, {}).get("names") or [
            str(i) for i in range(data.shape[1])
        ]
        if isinstance(names, dict):
            names = next(iter(names.values()))
        print(f"\n{key}: {data.shape[0]} frames x {data.shape[1]} dims")
        print(
            f"{'dim':>4} {'name':<22} {'min':>9} {'p1':>9} {'p50':>9} {'p99':>9} {'max':>9}"
        )
        not_radians = []
        for dim in range(data.shape[1]):
            col = data[:, dim]
            name = str(names[dim]) if dim < len(names) else str(dim)
            p1, p50, p99 = np.percentile(col, [1, 50, 99])
            note = ""
            if dim in gripper_dims:
                note = f"  gripper: {describe_gripper(col)}"
            elif np.max(np.abs(col)) > MAX_PLAUSIBLE_RAD:
                note = "  not radians?"
                not_radians.append(dim)
            print(
                f"{dim:>4} {name:<22} {col.min():>9.4f} {p1:>9.4f} {p50:>9.4f} {p99:>9.4f} {col.max():>9.4f}{note}"
            )
        if not_radians:
            warnings.append(
                f"{root.name}: {key} dims {not_radians} exceed {MAX_PLAUSIBLE_RAD} and are not radian angles."
            )
        for dim in gripper_dims:
            if (
                dim < data.shape[1]
                and describe_gripper(data[:, dim]) != "normalized 0-1"
            ):
                warnings.append(
                    f"{root.name}: {key}[{dim}] gripper values look like {describe_gripper(data[:, dim])}, but the RoboSynChallenge "
                    "convention is normalized 0-1."
                )
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=pathlib.Path,
        help="LeRobot dataset root directories.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=list(DEFAULT_KEYS),
        help="Vector features to report.",
    )
    parser.add_argument(
        "--gripper_dims", nargs="+", type=int, default=list(DEFAULT_GRIPPER_DIMS)
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Read only the first N parquet files.",
    )
    args = parser.parse_args(argv)

    warnings = []
    fps = {}
    for root in args.datasets:
        warnings += inspect(
            root, tuple(args.keys), tuple(args.gripper_dims), args.max_files
        )
        fps[str(root)] = load_info(root).get("fps")
    if len(set(fps.values())) > 1:
        warnings.append(
            f"Datasets use different frame rates: {fps}. Resample them to one rate before mixing."
        )

    print(
        "\nGripper scale for mixing: simulated V100 values (1.0 = 100 mm) need gripper_scale = 100 / 70"
    )
    print(
        "to match a 70 mm PiPER gripper. Real data already normalized to 70 mm needs gripper_scale = 1.0."
    )
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
        return 1
    print("\nNo problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
