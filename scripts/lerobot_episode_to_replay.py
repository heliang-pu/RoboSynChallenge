"""Convert one LeRobot episode to RoboSynChallenge's legacy replay format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch


def convert_episode(dataset: Path, episode: int, output: Path) -> None:
    parquet_files = sorted((dataset / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset / 'data'}")

    rows = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        names = set(table.column_names)
        state_key = "observation.state" if "observation.state" in names else "observation.qpos"
        required = {state_key, "action", "episode_index", "frame_index"}
        if not required.issubset(names):
            continue
        selected = table.select(list(required)).to_pylist()
        rows.extend(row for row in selected if int(row["episode_index"]) == episode)

    if not rows:
        raise ValueError(f"Episode {episode} was not found in {dataset}")

    rows.sort(key=lambda row: int(row["frame_index"]))
    state_key = "observation.state" if "observation.state" in rows[0] else "observation.qpos"
    states = torch.tensor([row[state_key] for row in rows], dtype=torch.float32)
    actions = torch.tensor([row["action"] for row in rows], dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": states,
            "action": actions,
            "reward": torch.zeros(len(rows), dtype=torch.float32),
        },
        output,
    )
    print(f"Wrote episode {episode} ({len(rows)} frames) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    convert_episode(args.dataset, args.episode, args.output)


if __name__ == "__main__":
    main()
