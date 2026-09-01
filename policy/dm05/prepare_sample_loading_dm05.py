"""Convert a LeRobot v2.1 sample_loading dataset to OpenDM JSONL episodes.

The converter keeps the original MP4 files and writes one JSON object per
frame. OpenDM then reads the requested video frame lazily during training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


PROMPT = "Pick up the test tube, and move it to the other arm, and insert it into the rack."


def convert(dataset_dir: Path, output_dir: Path) -> None:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    video_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'data'}")

    index: dict[str, int] = {}
    for parquet_path in files:
        table = pq.read_table(parquet_path, columns=["frame_index", "observation.state", "action"])
        rows = table.to_pylist()
        episode_index = int(parquet_path.stem.split("_")[-1])
        chunk = episode_index // int(info.get("chunks_size", 1000))
        # The v2.1 metadata uses this exact path convention for videos.
        video_urls = {
            f"images_{i + 1}": (
                f"videos/chunk-{chunk:03d}/{key}/episode_{episode_index:06d}.mp4"
            )
            for i, key in enumerate(video_keys)
        }
        out_path = output_dir / f"episode_{episode_index:06d}.jsonl"
        with out_path.open("w") as out:
            for row in rows:
                frame_index = int(row["frame_index"])
                item = {
                    name: {"type": "video", "url": url, "frame_idx": frame_index}
                    for name, url in video_urls.items()
                }
                item.update(
                    {
                        "state": [float(x) for x in row["observation.state"]],
                        "action": [float(x) for x in row["action"]],
                        "prompt": PROMPT,
                        "is_robot": True,
                    }
                )
                out.write(json.dumps(item, separators=(",", ":")) + "\n")
        index[str(out_path)] = len(rows)

    (output_dir / "index_cache.json").write_text(
        json.dumps({"data": index}, indent=2) + "\n"
    )
    print(f"Converted {len(files)} episodes / {sum(index.values())} frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    convert(args.dataset_dir, args.output_dir)
