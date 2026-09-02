#!/usr/bin/env python3
"""Run a small offline DM0.5 inference evaluation on LeRobot expert episodes.

The inference service must already be running.  This script selects evenly
spaced episodes, evaluates their middle frames, and compares the predicted
absolute action chunk with the recorded expert chunk.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

import av
from PIL import Image
import numpy as np


def encode_video_frame(path: Path, frame_index: int) -> str:
    # Match OpenDM's training input path exactly.  These videos are AV1; the
    # imageio/OpenCV builds in the DCU image cannot decode them, while PyAV can.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate)
        time_base = float(stream.time_base)
        container.seek(int(frame_index / fps / time_base), stream=stream)
        image = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_index = int(frame.pts * time_base * fps + 0.5)
            if current_index == frame_index:
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                break
            if current_index > frame_index:
                break
    if image is None:
        raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=95)
    return base64.b64encode(encoded.getvalue()).decode("ascii")


def infer(server_url: str, observation: dict, seed: int, timeout: float) -> dict:
    payload = {"observation": observation, "sampling": {"seed": seed}}
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/infer",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode())
    result["client_latency_ms"] = (time.perf_counter() - started) * 1000
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:7891")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    episodes = sorted(args.jsonl_dir.glob("episode_*.jsonl"))
    if not episodes:
        raise FileNotFoundError(f"No episode JSONL files in {args.jsonl_dir}")
    indices = np.linspace(0, len(episodes) - 1, args.num_samples, dtype=int)

    samples = []
    for sample_index, episode_index in enumerate(indices):
        episode_path = episodes[int(episode_index)]
        rows = [json.loads(line) for line in episode_path.read_text().splitlines()]
        frame_index = len(rows) // 2
        row = rows[frame_index]
        images = {}
        for slot in range(1, 4):
            image = row[f"images_{slot}"]
            images[str(slot)] = encode_video_frame(
                args.image_dir / image["url"], int(image["frame_idx"])
            )

        result = infer(
            args.server_url,
            {
                "prompt": row["prompt"],
                "state": row["state"],
                "images": images,
            },
            seed=args.seed + sample_index,
            timeout=args.timeout,
        )
        predicted = np.asarray(result["actions"], dtype=np.float32)
        expert_rows = rows[frame_index : frame_index + args.chunk_size]
        expert = np.asarray([item["action"] for item in expert_rows], dtype=np.float32)
        if len(expert) < args.chunk_size:
            expert = np.concatenate(
                [expert, np.repeat(expert[-1:], args.chunk_size - len(expert), axis=0)]
            )
        compared_steps = min(len(predicted), len(expert))
        error = np.abs(predicted[:compared_steps] - expert[:compared_steps])
        record = {
            "episode": episode_path.stem,
            "frame": frame_index,
            "shape": list(predicted.shape),
            "finite": bool(np.isfinite(predicted).all()),
            "first_action_mae": float(error[0].mean()),
            "chunk_mae": float(error.mean()),
            "pred_abs_max": float(np.abs(predicted).max()),
            "server_latency_ms": float(result.get("metadata", {}).get("latency_ms", -1)),
            "client_latency_ms": float(result["client_latency_ms"]),
        }
        samples.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "num_samples": len(samples),
        "all_finite": all(item["finite"] for item in samples),
        "all_shapes_50x14": all(item["shape"] == [50, 14] for item in samples),
        "mean_first_action_mae": float(
            np.mean([item["first_action_mae"] for item in samples])
        ),
        "mean_chunk_mae": float(np.mean([item["chunk_mae"] for item in samples])),
        "mean_server_latency_ms": float(
            np.mean([item["server_latency_ms"] for item in samples])
        ),
        "samples": samples,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
