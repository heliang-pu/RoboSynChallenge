#!/usr/bin/env python3
"""Render an expert-trajectory DM0.5 action replay evaluation as MP4."""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_video_frame(path: Path, frame_index: int) -> Image.Image:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate)
        time_base = float(stream.time_base)
        container.seek(int(frame_index / fps / time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_index = int(frame.pts * time_base * fps + 0.5)
            if current_index == frame_index:
                return Image.fromarray(frame.to_ndarray(format="rgb24"))
            if current_index > frame_index:
                break
    raise RuntimeError(f"Cannot read frame {frame_index} from {path}")


def image_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def query(server_url: str, row: dict, image_dir: Path, seed: int) -> tuple[np.ndarray, float]:
    images = {}
    for slot in range(1, 4):
        spec = row[f"images_{slot}"]
        images[str(slot)] = image_b64(
            read_video_frame(image_dir / spec["url"], int(spec["frame_idx"]))
        )
    payload = {
        "observation": {
            "prompt": row["prompt"],
            "state": row["state"],
            "images": images,
        },
        "sampling": {"seed": seed},
    }
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/infer",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode())
    return (
        np.asarray(result["actions"], dtype=np.float32),
        float(result.get("metadata", {}).get("latency_ms", -1)),
    )


def decode_frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            yield Image.fromarray(frame.to_ndarray(format="rgb24"))


def plot_actions(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    predicted: np.ndarray,
    expert: np.ndarray,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(100, 100, 100), width=1)
    lo, hi = -3.2, 3.2

    def points(values: np.ndarray):
        return [
            (
                left + 15 + i * (right - left - 30) / 13,
                bottom - 15 - (float(np.clip(v, lo, hi)) - lo) / (hi - lo) * (bottom - top - 30),
            )
            for i, v in enumerate(values)
        ]

    for value in (-3, 0, 3):
        y = bottom - 15 - (value - lo) / (hi - lo) * (bottom - top - 30)
        draw.line((left, y, right, y), fill=(55, 55, 55), width=1)
        draw.text((left + 3, y - 12), str(value), fill=(160, 160, 160))
    draw.line(points(expert), fill=(70, 220, 110), width=3)
    draw.line(points(predicted), fill=(255, 100, 90), width=3)
    for i, (x, _) in enumerate(points(expert)):
        draw.text((x - 4, bottom - 14), str(i), fill=(170, 170, 170))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-jsonl", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:7891")
    parser.add_argument("--replan-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.episode_jsonl.read_text().splitlines()]
    expert = np.asarray([row["action"] for row in rows], dtype=np.float32)
    predicted = np.zeros_like(expert)
    query_latencies = []
    query_frames = list(range(0, len(rows), args.replan_steps))
    for query_index, frame_index in enumerate(query_frames):
        chunk, latency = query(
            args.server_url,
            rows[frame_index],
            args.image_dir,
            args.seed + query_index,
        )
        length = min(args.replan_steps, len(rows) - frame_index, len(chunk))
        predicted[frame_index : frame_index + length] = chunk[:length]
        query_latencies.append(latency)
        print(
            f"query frame={frame_index} length={length} latency_ms={latency:.1f}",
            flush=True,
        )

    per_frame_mae = np.abs(predicted - expert).mean(axis=1)
    metrics = {
        "checkpoint": "checkpoint-400",
        "episode": args.episode_jsonl.stem,
        "frames": len(rows),
        "replan_steps": args.replan_steps,
        "num_queries": len(query_frames),
        "all_finite": bool(np.isfinite(predicted).all()),
        "action_mae": float(per_frame_mae.mean()),
        "action_mae_p95": float(np.quantile(per_frame_mae, 0.95)),
        "mean_query_latency_ms": float(np.mean(query_latencies)),
        "warm_latency_ms": float(np.mean(query_latencies[1:])),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")

    video_paths = [
        args.image_dir / rows[0][f"images_{slot}"]["url"] for slot in range(1, 4)
    ]
    decoders = [decode_frames(path) for path in video_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # H.264/yuv420p with the moov atom at the beginning is directly playable
    # by Chromium and VS Code's built-in video preview.
    output = av.open(str(args.output), mode="w", options={"movflags": "+faststart"})
    stream = output.add_stream(
        "libx264", rate=25, options={"preset": "fast", "crf": "22"}
    )
    stream.width = 1200
    stream.height = 720
    stream.pix_fmt = "yuv420p"
    font = ImageFont.load_default()
    labels = ["Head camera", "Left wrist", "Right wrist"]
    for frame_index, camera_frames in enumerate(zip(*decoders)):
        if frame_index >= len(rows):
            break
        canvas = Image.new("RGB", (1200, 720), (18, 18, 22))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (20, 14),
            "DM0.5 checkpoint-400 | expert-trajectory policy replay | sample_loading",
            fill=(245, 245, 245),
            font=font,
        )
        for slot, image in enumerate(camera_frames):
            resized = image.resize((380, 285), Image.Resampling.LANCZOS)
            x = 10 + slot * 395
            canvas.paste(resized, (x, 45))
            draw.text((x + 8, 52), labels[slot], fill=(255, 255, 255), font=font)
        mae = float(per_frame_mae[frame_index])
        query_frame = frame_index - frame_index % args.replan_steps
        draw.text(
            (20, 350),
            f"frame {frame_index + 1}/{len(rows)}  policy query frame={query_frame}  "
            f"action MAE={mae:.5f}  running MAE={per_frame_mae[:frame_index + 1].mean():.5f}",
            fill=(235, 235, 235),
            font=font,
        )
        draw.text((20, 375), "expert", fill=(70, 220, 110), font=font)
        draw.text((85, 375), "predicted", fill=(255, 100, 90), font=font)
        plot_actions(
            draw,
            (20, 400, 1180, 680),
            predicted[frame_index],
            expert[frame_index],
        )
        video_frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in stream.encode(video_frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    print("SUMMARY " + json.dumps(metrics), flush=True)
    print(f"VIDEO {args.output} bytes={args.output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
