#!/usr/bin/env python3
"""Turn an RTC matrix run into a report.

Every cell replays the same episode seeds, so outcomes are *paired*: for a given
seed we know whether each configuration succeeded.  That lets us use McNemar's
test on the discordant pairs, which is far more powerful at n=20 than treating
the two arms as independent samples.  Per-episode outcomes are recovered from the
recorded video filenames (`episode_NNN_seed_SEED_{success,fail}.mp4`).

    python policy/pi05/summarize_rtc_matrix.py report/rtc_matrix/<timestamp>
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_RE = re.compile(r"episode_(\d+)_seed_(-?\d+)_(success|fail)\.mp4$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# "  Episode 7: SUCCESS; action_steps=312/500; inference=0.061430s over 32 calls"
EPISODE_RE = re.compile(
    r"Episode (\d+): (SUCCESS|FAIL); action_steps=(\d+)/(\d+); "
    r"inference=([\d.]+)s over (\d+) calls"
)


def parse_log(log_path):
    """Recover per-episode results from a cell log.

    `evaluation_metrics.json` is not reliable here: EmbodiChain's `env.close()`
    tears the process down before the summary is written, so the run ends with
    exit status 0 and no metrics file even though every episode completed. The
    per-episode lines and the recorded videos survive, and together they carry
    everything the summary would have.
    """
    path = Path(log_path)
    if not path.exists():
        return None
    text = ANSI_RE.sub("", path.read_text(errors="replace"))
    episodes = [
        {"index": int(i), "success": status == "SUCCESS", "steps": int(steps),
         "max_steps": int(cap), "infer_s": float(infer), "calls": int(calls)}
        for i, status, steps, cap, infer, calls in EPISODE_RE.findall(text)
    ]
    if not episodes:
        return None
    return {
        "episode_count": len(episodes),
        "success_count": sum(e["success"] for e in episodes),
        "success_rate": sum(e["success"] for e in episodes) / len(episodes),
        "average_action_steps": sum(e["steps"] for e in episodes) / len(episodes),
        # The first call of a run pays XLA compilation; it would swamp the mean.
        "average_inference_time_seconds": (
            sorted(e["infer_s"] for e in episodes)[len(episodes) // 2]
        ),
        "episodes": episodes,
    }


def episode_outcomes(metrics_path):
    """seed -> bool, recovered from the run's recorded videos."""
    if not metrics_path:
        return {}
    videos = (REPO_ROOT / metrics_path).parent / "videos"
    if not videos.is_dir():
        return {}
    out = {}
    for f in videos.iterdir():
        m = VIDEO_RE.search(f.name)
        if m:
            out[int(m.group(2))] = m.group(3) == "success"
    return out


def mcnemar(a, b):
    """Paired comparison of two seed->success maps. Returns (a_only, b_only, p)."""
    shared = set(a) & set(b)
    a_only = sum(1 for s in shared if a[s] and not b[s])
    b_only = sum(1 for s in shared if b[s] and not a[s])
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return a_only, b_only, p


def load(run_dirs):
    """Merge one or more matrix runs. Shards write disjoint task sets."""
    data, results = None, []
    for d in run_dirs:
        shard = json.loads((d / "results.json").read_text())
        data = data or shard
        results.extend(shard["results"])
    data = dict(data, results=results)
    cells = {}
    for r in data["results"]:
        summary = (r.get("metrics") or {}).get("summary") or parse_log(r["log"]) or {}
        outcomes = episode_outcomes(r.get("metrics_path"))
        if not outcomes and summary.get("episodes"):
            # No videos (or a different run dir): fall back to episode order,
            # which is still paired because every cell replays the same seeds.
            outcomes = {e["index"]: e["success"] for e in summary["episodes"]}
        elif outcomes and len(outcomes) > summary.get("episode_count", 0):
            # The teardown crash swallows the last episode's log line but its
            # video is already on disk, so the videos are the complete record.
            summary = dict(summary)
            summary["episode_count"] = len(outcomes)
            summary["success_count"] = sum(outcomes.values())
            summary["success_rate"] = sum(outcomes.values()) / len(outcomes)
        cells[(r["task"], r["cell"]["name"])] = {
            "result": r, "summary": summary, "outcomes": outcomes,
        }
    return data, cells


def rate(summary):
    if not summary:
        return "—"
    return f"{summary['success_count']}/{summary['episode_count']} = {summary['success_rate']:.0%}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, nargs="+", help="one or more shard directories")
    p.add_argument("--out", default="report.md")
    args = p.parse_args()

    run_dirs = [d if d.is_absolute() else REPO_ROOT / d for d in args.run_dir]
    data, cells = load(run_dirs)
    run_dir = run_dirs[0]
    horizons = data["args"]["horizons"]
    delay = data["args"]["delay"]

    lines = [f"# RTC / async inference matrix — pi0.5", "",
             f"- inference delay `d` = {delay} env steps",
             f"- execution horizons `H` = {horizons}",
             f"- {data['args']['episodes']} episodes per cell, identical seeds across cells",
             "- runs: " + ", ".join(f"`{d.relative_to(REPO_ROOT)}`" for d in run_dirs), ""]

    tasks = []
    for (task, _), _ in cells.items():
        if task not in tasks:
            tasks.append(task)

    lines += ["## Success rates", "",
              "| task | H | sync | async (d>0) | async + RTC |", "|---|---|---|---|---|"]
    for task in tasks:
        for h in horizons:
            row = [rate(cells.get((task, f"{k}_H{h}"), {}).get("summary")) for k in ("sync", "async", "rtc")]
            lines.append(f"| {task} | {h} | {row[0]} | {row[1]} | {row[2]} |")

    lines += ["", "## Paired contrasts (McNemar on discordant seeds)", "",
              "`async - sync` isolates the cost of acting on a stale plan; "
              "`RTC - async` isolates what the guidance buys back.", "",
              "| task | H | async better | sync better | p | RTC better | async better | p |",
              "|---|---|---|---|---|---|---|---|"]

    pooled = defaultdict(lambda: [0, 0, 0, 0])
    for task in tasks:
        for h in horizons:
            sync = cells.get((task, f"sync_H{h}"), {}).get("outcomes", {})
            asyn = cells.get((task, f"async_H{h}"), {}).get("outcomes", {})
            rtc = cells.get((task, f"rtc_H{h}"), {}).get("outcomes", {})
            if not (sync and asyn and rtc):
                continue
            a1, b1, p1 = mcnemar(asyn, sync)
            a2, b2, p2 = mcnemar(rtc, asyn)
            pooled[h][0] += a1; pooled[h][1] += b1
            pooled[h][2] += a2; pooled[h][3] += b2
            lines.append(f"| {task} | {h} | {a1} | {b1} | {p1:.3f} | {a2} | {b2} | {p2:.3f} |")

    lines += ["", "### Pooled over tasks", "",
              "| H | async better | sync better | p | RTC better | async better | p |",
              "|---|---|---|---|---|---|---|"]
    for h in horizons:
        a1, b1, a2, b2 = pooled[h]
        p1 = binomtest(a1, a1 + b1, 0.5).pvalue if a1 + b1 else 1.0
        p2 = binomtest(a2, a2 + b2, 0.5).pvalue if a2 + b2 else 1.0
        lines.append(f"| {h} | {a1} | {b1} | {p1:.3f} | {a2} | {b2} | {p2:.3f} |")

    lines += ["", "## Inference latency", "", "| task | H | cell | median infer ms |", "|---|---|---|---|"]
    for task in tasks:
        for h in horizons:
            for k in ("sync", "async", "rtc"):
                s = cells.get((task, f"{k}_H{h}"), {}).get("summary")
                if s and s.get("average_inference_time_seconds"):
                    lines.append(f"| {task} | {h} | {k} | {s['average_inference_time_seconds']*1000:.0f} |")

    missing = [f"{t}/{c}" for (t, c), v in cells.items() if not v["summary"]]
    if missing:
        lines += ["", f"## Incomplete cells ({len(missing)})", "", ", ".join(sorted(missing))]

    text = "\n".join(lines) + "\n"
    (run_dir / args.out).write_text(text)
    print(text)
    print(f"wrote {run_dir / args.out}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
