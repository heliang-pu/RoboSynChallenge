#!/usr/bin/env python3
"""Run the RTC / async-inference evaluation matrix for pi0.5 and tabulate it.

Each cell is one `eval.sh` invocation differing only in how chunks are executed,
so every configuration sees the same episode seeds and the comparison is paired.

    python policy/pi05/run_rtc_matrix.py --tasks all --episodes 20 --delay 3
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_RE = re.compile(r"Metrics file:\s*(\S+)")

# Checkpoint per task, matching the ones report/official_random100.csv scored,
# ordered so the tasks that can actually show a difference run first: a policy
# that never succeeds cannot reveal whether RTC helps.
TASKS = [
    ("mixer_operating", 28000),
    ("water_pouring", 28000),
    ("items_handover", 28000),
    ("table_rearrangement", 30000),
    ("click_bell", 19999),
    ("manipulate_pipette", 28000),
    ("item_assembly", 28000),
    ("sample_loading", 28000),
    ("handle_basket", 28000),
    ("drawer_open_place", 28000),
]


def build_matrix(horizons, delay, extra_delay, probe_correction):
    """Cells: baseline (synchronous) vs async replanning vs async + RTC."""
    cells = []
    for h in horizons:
        cells.append({"name": f"sync_H{h}", "pi0_step": h, "async_mode": "off",
                      "inference_delay": 0, "rtc": False})
        cells.append({"name": f"async_H{h}", "pi0_step": h, "async_mode": "sim",
                      "inference_delay": delay, "rtc": False})
        cells.append({"name": f"rtc_H{h}", "pi0_step": h, "async_mode": "sim",
                      "inference_delay": delay, "rtc": True,
                      "rtc_correction": "identity"})

    if extra_delay:
        # Does RTC's benefit grow with latency?  Same horizon, longer delay.
        h = horizons[0]
        cells.append({"name": f"async_H{h}_d{extra_delay}", "pi0_step": h, "async_mode": "sim",
                      "inference_delay": extra_delay, "rtc": False})
        cells.append({"name": f"rtc_H{h}_d{extra_delay}", "pi0_step": h, "async_mode": "sim",
                      "inference_delay": extra_delay, "rtc": True,
                      "rtc_correction": "identity"})

    if probe_correction:
        h = horizons[0]
        cells.append({"name": f"rtc_H{h}_vjp", "pi0_step": h, "async_mode": "sim",
                      "inference_delay": delay, "rtc": True, "rtc_correction": "vjp"})
    return cells


def run_cell(cell, task, checkpoint_id, args, log_dir):
    overrides = [
        "--checkpoint_id", str(checkpoint_id),
        "--max_episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--headless", "True",
        "--eval_video_log", str(args.video),
        "--pi0_step", str(cell["pi0_step"]),
        "--async_mode", cell["async_mode"],
        "--inference_delay", str(cell["inference_delay"]),
        "--rtc", str(cell["rtc"]),
        "--rtc_correction", cell.get("rtc_correction", "identity"),
        "--prefix_attention_schedule", args.schedule,
    ]
    cmd = ["bash", str(REPO_ROOT / "policy" / "pi05" / "eval.sh"),
           task, args.setting, args.train_config, task, str(args.gpu), *overrides]

    # Each task's checkpoint ships norm stats under its own repo id.
    env = dict(os.environ, ROBOSYN_REPO_ID=f"RoboSynChallenge/cobotmagic_Sim_{task}")

    log_path = log_dir / f"{task}__{cell['name']}.log"
    started = time.time()
    print(f"\n=== {task} / {cell['name']} ===\n  -> {log_path}", flush=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started

    text = log_path.read_text(errors="replace")
    match = METRICS_RE.search(text)
    result = {"task": task, "checkpoint_id": checkpoint_id, "cell": cell,
              "returncode": proc.returncode, "wall_seconds": round(elapsed, 1),
              "log": str(log_path), "metrics_path": match.group(1) if match else None}
    if proc.returncode != 0:
        print(f"  FAILED (rc={proc.returncode}); see {log_path}", flush=True)
        result["tail"] = text[-2000:]
        return result

    metrics_file = REPO_ROOT / match.group(1) if match else None
    if metrics_file and metrics_file.exists():
        result["metrics"] = json.loads(metrics_file.read_text())
        summary = result["metrics"].get("summary", result["metrics"])
        print(f"  done in {elapsed/60:.1f} min: "
              f"success={summary.get('success_count')}/{summary.get('episode_count')}", flush=True)
    else:
        print(f"  done in {elapsed/60:.1f} min but no metrics file found", flush=True)
    return result


def tabulate(results):
    rows = ["| task | cell | H | mode | d | RTC | success | avg steps | infer ms | wall min |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["cell"]
        m = (r.get("metrics") or {}).get("summary", r.get("metrics") or {})
        sr = f"{m.get('success_count','?')}/{m.get('episode_count','?')}"
        steps = m.get("average_action_steps")
        infer = m.get("average_inference_time_seconds")
        rows.append(
            f"| {r['task']} | {c['name']} | {c['pi0_step']} | {c['async_mode']} | {c['inference_delay']} | "
            f"{'yes/' + c.get('rtc_correction','identity') if c['rtc'] else 'no'} | {sr} | "
            f"{steps if steps is None else round(steps,1)} | "
            f"{'' if infer is None else round(infer*1000)} | {r['wall_seconds']/60:.1f} |"
        )
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="task names, or 'all' for the built-in table")
    p.add_argument("--setting", default="random")
    p.add_argument("--train-config", default="pi05_base_robosynchallenge_full")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 50])
    p.add_argument("--delay", type=int, required=True,
                   help="inference delay in env steps; measure it, do not guess")
    p.add_argument("--extra-delay", type=int, default=0,
                   help="second, larger delay probed at the smallest horizon")
    p.add_argument("--probe-correction", action="store_true",
                   help="add one vjp-correction cell at the smallest horizon")
    p.add_argument("--schedule", default="exp")
    p.add_argument("--video", default="True")
    p.add_argument("--out", default="report/rtc_matrix")
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out / time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = TASKS if args.tasks == ["all"] else [(t, dict(TASKS)[t]) for t in args.tasks]
    cells = build_matrix(args.horizons, args.delay, args.extra_delay, args.probe_correction)
    total = len(tasks) * len(cells)
    print(f"{len(tasks)} tasks x {len(cells)} cells = {total} runs x {args.episodes} episodes "
          f"-> {out_dir}", flush=True)

    results = []
    for task, checkpoint_id in tasks:
        for cell in cells:
            results.append(run_cell(cell, task, checkpoint_id, args, log_dir))
            print(f"  [{len(results)}/{total}] complete", flush=True)
            (out_dir / "results.json").write_text(
                json.dumps({"args": vars(args), "results": results}, indent=2, ensure_ascii=False))
            (out_dir / "table.md").write_text(tabulate(results) + "\n")

    print("\n" + tabulate(results))
    print(f"\nwrote {out_dir}/results.json and table.md")
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
