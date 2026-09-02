"""Organizer leaderboard score from consolidated evaluation_metrics.json files.

Overall = 75% x SuccessRate + 20% x ActionEfficiency + 5% x InferenceEfficiency
  ActionEfficiency    = mean over episodes of (1 - used_steps/H) x 100  (= (1 - average_action_steps_ratio) x 100)
  InferenceEfficiency = mean over episodes of max(0, 1 - inference_time/T) x 100
T = organizer ACT baseline total inference time per episode on RTX 5090 (released_checkpoint_results.json);
the leaderboard code (pages-app.js: actInferenceBaselineSeconds -> inferenceEfficiencyScore) has T only for
5 tasks; for the other 5 it returns 0 inference efficiency for everyone, so we do the same.
usage: score_official.py <run10_root> [--json out.json]
"""
import json, sys, glob, math, re
from pathlib import Path
root = Path(sys.argv[1]); out = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else None
H = {"click_bell":361,"handle_basket":500,"water_pouring":500,"table_rearrangement":361,"items_handover":350,"drawer_open_place":900,"mixer_operating":500,"item_assembly":361,"manipulate_pipette":1000,"sample_loading":500}
T_PUB = {"click_bell":0.0669,"drawer_open_place":0.1737,"mixer_operating":0.0746,"table_rearrangement":0.0576,"water_pouring":0.0659}
rows = []
for t in H:
    ms = sorted(glob.glob(str(root / t / "**" / "evaluation_metrics.json"), recursive=True))
    if not ms: rows.append({"task": t, "status": "missing"}); continue
    n = succ = calls = 0; steps = infer = 0.0
    for m in ms:
        s = json.load(open(m))["summary"]; n += s["episode_count"]; succ += s["success_count"]
        steps += s["average_action_steps"] * s["episode_count"]; c = s.get("inference_call_count") or 0; calls += c
        infer += (s.get("average_inference_time_seconds") or 0) * c
    sr = 100 * succ / n; ratio = steps / n / H[t]; ae = (1 - ratio) * 100
    T = T_PUB.get(t); est = T is None
    inf_ep = infer / n if n else 0.0
    ie = max(0.0, min(100.0, (1 - inf_ep / T) * 100)) if T else 0.0   # site: no baseline -> 0
    score = 0.75 * sr + 0.20 * ae + 0.05 * ie
    rows.append({"task": t, "status": "ok", "episodes": n, "success": succ, "success_rate": sr, "avg_steps": steps / n, "H": H[t],
                 "action_efficiency": ae, "inference_per_episode_s": inf_ep, "inference_per_call_ms": 1000 * infer / calls if calls else None,
                 "T_s": T, "T_missing_on_leaderboard": est, "inference_efficiency": ie, "overall": score, "shards": len(ms)})
ok = [r for r in rows if r["status"] == "ok"]
macro = {k: sum(r[k] for r in ok) / len(ok) for k in ("success_rate","action_efficiency","inference_efficiency","overall")} if ok else {}
print(f"{'task':22s} {'eps':>4s} {'SR%':>6s} {'steps':>7s} {'AE':>6s} {'inf/ep s':>9s} {'IE':>5s} {'score':>6s}")
for r in rows:
    if r["status"] != "ok": print(f"{r['task']:22s} (missing)"); continue
    print(f"{r['task']:22s} {r['episodes']:4d} {r['success_rate']:6.1f} {r['avg_steps']:7.1f} {r['action_efficiency']:6.1f} {r['inference_per_episode_s']:9.2f} {r['inference_efficiency']:5.1f} {r['overall']:6.1f}{' (no ACT T on site -> IE=0)' if r['T_missing_on_leaderboard'] else ''}")
if ok: print(f"{'macro avg ('+str(len(ok))+' tasks)':22s} {'':4s} {macro['success_rate']:6.1f} {'':7s} {macro['action_efficiency']:6.1f} {'':9s} {macro['inference_efficiency']:5.1f} {macro['overall']:6.1f}")
if out: out.write_text(json.dumps({"rows": rows, "macro": macro}, indent=1, ensure_ascii=False))
