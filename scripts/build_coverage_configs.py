#!/usr/bin/env python3
"""Build collection-ready coverage gym_configs from a coverage summary.

Reads ``report/coverage/<task>/coverage_summary.json`` (produced by
``scripts/analyze_random_coverage.py``) plus the official
``configs/<task>/random/gym_config.json`` and writes, for every recommendation,
``configs/<task>/coverage_<name>/gym_config.json`` (+ README.md):

- pair mode (two rigid objects placed by ``randomize_rigid_object_pose``):
  the two pose events are replaced by a single
  ``randomize_rigid_object_pair_pose_constrained`` event with the narrowed
  ranges, mesh-derived half extents and a conservative ``min_xy_clearance``
  (default 0.03, relaxed only when the audited expert data shows successful
  layouts closer than that).
- keep mode (articulations / group events / clearance-incompatible layouts):
  the original event functions are kept and only their
  ``position_range`` / ``rotation_range`` are narrowed.

Every generated config is JSON-parsed and deep-diffed against the official
random config; only the whitelisted paths may differ. Results are merged into
``report/coverage/PLAN.json``.

Example:
    python scripts/build_coverage_configs.py --task drawer_open_place
    python scripts/build_coverage_configs.py --task click_bell --dry-run
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _coverage_common import (  # noqa: E402
    DEFAULT_REPORT_ROOT,
    DEFAULT_SAVE_ROOT,
    PAIR_EVENT_FUNC,
    REPO_ROOT,
    entity_init,
    find_pose_event,
    load_json,
    mc_pair_acceptance,
)

MC_RELAX_THRESHOLD = 0.02  # relax clearance below this acceptance
MC_SKIP_THRESHOLD = 0.005  # skip the config below this acceptance
RELAXED_CLEARANCE = 0.01


# --------------------------------------------------------------------------
# Config assembly
# --------------------------------------------------------------------------
def _event_relative_flags(event: dict) -> tuple[bool, bool]:
    params = event.get("params", {})
    return (
        bool(params.get("relative_position", True)),
        bool(params.get("relative_rotation", False)),
    )


def _final_ranges(event: dict, override: dict | None) -> tuple[list, list | None, bool]:
    """(position_range, rotation_range, rotation_is_placeholder)."""
    params = event.get("params", {})
    override = override or {}
    position_range = override.get("position_range") or copy.deepcopy(
        params.get("position_range")
    )
    if position_range is None:
        raise ValueError("target event has no position_range")
    rotation_range = override.get("rotation_range") or copy.deepcopy(
        params.get("rotation_range")
    )
    placeholder = rotation_range is None
    if placeholder:
        rotation_range = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return position_range, rotation_range, placeholder


def _absolute_xy_box(position_range: list, relative: bool, init_pos) -> list:
    low = [float(v) for v in position_range[0][:2]]
    high = [float(v) for v in position_range[1][:2]]
    if relative:
        low = [low[i] + float(init_pos[i]) for i in range(2)]
        high = [high[i] + float(init_pos[i]) for i in range(2)]
    return [low, high]


def _max_corner_distance(box_a: list, box_b: list) -> float:
    best = 0.0
    for xa in (box_a[0][0], box_a[1][0]):
        for ya in (box_a[0][1], box_a[1][1]):
            for xb in (box_b[0][0], box_b[1][0]):
                for yb in (box_b[0][1], box_b[1][1]):
                    best = max(best, math.hypot(xa - xb, ya - yb))
    return best


def build_pair_event(
    base: dict,
    hints: dict,
    rec: dict,
    min_clearance: float,
) -> tuple[str, dict, dict]:
    """Return (event_name, event_dict, mc_specs) for the constrained pair."""
    pair = hints["pair"]
    first_uid, second_uid = pair["first_uid"], pair["second_uid"]
    specs = {}
    ranges = {}
    for role, uid in (("first", first_uid), ("second", second_uid)):
        _, event = find_pose_event(base, uid)
        rel_pos, rel_rot = _event_relative_flags(event)
        override = rec.get("overrides", {}).get(uid)
        position_range, rotation_range, placeholder = _final_ranges(event, override)
        if placeholder:
            rel_rot = True  # zero-width relative rotation keeps the init rotation
        init_pos, init_rot, _ = entity_init(base, uid)
        ranges[role] = {
            "uid": uid,
            "position_range": position_range,
            "rotation_range": rotation_range,
            "relative_position": rel_pos,
            "relative_rotation": rel_rot,
        }
        specs[role] = {
            "position_range": position_range,
            "rotation_range": rotation_range,
            "relative_position": rel_pos,
            "relative_rotation": rel_rot,
            "init_pos": init_pos,
            "init_rot": init_rot,
        }
    box_a = _absolute_xy_box(
        ranges["first"]["position_range"],
        ranges["first"]["relative_position"],
        specs["first"]["init_pos"],
    )
    box_b = _absolute_xy_box(
        ranges["second"]["position_range"],
        ranges["second"]["relative_position"],
        specs["second"]["init_pos"],
    )
    max_distance = math.ceil((_max_corner_distance(box_a, box_b) + 0.02) * 100) / 100
    event_name = f"randomize_{first_uid}_{second_uid}_pose_constrained"
    event = {
        "func": PAIR_EVENT_FUNC,
        "mode": "reset",
        "params": {
            "first_entity_cfg": {"uid": first_uid},
            "second_entity_cfg": {"uid": second_uid},
            "first_position_range": ranges["first"]["position_range"],
            "second_position_range": ranges["second"]["position_range"],
            "first_rotation_range": ranges["first"]["rotation_range"],
            "second_rotation_range": ranges["second"]["rotation_range"],
            "first_half_extents": pair["first_half_extents"],
            "second_half_extents": pair["second_half_extents"],
            "first_relative_position": ranges["first"]["relative_position"],
            "second_relative_position": ranges["second"]["relative_position"],
            "first_relative_rotation": ranges["first"]["relative_rotation"],
            "second_relative_rotation": ranges["second"]["relative_rotation"],
            "min_xy_clearance": min_clearance,
            "max_xy_center_distance": max_distance,
            "max_resample_attempts": 256,
            "physics_update_step": 1,
        },
    }
    return event_name, event, specs


def apply_pair_mode(config: dict, hints: dict, event_name: str, event: dict) -> None:
    pair = hints["pair"]
    targets = {pair["first_event"], pair["second_event"]}
    events = config["env"]["events"]
    new_events = {}
    inserted = False
    for name, value in events.items():
        if name in targets:
            if not inserted:
                new_events[event_name] = event
                inserted = True
            continue
        new_events[name] = value
    if not inserted:
        raise KeyError(f"target events {targets} not found in base config")
    config["env"]["events"] = new_events


def apply_keep_mode(config: dict, rec: dict) -> list[str]:
    touched = []
    for uid, override in rec.get("overrides", {}).items():
        name, event = find_pose_event(config, uid)
        params = event["params"]
        if "position_range" in override and "position_range" in params:
            params["position_range"] = override["position_range"]
        if (
            override.get("rotation_range") is not None
            and "rotation_range" in params
        ):
            params["rotation_range"] = override["rotation_range"]
        touched.append(name)
    return touched


def apply_recorder(config: dict, task: str, rec: dict, save_root: Path) -> None:
    recorder = config["env"]["dataset"]["lerobot"]["params"]
    recorder["save_path"] = str(save_root / f"{task}_coverage" / f"coverage_{rec['name']}")
    extra = recorder.setdefault("extra", {})
    extra["coverage_group"] = f"coverage_{rec['name']}"
    extra["coverage_reason"] = rec["reason"]
    extra["coverage_kind"] = rec["kind"]


# --------------------------------------------------------------------------
# Verification (deep diff against the official random config)
# --------------------------------------------------------------------------
def deep_diff(a, b, path=()):  # noqa: ANN001
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in b:
                yield (*path, str(key)), "removed"
            elif key not in a:
                yield (*path, str(key)), "added"
            else:
                yield from deep_diff(a[key], b[key], (*path, str(key)))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from deep_diff(x, y, (*path, str(i)))
    else:
        if a != b:
            yield path, "changed"


def verify_config(
    base: dict,
    generated: dict,
    mode: str,
    replaced_events: set[str],
    added_events: set[str],
) -> list[str]:
    violations = []
    for path, kind in deep_diff(base, generated):
        if path == ("max_episodes",):
            continue
        if len(path) >= 3 and path[:2] == ("env", "events"):
            name = path[2]
            if mode == "pair":
                if name in replaced_events and kind == "removed" and len(path) == 3:
                    continue
                if name in added_events and kind == "added" and len(path) == 3:
                    continue
            else:
                if name in replaced_events and len(path) >= 5 and path[3] == "params" and path[4] in (
                    "position_range",
                    "rotation_range",
                ):
                    continue
            violations.append(f"{kind}: {'/'.join(path)}")
            continue
        if path[:5] == ("env", "dataset", "lerobot", "params", "save_path"):
            continue
        if len(path) >= 5 and path[:5] == ("env", "dataset", "lerobot", "params", "extra"):
            continue
        violations.append(f"{kind}: {'/'.join(path)}")
    # event ordering must be preserved
    if mode == "pair":
        base_order = [n for n in base["env"]["events"] if n not in replaced_events]
        gen_order = [n for n in generated["env"]["events"] if n not in added_events]
    else:
        base_order = list(base["env"]["events"])
        gen_order = list(generated["env"]["events"])
    if base_order != gen_order:
        violations.append("event order changed for untouched events")
    return violations


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------
def _fmt_range(value) -> str:
    if value is None:
        return "-"
    return json.dumps(value, ensure_ascii=False)


def write_readme(
    path: Path,
    task: str,
    rec: dict,
    mode: str,
    range_rows: list[tuple[str, str, str, str]],
    geometry: dict | None,
    acceptance: float | None,
    save_path: str,
) -> None:
    name = f"coverage_{rec['name']}"
    lines = [
        f"# {task} / {name}",
        "",
        f"- 用途: {'密度缺口补采' if rec['kind'] != 'strat' else '全范围分层补匀'}"
        f"(kind={rec['kind']})",
        f"- 建议集数: {rec['episodes']}",
        f"- 事件模式: {'pair-constrained(约束对采样)' if mode == 'pair' else 'keep-original(仅收窄官方事件范围)'}",
        f"- 依据: {rec['reason']}",
    ]
    if geometry:
        lines += [
            "- 几何依据: "
            f"half_extents {geometry['first_uid']}={geometry['first_half_extents']}, "
            f"{geometry['second_uid']}={geometry['second_half_extents']}"
            "(mesh AABB x body_scale x 1.05), "
            f"min_xy_clearance={geometry['min_xy_clearance']}, "
            f"max_xy_center_distance={geometry['max_xy_center_distance']}(非约束上界)",
        ]
    if acceptance is not None:
        lines.append(f"- 离线 MC 可行率(2 万采样): {acceptance:.1%}")
    lines += [
        "",
        "## 区间(config 单位,与官方 random 对比)",
        "",
        "| 对象 | 参数 | 官方 random | 本配置 |",
        "|---|---|---|---|",
    ]
    for row in range_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## 采集",
        "",
        "```bash",
        "python -m scripts.run_env \\",
        f"    --gym_config configs/{task}/{name}/gym_config.json \\",
        f"    --action_config configs/{task}/action_config.json \\",
        f"    --num_envs 1 --max_episodes {rec['episodes']} --headless --report_task_success",
        "```",
        "",
        f"输出数据集: `{save_path}`",
        "",
        f"由 scripts/build_coverage_configs.py 生成于 {_dt.date.today().isoformat()},"
        "分析来源 report/coverage/" + task + "/coverage_summary.json。",
        "",
    ]
    path.write_text("\n".join(lines))


def collect_range_rows(base: dict, generated: dict, rec: dict, mode: str, hints: dict):
    rows = []
    uids: list[str] = []
    if mode == "pair":
        uids = [hints["pair"]["first_uid"], hints["pair"]["second_uid"]]
    else:
        uids = list(rec.get("overrides", {}).keys())
    for uid in uids:
        _, base_event = find_pose_event(base, uid)
        base_params = base_event.get("params", {})
        if mode == "pair":
            params = None
            for event in generated["env"]["events"].values():
                if event.get("func") == PAIR_EVENT_FUNC:
                    params = event["params"]
            role = "first" if uid == hints["pair"]["first_uid"] else "second"
            new_pos = params[f"{role}_position_range"]
            new_rot = params[f"{role}_rotation_range"]
        else:
            _, gen_event = find_pose_event(generated, uid)
            new_pos = gen_event["params"].get("position_range")
            new_rot = gen_event["params"].get("rotation_range")
        rows.append(
            (uid, "position_range", _fmt_range(base_params.get("position_range")), _fmt_range(new_pos))
        )
        rows.append(
            (uid, "rotation_range", _fmt_range(base_params.get("rotation_range")), _fmt_range(new_rot))
        )
    return rows


# --------------------------------------------------------------------------
# Plan merge
# --------------------------------------------------------------------------
def merge_plan(plan_path: Path, task: str, entry: dict) -> None:
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = {"tasks": {}}
    plan["generated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    plan["tasks"][task] = entry
    totals = {"configs": 0, "episodes": 0, "tasks": len(plan["tasks"])}
    for item in plan["tasks"].values():
        totals["configs"] += len(item.get("configs", []))
        totals["episodes"] += item.get("total_episodes", 0)
    plan["totals"] = totals
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--config-root", type=Path, default=None)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_REPORT_ROOT / "PLAN.json")
    parser.add_argument(
        "--mode", choices=["auto", "pair", "keep"], default="auto",
        help="Override the analysis builder hint",
    )
    parser.add_argument("--min-clearance", type=float, default=None)
    parser.add_argument("--mc-samples", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing coverage dirs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = args.task
    summary_path = args.summary or (DEFAULT_REPORT_ROOT / task / "coverage_summary.json")
    base_path = args.base_config or (
        REPO_ROOT / "configs" / task / "random" / "gym_config.json"
    )
    config_root = args.config_root or (REPO_ROOT / "configs" / task)
    summary = load_json(summary_path)
    base = load_json(base_path)
    hints = summary.get("builder_hints", {"mode": "keep"})
    mode = hints.get("mode", "keep") if args.mode == "auto" else args.mode
    if mode == "pair" and not hints.get("pair"):
        raise SystemExit("--mode pair requested but the summary has no pair hints")

    recommendations = summary.get("recommendations", [])
    if not recommendations:
        raise SystemExit(f"No recommendations in {summary_path}")

    written, skipped = [], []
    for rec in recommendations:
        name = f"coverage_{rec['name']}"
        out_dir = config_root / name
        out_path = out_dir / "gym_config.json"
        if out_path.exists() and not args.force:
            skipped.append({"name": name, "reason": "already exists (no --force)"})
            print(f"  skip {name}: already exists")
            continue

        config = copy.deepcopy(base)
        config["max_episodes"] = int(rec["episodes"])
        acceptance = None
        geometry = None
        replaced: set[str] = set()
        added: set[str] = set()
        if mode == "pair":
            clearance = (
                args.min_clearance
                if args.min_clearance is not None
                else float(hints["pair"]["min_xy_clearance"])
            )
            event_name, event, specs = build_pair_event(base, hints, rec, clearance)
            acceptance = mc_pair_acceptance(
                specs["first"],
                specs["second"],
                np.asarray(hints["pair"]["first_half_extents"]),
                np.asarray(hints["pair"]["second_half_extents"]),
                clearance,
                event["params"]["max_xy_center_distance"],
                samples=args.mc_samples,
            )
            if acceptance < MC_RELAX_THRESHOLD:
                clearance = RELAXED_CLEARANCE
                event["params"]["min_xy_clearance"] = clearance
                acceptance = mc_pair_acceptance(
                    specs["first"],
                    specs["second"],
                    np.asarray(hints["pair"]["first_half_extents"]),
                    np.asarray(hints["pair"]["second_half_extents"]),
                    clearance,
                    event["params"]["max_xy_center_distance"],
                    samples=args.mc_samples,
                )
            if acceptance < MC_SKIP_THRESHOLD:
                skipped.append(
                    {
                        "name": name,
                        "reason": (
                            f"pair constraint infeasible (MC acceptance "
                            f"{acceptance:.2%} even at clearance {clearance})"
                        ),
                    }
                )
                print(f"  skip {name}: MC acceptance {acceptance:.2%}")
                continue
            apply_pair_mode(config, hints, event_name, event)
            replaced = {hints["pair"]["first_event"], hints["pair"]["second_event"]}
            added = {event_name}
            geometry = {
                "first_uid": hints["pair"]["first_uid"],
                "second_uid": hints["pair"]["second_uid"],
                "first_half_extents": hints["pair"]["first_half_extents"],
                "second_half_extents": hints["pair"]["second_half_extents"],
                "min_xy_clearance": event["params"]["min_xy_clearance"],
                "max_xy_center_distance": event["params"]["max_xy_center_distance"],
            }
        else:
            replaced = set(apply_keep_mode(config, rec))
        apply_recorder(config, task, rec, args.save_root)

        violations = verify_config(base, config, mode, replaced, added)
        if violations:
            skipped.append({"name": name, "reason": f"verify failed: {violations}"})
            print(f"  skip {name}: verify failed {violations}")
            continue

        save_path = config["env"]["dataset"]["lerobot"]["params"]["save_path"]
        if args.dry_run:
            print(f"  [dry-run] would write {out_path} ({rec['episodes']} episodes)")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n")
            range_rows = collect_range_rows(base, config, rec, mode, hints)
            write_readme(
                out_dir / "README.md",
                task,
                rec,
                mode,
                range_rows,
                geometry,
                acceptance,
                save_path,
            )
            print(
                f"  wrote {out_path.relative_to(REPO_ROOT)} "
                f"({rec['episodes']} eps, mode={mode}"
                + (f", MC {acceptance:.1%}" if acceptance is not None else "")
                + ")"
            )
        written.append(
            {
                "name": name,
                "path": str(out_path.relative_to(REPO_ROOT)),
                "episodes": int(rec["episodes"]),
                "kind": rec["kind"],
                "mode": mode,
                "mc_acceptance": round(acceptance, 4) if acceptance is not None else None,
                "verify_ok": True,
                "save_path": save_path,
            }
        )

    entry = {
        "mode": mode,
        "summary": str(summary_path),
        "base_config": str(base_path),
        "analysis_notes": summary.get("notes", []),
        "builder_notes": hints.get("notes", []),
        "configs": written,
        "skipped": skipped,
        "total_episodes": int(sum(c["episodes"] for c in written)),
    }
    if not args.dry_run:
        merge_plan(args.plan, task, entry)
        print(f"  plan updated: {args.plan}")
    print(
        f"[{task}] mode={mode} configs={len(written)} "
        f"episodes={entry['total_episodes']} skipped={len(skipped)}"
    )


if __name__ == "__main__":
    main()
