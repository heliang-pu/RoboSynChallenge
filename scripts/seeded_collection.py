"""Deterministic per-episode seeding for offline data generation (opt-in).

``run_env.py --seed <master>`` wraps the environment in :class:`SeededCollection`.
Every scene-creating ``env.reset()`` then draws a per-episode seed from a
``numpy`` RNG derived from the master seed, seeds both numpy and torch (the
EmbodiChain randomization events and planners), and records which seed produced
which *saved* episode.  On close (or interpreter exit) the mapping is written to
``episode_success.json`` inside the recorder dataset directory, together with
the official task-success verdict of each saved episode, the config hash and
the git commit — enough to reproduce or extend the collection later without
ever replaying the same scene twice.

Without ``--seed`` nothing here is imported and the official behaviour is
byte-identical.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        commit = out.stdout.strip() or "unknown"
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "unknown"


class SeededCollection:
    """Thin env proxy: inject per-episode seeds into reset(), log seed→episode."""

    def __init__(self, env, master_seed: int, gym_config: dict, gym_config_path: str | None = None):
        self._env = env
        self._master_seed = int(master_seed)
        self._rng = np.random.RandomState(self._master_seed)
        self._records = []
        self._current_scene_seed = None
        self._last_saved_count = None
        self._sidecar_written = False
        self._repo_root = Path(__file__).resolve().parent.parent
        self._config_path = gym_config_path
        self._config_sha1 = None
        if gym_config_path and Path(gym_config_path).is_file():
            self._config_sha1 = hashlib.sha1(Path(gym_config_path).read_bytes()).hexdigest()[:16]
        self._task_description = (
            gym_config.get("env", {}).get("dataset", {}).get("lerobot", {})
            .get("params", {}).get("extra", {}).get("task_description")
        )
        atexit.register(self._write_sidecar)

    # -- env delegation ----------------------------------------------------
    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def close(self):
        self._write_sidecar()
        return self._env.close()

    # -- the interception --------------------------------------------------
    def reset(self, seed=None, options=None):
        prev_seed = self._current_scene_seed
        prev_steps = self._elapsed_steps_snapshot()
        save_decision = (
            bool(options["save_data"])
            if isinstance(options, dict) and "save_data" in options
            else None
        )
        ep_seed = int(seed) if seed is not None else int(self._rng.randint(0, 2**31 - 1))
        print(f"[seeded_collection] reset #{len(self._records)}+ scene_seed={ep_seed}", flush=True)
        np.random.seed(ep_seed)
        torch.manual_seed(ep_seed)  # base_env.reset does this too; keep both sources aligned
        result = self._env.reset(seed=ep_seed, options=options)

        saved = self._saved_episode_count()
        if self._last_saved_count is None:
            self._last_saved_count = saved
        elif saved is not None and saved > self._last_saved_count:
            # In save-only-success collection, run_env passes the official
            # verdict explicitly as options["save_data"] before reset.  Read
            # that decision captured above; _task_success is cleared by reset
            # on several task implementations and therefore falsely labels
            # every saved episode as failure when inspected afterward.
            success = (
                save_decision
                if save_decision is not None
                else self._official_success_verdict()
            )
            for idx in range(self._last_saved_count, saved):
                self._records.append(
                    {
                        "episode_index": idx,
                        "seed": prev_seed,
                        "success": success,
                        "env_steps": prev_steps,
                    }
                )
            self._last_saved_count = saved
        self._current_scene_seed = ep_seed
        return result

    # -- helpers -----------------------------------------------------------
    def _elapsed_steps_snapshot(self):
        try:
            steps = self._env.unwrapped._elapsed_steps
            return int(torch.as_tensor(steps).max().item())
        except Exception:
            return None

    def _official_success_verdict(self):
        """base_env.reset() evaluates is_task_success() on the closing scene
        before re-randomizing, so right after reset this holds the official
        verdict of the episode that was just saved."""
        try:
            verdict = self._env.unwrapped._task_success
            return bool(torch.as_tensor(verdict).reshape(-1)[0].item())
        except Exception:
            return None

    def _saved_episode_count(self):
        try:
            dataset_manager = self._env.get_wrapper_attr("dataset_manager")
        except AttributeError:
            dataset_manager = getattr(self._env.unwrapped, "dataset_manager", None)
        if dataset_manager is None:
            return None
        counts = []
        for mode_cfgs in getattr(dataset_manager, "_mode_functor_cfgs", {}).values():
            for functor_cfg in mode_cfgs:
                functor = getattr(functor_cfg, "func", None)
                if hasattr(functor, "curr_episode"):
                    counts.append(int(functor.curr_episode))
        return max(counts) if counts else None

    def _dataset_dir(self):
        try:
            dataset_manager = self._env.get_wrapper_attr("dataset_manager")
        except AttributeError:
            dataset_manager = getattr(self._env.unwrapped, "dataset_manager", None)
        for mode_cfgs in getattr(dataset_manager, "_mode_functor_cfgs", {}).values():
            for functor_cfg in mode_cfgs:
                functor = getattr(functor_cfg, "func", None)
                dataset = getattr(functor, "dataset", None)
                root = getattr(dataset, "root", None)
                if root:
                    return Path(root)
                root = getattr(functor, "save_path", None)
                if root:
                    return Path(root)
        return None

    def _write_sidecar(self):
        if self._sidecar_written or not self._records:
            return
        dataset_dir = self._dataset_dir()
        if dataset_dir is None or not dataset_dir.is_dir():
            print(f"[seeded_collection] 找不到录制目录,边车未写(记录 {len(self._records)} 条)")
            return
        payload = {
            "labels_field": "episode_success",
            "master_seed": self._master_seed,
            "gym_config": str(self._config_path) if self._config_path else None,
            "gym_config_sha1": self._config_sha1,
            "git_commit": _git_commit(self._repo_root),
            "task_description": self._task_description,
            "saved_episode_count": len(self._records),
            "episodes": [
                {**rec, "success": rec["success"]}
                for rec in self._records
            ],
        }
        out = dataset_dir / "episode_success.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        tmp.replace(out)
        self._sidecar_written = True
        n_succ = sum(1 for r in self._records if r["success"])
        print(
            f"[seeded_collection] 边车已写: {out} "
            f"({len(self._records)} 集, 官方判定成功 {n_succ}, master_seed={self._master_seed})"
        )
