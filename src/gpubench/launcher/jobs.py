"""Expand an experiment file (configs/exp/*.yaml) into an ordered list of shell jobs.

Job entry forms:
    - matrix: {algo: [...], task: [...], seed: [...]}    # cartesian product -> scripts/train.py runs
      overrides: {dotted.key: value}                     # optional, applied to every expanded run
      kind: train | profile                              # profile -> scripts/profile.sh wrapper
      allow_fail: bool
    - name: foo
      cmd: python scripts/whatever.py --flag             # arbitrary command, run from the repo root

Every train/profile job gets run.exp=<exp name> unless overridden, so results land in
results/<exp>/<run_id>/ on the instance and results/<gpu>/<exp>/<run_id>/ locally.
"""

from __future__ import annotations

import itertools
import shlex
from pathlib import Path

import yaml


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def expand(exp_path: str | Path) -> dict:
    exp_path = Path(exp_path)
    spec = yaml.safe_load(exp_path.read_text())
    name = spec.get("name", exp_path.stem)
    default_timeout = int(spec.get("timeout_min", 120)) * 60
    jobs = []
    for entry in spec["jobs"]:
        timeout = int(entry.get("timeout_min", default_timeout // 60)) * 60
        allow_fail = bool(entry.get("allow_fail", False))
        if "cmd" in entry:
            jobs.append({"name": entry["name"], "cmd": entry["cmd"], "timeout_s": timeout, "allow_fail": allow_fail})
            continue
        matrix = entry["matrix"]
        keys = list(matrix)
        kind = entry.get("kind", "train")
        combos = list(itertools.product(*(matrix[k] if isinstance(matrix[k], list) else [matrix[k]] for k in keys)))
        for combo in combos:
            params = dict(zip(keys, combo))
            ov = {"run.exp": name, **(entry.get("overrides") or {})}
            auto_id = "_".join(f"{params[k]}" if k != "seed" else f"s{params[k]}" for k in keys)
            if entry.get("name"):  # a named entry with several combos still needs unique run ids
                run_id = entry["name"] if len(combos) == 1 else f"{entry['name']}_s{params.get('seed', len(jobs))}"
            else:
                run_id = auto_id
            if kind == "profile":
                run_id = f"prof_{run_id}"
            args = [f"{k}={_fmt(v)}" for k, v in params.items()] + [f"{k}={_fmt(v)}" for k, v in ov.items()]
            if kind == "profile":
                cmd = "bash scripts/profile.sh " + shlex.quote(run_id) + " " + " ".join(shlex.quote(a) for a in args)
            else:
                args.append(f"run.run_id={run_id}")
                cmd = "python scripts/train.py " + " ".join(shlex.quote(a) for a in args)
            jobs.append({"name": run_id, "cmd": cmd, "timeout_s": timeout, "allow_fail": allow_fail})
    names = [j["name"] for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate job names in {exp_path}: {sorted(dup)}")
    return {"exp": name, "jobs": jobs}
