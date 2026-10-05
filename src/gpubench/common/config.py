"""Tiny YAML config system: configs/algo/<algo>.yaml + configs/task/<task>.yaml + key=value overrides.

Resolved config layout:
    seed: int
    device: str
    algo: {name, impl, ...}          # from configs/algo/<algo>.yaml
    task: {name, gym_id, ...}        # from configs/task/<task>.yaml
    run:  {exp, run_id, results_root}
    profile: {start, length}         # Nsight capture window (iterations / env steps)

A task file may carry `algo_overrides: {<algo file name>: {...}}`, merged into `algo`, so per-task
settings live next to the task. CLI overrides are applied last, e.g.
    algo=pets_naive task=cartpole seed=3 algo.mppi.horizon=20 task.num_envs=64 run.exp=step1_pets
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

DEFAULTS = {
    "seed": 0,
    "device": "cuda:0",
    "run": {"exp": "adhoc", "run_id": None, "results_root": "results"},
    "profile": {"start": None, "length": None},
}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        choices = sorted(p.stem for p in path.parent.glob("*.yaml"))
        raise FileNotFoundError(f"{path} not found. Available: {choices}")
    return yaml.safe_load(path.read_text()) or {}


def _set_dotted(d: dict, key: str, val):
    parts = key.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {key}: {p} is not a mapping")
    node[parts[-1]] = val


def load_config(args: list[str] | dict, config_dir: Path = CONFIG_DIR) -> dict:
    """`args` is a list of key=value strings (CLI) or a flat {dotted_key: value} dict (job files)."""
    if isinstance(args, list):
        pairs = []
        for item in args:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got {item!r}")
            k, raw = item.split("=", 1)
            pairs.append((k, yaml.safe_load(raw) if raw != "" else None))
    else:
        pairs = list(args.items())

    names = {k: v for k, v in pairs if k in ("algo", "task")}
    if "algo" not in names or "task" not in names:
        raise ValueError("algo=<name> and task=<name> are required")

    algo = _load_yaml(config_dir / "algo" / f"{names['algo']}.yaml")
    task = _load_yaml(config_dir / "task" / f"{names['task']}.yaml")
    algo.setdefault("name", names["algo"])
    task.setdefault("name", names["task"])
    overrides_by_algo = task.pop("algo_overrides", None) or {}
    # keyed by algo family (e.g. "pets", shared by pets_naive/pets_fused) then by exact file name
    per_task = deep_merge(overrides_by_algo.get(algo["name"], {}), overrides_by_algo.get(names["algo"], {}) if names["algo"] != algo["name"] else {})

    cfg = copy.deepcopy(DEFAULTS)
    cfg["algo"] = deep_merge(algo, per_task)
    cfg["algo"]["config_file"] = names["algo"]
    cfg["task"] = task
    for k, v in pairs:
        if k not in ("algo", "task"):
            _set_dotted(cfg, k, v)
    return cfg


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False)
