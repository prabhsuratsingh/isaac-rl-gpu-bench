"""Run directory + JSONL metrics writer. Schema is documented in docs/protocol.md (section 'Logging format')."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from . import sysinfo

SCHEMA_VERSION = 1


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item"):  # 0-d tensor / numpy scalar
        return _clean(v.item())
    return v


class RunLogger:
    """Owns results/<exp>/<run_id>/ : run_meta.json, config.json, metrics.jsonl, summary.json."""

    def __init__(self, cfg: dict, results_root: str | os.PathLike = "results"):
        self.cfg = cfg
        run = cfg["run"]
        self.run_id = run.get("run_id") or f"{cfg['algo']['name']}_{cfg['task']['name']}_s{cfg['seed']}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.dir = Path(results_root) / run.get("exp", "adhoc") / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        self._f = open(self.dir / "metrics.jsonl", "a", buffering=1)
        self.common = {
            "run_id": self.run_id,
            "algo": cfg["algo"]["name"],
            "impl": cfg["algo"].get("impl", "reference"),
            "task": cfg["task"]["name"],
            "seed": cfg["seed"],
        }
        self.meta = {
            "schema_version": SCHEMA_VERSION,
            **self.common,
            "exp": run.get("exp", "adhoc"),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sys": sysinfo.collect(),
        }
        self._write_json("run_meta.json", self.meta)
        self._write_json("config.json", cfg)

    def _write_json(self, name: str, obj):
        with open(self.dir / name, "w") as f:
            json.dump(_clean(obj), f, indent=2, default=str)

    @property
    def wall_s(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, row: dict):
        rec = {**self.common, "wall_s": round(self.wall_s, 3), **row}
        self._f.write(json.dumps(_clean(rec), default=str) + "\n")

    def close(self, summary: dict | None = None):
        s = {**self.common, "wall_s": round(self.wall_s, 3), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        dph = self.meta["sys"]["vast"].get("dph")
        s["gpu_hours"] = round(self.wall_s / 3600, 5)
        s["cost_usd"] = round(s["gpu_hours"] * dph, 4) if dph else None
        s.update(summary or {})
        self._write_json("summary.json", s)
        self._f.close()
