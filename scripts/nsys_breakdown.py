"""Turn an Nsight Systems capture into the per-phase time breakdown used by the stacked-bar figure.

    python scripts/nsys_breakdown.py results/profiles/<name>      # reads <name>.sqlite and <name>_*.csv

Per phase (our NVTX ranges: sim, plan, model_train, policy_act, policy_update, ...):
    host_ms      time the range was open on the CPU (wall time attributed to the phase)
    gpu_busy_ms  sum of GPU activity (kernels + memcpy + memset) *launched from inside* the range,
                 joined via CUPTI correlation ids. This is real device busy time, unlike nsys'
                 "projected" GPU time, which spans first-to-last GPU op and counts the idle gaps.
    gpu_proj_ms  nsys nvtx_gpu_proj_sum (kept for reference)
    calls
_total: window_ms (capture span), host_ms, gpu_busy_ms (union of all device activity), gpu_util.
top_kernels: from cuda_gpu_kern_sum.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

PHASES = {"sim", "plan", "model_train", "policy_act", "policy_update", "storage", "buffer", "random_act"}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        lines = [l for l in f if l.strip() and not l.startswith(("Processing", "Generating", "WARNING", "SKIPPED"))]
    return list(csv.DictReader(lines))


def _col(row: dict, *names):
    for n in names:
        for k, v in row.items():
            if k and k.strip().lower() == n.lower():
                return v
    return None


def _name(row: dict) -> str:
    return (_col(row, "Range", "Name") or "?").lstrip(":").strip()


def _union_ms(intervals) -> float:
    total, cur_s, cur_e = 0, None, None
    for s, e in intervals:  # sorted by start
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1e6


def from_sqlite(db: Path) -> dict:
    import pandas as pd

    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    ranges = pd.read_sql(
        "select e.start, e.end, e.globalTid as tid, coalesce(e.text, s.value) as phase from NVTX_EVENTS e "
        "left join StringIds s on e.textId = s.id where e.end is not null", con)
    ranges = ranges[ranges.phase.isin(PHASES)].sort_values("start")
    rt = pd.read_sql("select start, globalTid as tid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME", con).sort_values("start")
    acts = [pd.read_sql(f"select start, end, correlationId from {t}", con)
            for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET") if t in tables]
    gpu = pd.concat(acts, ignore_index=True)
    con.close()

    # Attribute each runtime API call (from ANY thread) to the phase open at that moment. Our phases are
    # strictly sequential on the training thread, and PhysX launches its kernels from its own worker
    # threads during env.step(), so per-thread attribution would leave the simulator's GPU work orphaned.
    m = pd.merge_asof(rt.drop(columns="tid"), ranges.drop(columns="tid").rename(columns={"start": "r_start", "end": "r_end"}),
                      left_on="start", right_on="r_start", direction="backward")
    m = m[(m.phase.notna()) & (m.start <= m.r_end)]
    g = gpu.merge(m[["correlationId", "phase"]], on="correlationId", how="inner")
    busy = ((g.end - g.start).groupby(g.phase).sum() / 1e6).to_dict()

    host = ((ranges.end - ranges.start).groupby(ranges.phase).sum() / 1e6).to_dict()
    calls = ranges.groupby("phase").size().to_dict()
    window_start = min(gpu.start.min(), ranges.start.min())
    window_end = max(gpu.end.max(), ranges.end.max())
    gs = gpu.sort_values("start")
    total_busy = _union_ms(zip(gs.start.to_numpy(), gs.end.to_numpy()))
    out = {p: {"host_ms": host.get(p, 0.0), "gpu_busy_ms": busy.get(p, 0.0), "calls": int(calls.get(p, 0))} for p in host}
    window_ms = (window_end - window_start) / 1e6
    out["_total"] = {"window_ms": window_ms, "host_ms": sum(host.values()), "gpu_busy_ms": total_busy,
                     "gpu_util": total_busy / window_ms if window_ms else None,
                     "gpu_busy_unattributed_ms": max(0.0, total_busy - sum(busy.values()))}
    return out


def breakdown(base: str) -> dict:
    base = Path(base)
    db = base.with_suffix(".sqlite") if base.suffix != ".sqlite" else base
    out: dict = from_sqlite(db) if db.exists() else {}
    if not out:  # csv-only fallback
        for r in _read_csv(Path(f"{base}_nvtx_sum.csv")):
            n = _name(r)
            if n in PHASES:
                out[n] = {"host_ms": float(_col(r, "Total Time (ns)") or 0) / 1e6, "calls": int(float(_col(r, "Instances") or 0))}
        out["_total"] = {"host_ms": sum(v["host_ms"] for v in out.values())}
    for r in _read_csv(Path(f"{base}_nvtx_gpu_proj_sum.csv")):
        n = _name(r)
        if n in out:
            out[n]["gpu_proj_ms"] = float(_col(r, "Total Proj Time (ns)", "Projected Duration (ns)", "Total Time (ns)") or 0) / 1e6
    th = out["_total"].get("host_ms") or 1.0
    tb = sum(v.get("gpu_busy_ms", 0) for k, v in out.items() if k != "_total") or 1.0
    for k, v in out.items():
        if k != "_total":
            v["host_frac"] = round(v["host_ms"] / th, 4)
            if "gpu_busy_ms" in v:
                v["gpu_busy_frac"] = round(v["gpu_busy_ms"] / tb, 4)
                v["gpu_util_in_phase"] = round(v["gpu_busy_ms"] / v["host_ms"], 4) if v["host_ms"] else None
    out["top_kernels"] = [
        {"name": (_col(r, "Name") or "")[:120], "ms": float(_col(r, "Total Time (ns)") or 0) / 1e6, "instances": _col(r, "Instances")}
        for r in _read_csv(Path(f"{base}_cuda_gpu_kern_sum.csv"))[:15]
    ]
    return out


if __name__ == "__main__":
    print(json.dumps(breakdown(sys.argv[1]), indent=2, default=float))
