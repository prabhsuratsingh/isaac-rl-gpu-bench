"""Turn `nsys stats` CSVs into the per-phase time breakdown used by the stacked-bar figure.

    python scripts/nsys_breakdown.py results/profiles/<name>      # reads <name>_nvtx_sum.csv etc.

Output JSON: {phase: {host_ms, gpu_ms, calls, host_frac, gpu_frac}, "_total": {...}, "top_kernels": [...]}.
host_ms = time the phase's NVTX range was open on the CPU; gpu_ms = GPU work projected onto the range.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _read(path: Path) -> list[dict]:
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


def breakdown(base: str) -> dict:
    base = Path(base)
    host = _read(Path(f"{base}_nvtx_sum.csv"))
    gpu = _read(Path(f"{base}_nvtx_gpu_proj_sum.csv"))
    kern = _read(Path(f"{base}_cuda_gpu_kern_sum.csv"))
    out: dict = {}
    for r in host:
        n = _name(r)
        out.setdefault(n, {})
        out[n]["host_ms"] = float(_col(r, "Total Time (ns)") or 0) / 1e6
        out[n]["calls"] = int(float(_col(r, "Instances") or 0))
    for r in gpu:
        n = _name(r)
        out.setdefault(n, {})
        out[n]["gpu_ms"] = float(_col(r, "Total Proj Time (ns)", "Projected Duration (ns)", "Total Time (ns)") or 0) / 1e6
    th = sum(v.get("host_ms", 0) for v in out.values()) or 1.0
    tg = sum(v.get("gpu_ms", 0) for v in out.values()) or 1.0
    for v in out.values():
        v["host_frac"] = round(v.get("host_ms", 0) / th, 4)
        v["gpu_frac"] = round(v.get("gpu_ms", 0) / tg, 4)
    out["_total"] = {"host_ms": th, "gpu_ms": tg}
    out["top_kernels"] = [
        {"name": (_col(r, "Name") or "")[:120], "ms": float(_col(r, "Total Time (ns)") or 0) / 1e6, "instances": _col(r, "Instances")}
        for r in kern[:15]
    ]
    return out


if __name__ == "__main__":
    print(json.dumps(breakdown(sys.argv[1]), indent=2))
