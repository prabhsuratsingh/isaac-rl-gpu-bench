"""Figures + table from scripts/bench_planner.py output (runs locally).

    python scripts/plot_bench.py results/rtx4090/bench/planner.json [--paper step3]

Writes next to the JSON: bench_ladder.png (planner steps/s per level, both headline configs),
bench_latency_vs_E.png, bench_latency_vs_KH.png, bench_table.md / .tex (speed-up over naive).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import INK2, SERIES, _style  # noqa: E402

LEVELS = ["naive", "batched", "fused", "graph", "bf16"]
REFS = ["vmap", "compiled"]
LEVEL_COLOR = {lv: SERIES[i] for i, lv in enumerate(LEVELS)}


def headline(runs, task):
    if task == "cartpole":
        sel = [r for r in runs if r["task"] == task and r.get("H") == 36 and r.get("K") == 200 and r.get("B") == 32
               and r.get("E") == 5 and r.get("r") == 3 and "error" not in r]
    else:
        sel = [r for r in runs if r["task"] == task and "error" not in r]
    out = {}
    for r in sel:  # first occurrence per level = the headline config (sweeps come later)
        out.setdefault(r["level"], r)
    return out


def table(runs) -> tuple[str, str]:
    md = ["| config | level | p50 latency (ms) | planner steps/s | model rollouts/s | speed-up vs naive |",
          "|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"config & level & p50 latency (ms) & planner steps/s & model rollouts/s & speed-up \\", r"\midrule"]
    for task, label in (("cartpole", "Cartpole (tuned: K=200, H=36, r=3, B=32)"), ("ant", "Ant sizes (K=800, H=12, B=32)")):
        h = headline(runs, task)
        base = h.get("naive", {}).get("planner_steps_per_s")
        for lv in LEVELS + REFS:
            if lv not in h:
                continue
            r = h[lv]
            sp = r["planner_steps_per_s"] / base if base else float("nan")
            md.append(f"| {label} | {lv} | {r['p50_ms']:.2f} | {r['planner_steps_per_s']:,.0f} | {r['model_rollouts_per_s']:.3g} | {sp:.1f}x |")
            tex.append(f"{task} & {lv}{' (ref.)' if lv in REFS else ''} & {r['p50_ms']:.2f} & {r['planner_steps_per_s']:,.0f}".replace(",", "{,}")
                       + f" & ${r['model_rollouts_per_s'] / 1e6:.1f}$M & {sp:.1f}$\\times$ \\\\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    return "\n".join(md), "\n".join(tex) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_json", type=Path)
    ap.add_argument("--paper", metavar="TAG")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = json.loads(a.bench_json.read_text())
    runs = d["runs"]
    out = a.bench_json.parent
    import matplotlib.pyplot as plt

    _style()
    gpu = (d["sys"].get("gpu") or "").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")

    # ---- ladder ----------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, task, title in ((axes[0], "cartpole", "Cartpole (tuned planner)"), (axes[1], "ant", "Ant sizes")):
        h = headline(runs, task)
        lv = [x for x in LEVELS if x in h]
        vals = [h[x]["planner_steps_per_s"] for x in lv]
        base = vals[0] if lv and lv[0] == "naive" else None
        ax.bar(range(len(lv)), vals, color=[LEVEL_COLOR[x] for x in lv], width=0.62, edgecolor="#fcfcfb", linewidth=2)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:,.0f}" + (f"\n{v / base:.1f}x" if base else ""), (i, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(lv)), lv)
        ax.set_ylabel("planner steps/s")
        ax.set_title(f"{title} — {gpu}", loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.18)
    fig.savefig(out / "bench_ladder.png", dpi=200)
    plt.close(fig)

    # ---- latency vs E and vs K*H -----------------------------------------------------------------
    # latency vs ensemble size: a line per level (E is a genuine numeric axis)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for lv in LEVELS:
        pts = sorted({(r["E"], r["p50_ms"]) for r in runs if r["task"] == "cartpole" and r["level"] == lv
                      and "error" not in r and r.get("H") == 36 and r.get("B") == 32 and r.get("r") == 3})
        if len(pts) >= 2:
            x, y = zip(*pts)
            ax.plot(x, y, marker="o", markersize=5, color=LEVEL_COLOR[lv], label=lv)
    ax.set_yscale("log")
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xlabel("ensemble size E")
    ax.set_ylabel("plan() latency, p50 (ms, log)")
    ax.set_title(f"Planner latency vs ensemble size — Cartpole, {gpu}", loc="left")
    ax.legend(ncol=3, fontsize=9)
    fig.savefig(out / "bench_latency_vs_E.png", dpi=200)
    plt.close(fig)

    # latency per (samples, horizon) configuration: grouped bars. K x H is not one numeric axis -- latency
    # of the launch-bound levels follows the horizon (launches per plan), not the number of model steps.
    sel = [r for r in runs if r["task"] == "cartpole" and "error" not in r and r.get("E") == 5 and r.get("B") == 32
           and r.get("r") == 1 and r["level"] in LEVELS]
    cfgs = sorted({(r["K"], r["H"]) for r in sel}, key=lambda kh: (kh[0] * kh[1], kh[1]))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    width = 0.8 / len(LEVELS)
    for j, lv in enumerate(LEVELS):
        lat = {(r["K"], r["H"]): r["p50_ms"] for r in sel if r["level"] == lv}
        xs = [i + (j - (len(LEVELS) - 1) / 2) * width for i in range(len(cfgs))]
        ax.bar(xs, [lat.get(c, np.nan) for c in cfgs], width=width, color=LEVEL_COLOR[lv], label=lv,
               edgecolor="#fcfcfb", linewidth=1)
    ax.set_xticks(range(len(cfgs)), [f"K={k}\nH={h}" for k, h in cfgs])
    ax.set_yscale("log")
    ax.set_ylabel("plan() latency, p50 (ms, log)")
    ax.set_title(f"Planner latency per samples x horizon — Cartpole, E=5, B=32, {gpu}", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, fontsize=9, loc="upper left")
    ax.margins(y=0.15)
    fig.savefig(out / "bench_latency_vs_KH.png", dpi=200)
    plt.close(fig)

    md, tex = table(runs)
    fp8 = d.get("fp8_probe", {})
    md += "\n\nFP8 probe (one hidden layer, all members): " + json.dumps(fp8)
    (out / "bench_table.md").write_text(md + "\n", encoding="utf-8")
    (out / "bench_table.tex").write_text(tex, encoding="utf-8")
    print(md)
    if a.paper:
        gen = ROOT / "docs" / "paper" / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        prefix = f"{out.parent.name}_{a.paper}"
        for f in ("bench_ladder.png", "bench_latency_vs_E.png", "bench_latency_vs_KH.png", "bench_table.tex"):
            shutil.copy2(out / f, gen / f"{prefix}_{f}")
        print(f"paper artefacts -> {gen} ({prefix}_bench_*)")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
