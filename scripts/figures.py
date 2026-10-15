"""Paper figures (step 5). Runs locally on pulled results, no GPU.

    python scripts/figures.py                       # results/ -> docs/paper/figs/
    python scripts/figures.py --draft               # same, with a DRAFT stamp

Figures (each written as .pdf + .png; a panel whose data is not there yet is drawn as a placeholder, and
figs/status.md lists which figures are complete; that is the gate-5 check):
    fig_breakdown   share of wall time per env step: GPU-busy per phase + GPU idle (Nsight, main_profile)
    fig_curves      return vs wall-clock and vs env steps, Cartpole and Ant (main benchmark)
    fig_waterfall   PETS-Cartpole wall-clock to tau, one optimisation at a time (step5_ablation) + planner ladder
    fig_crossover   wall-clock to tau vs physics substep multiplier k: PPO at several N vs PETS / MBPO fast
    fig_xgpu        planner throughput and wall-clock to tau across GPUs (xgpu + main + bench_planner)
    fig_scaling     PPO env count, PETS ensemble size, planner samples x horizon (step5_scaling + bench)

Host time is not used for the breakdown: under CUDA graphs it lands on whichever phase next waits on the GPU
(F14). Time to tau uses the analysis rule (first of two consecutive log rows >= tau); runs that never get
there are censored. Medians treat censored seeds as +inf, so a median is shown only if most seeds solved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

ROOT = A.ROOT
GPUS = ["rtx4090", "a100", "h100"]
GPU_LABEL = {"rtx4090": "RTX 4090", "a100": "A100", "h100": "H100"}
GPU_COLOR = {"rtx4090": A.SERIES[6], "a100": A.SERIES[3], "h100": A.SERIES[4]}
LEVELS = ["naive", "batched", "fused", "graph", "bf16", "fast"]
LEVEL_LABEL = {"naive": "naive", "batched": "+ batched\nensemble", "fused": "+ fused\nTriton step",
               "graph": "+ CUDA\ngraphs", "bf16": "+ bf16", "fast": "+ graphed\nmodel training"}
CROSS_EXPS = ["step5_crossover_cartpole", "step5_crossover_ant", "step5_scaling", "main_rtx4090"]
ANT_FRACS = (0.3, 0.5)  # pre-registered (protocol v0.9): fractions of tau for the Ant crossover
HEADLINE = {"E": 5, "K": 200, "H": 36, "B": 32}  # bench_planner headline config (tuned PETS-Cartpole)
METHODS = [("ppo", "reference", "PPO"), ("pets", "naive+tuned", "PETS naive"), ("pets", "fast+tuned", "PETS fast"),
           ("mbpo", "naive+tuned", "MBPO naive"), ("mbpo", "fast+tuned", "MBPO fast")]


# ---------------------------------------------------------------------------------------------- helpers
def load(results: Path, gpu: str, exps: list[str]) -> list[dict]:
    d = results / gpu
    have = [e for e in exps if (d / e).is_dir()]
    return A.load_runs(d, have) if have else []


def thresholds() -> dict:
    return yaml.safe_load((ROOT / "configs" / "thresholds.yaml").read_text())


def time_to(run: dict, tau: float) -> float:
    """Wall-clock seconds to tau, +inf if the run never held it (censored)."""
    w, _ = A.crossing(run["rows"], tau)
    return float(w) if w is not None else float("inf")


def med_ci(times, n=2000, seed=0):
    """Median with a bootstrap 95% CI; censored seeds (+inf) are kept, so the median is inf if most failed."""
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.median(t))
    if t.size == 1:
        return m, m, m
    boots = np.sort(np.median(t[np.random.default_rng(seed).integers(0, t.size, (n, t.size))], axis=1))
    return m, float(boots[int(0.025 * (n - 1))]), float(boots[int(0.975 * (n - 1))])  # order stats: inf-safe


def crossover_k(ks, t_ppo, t_mb) -> float | None:
    """First k at which the model-based time drops below PPO's, by linear interpolation of
    log(t_ppo / t_mb) against log2 k between the bracketing grid points. Returns the smallest k if the
    model-based method is already faster there, None if it never gets faster in the sampled range.
    Infinite times are allowed: inf for PPO counts as 'model-based wins', inf for MB as 'PPO wins'."""
    ks, a, b = np.asarray(ks, float), np.asarray(t_ppo, float), np.asarray(t_mb, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log(a) - np.log(b)  # > 0: model-based faster
    ok = ~np.isnan(d)
    ks, d = ks[ok], d[ok]
    if len(ks) and d[0] > 0:
        return float(ks[0])  # model-based already faster at the smallest k sampled
    for i in range(len(ks) - 1):
        d0, d1 = d[i], d[i + 1]
        if np.isnan(d0) or np.isnan(d1) or not (d0 <= 0 < d1):
            continue
        x0, x1 = np.log2(ks[i]), np.log2(ks[i + 1])
        if not np.isfinite(d0) or not np.isfinite(d1):
            return float(2 ** ((x0 + x1) / 2))  # one side censored: midpoint of the bracket
        return float(2 ** (x0 + (x1 - x0) * (-d0) / (d1 - d0)))
    return None


def plain_log_y(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v >= 1e-3 else f"{v:.0e}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def placeholder(ax, msg: str):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(A.GRID)
        s.set_linestyle((0, (4, 3)))
    ax.text(0.5, 0.5, f"no data yet\n{msg}", ha="center", va="center", color=A.INK2, fontsize=9,
            transform=ax.transAxes)


class Fig:
    """Collects completeness per figure for status.md."""

    def __init__(self, out: Path, draft: bool):
        self.out, self.draft, self.status = out, draft, {}

    def save(self, fig, name: str, missing: list[str]):
        import matplotlib.pyplot as plt

        if self.draft:
            fig.text(0.995, 0.005, "DRAFT", ha="right", va="bottom", color=A.INK2, fontsize=8, alpha=0.6)
        for ext in ("png", "pdf"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        self.status[name] = missing


# ---------------------------------------------------------------------------------------------- 1 breakdown
def fig_breakdown(F: Fig, results: Path, gpu="rtx4090"):
    import matplotlib.pyplot as plt

    prof = results / gpu / "profiles"
    tasks = ["cartpole", "ant"]
    rows_spec = [("ppo", None, "PPO"), ("pets", "naive", "PETS naive"), ("pets", "fast", "PETS fast"),
                 ("mbpo", "naive", "MBPO naive"), ("mbpo", "fast", "MBPO fast")]
    phases = ["sim", "plan", "model_train", "policy_update", "policy_act"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True, sharey=True)
    missing = []
    for ax, task in zip(axes, tasks):
        y, found = 0, False
        labels = []
        for algo, ver, lab in rows_spec:
            f = prof / (f"prof_{algo}_{task}.breakdown.json" if ver is None else f"prof_{algo}_{ver}_{task}.breakdown.json")
            labels.append(lab)
            if not f.exists():
                missing.append(f"{task}/{lab}")
                y += 1
                continue
            found = True
            d = json.loads(f.read_text())
            ph = {k: v for k, v in d.items() if not k.startswith("_") and k != "top_kernels"}
            steps = max(1, (ph.get("sim") or {}).get("calls") or 1)
            window = d["_total"]["window_ms"]
            busy = {p: (ph.get(p) or {}).get("gpu_busy_ms", 0.0) for p in phases}
            other = sum(v.get("gpu_busy_ms", 0.0) for k, v in ph.items() if k not in phases)
            segs = [(p, busy[p]) for p in phases] + [("other", other)]
            idle = max(0.0, window - sum(v for _, v in segs))
            left = 0.0
            for p, v in segs + [("GPU idle", idle)]:
                w = v / window
                color = A.PHASE_COLOR.get(p, "#b5b3ad") if p != "GPU idle" else "#ffffff"
                ax.barh(y, w, left=left, color=color, edgecolor=A.SURFACE if p != "GPU idle" else A.GRID,
                        hatch="////" if p == "GPU idle" else None, linewidth=0.8,
                        label=p if (task == tasks[0] and y == 0) or False else None)
                left += w
            ax.annotate(f"{window / steps:.1f} ms/step", (1.0, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=A.INK2, annotation_clip=False)
            y += 1
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("share of wall time (GPU busy by phase, rest GPU idle)")
        ax.set_title(f"{task}", loc="left")
        if not found:
            placeholder(ax, f"profiles/prof_*_{task}.breakdown.json")
    axes[0].invert_yaxis()  # shared y: invert once
    from matplotlib.patches import Patch

    handles = [Patch(color=A.PHASE_COLOR.get(p, "#b5b3ad"), label=p) for p in phases + ["other"]]
    handles.append(Patch(facecolor="#ffffff", edgecolor=A.GRID, hatch="////", label="GPU idle"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(f"Where an environment step's time goes (Nsight, {GPU_LABEL.get(gpu, gpu)})", x=0.01, ha="left",
                 fontweight="bold")
    F.save(fig, "fig_breakdown", missing)


# ---------------------------------------------------------------------------------------------- 2 curves
def fig_curves(F: Fig, results: Path, gpu="rtx4090"):
    import matplotlib.pyplot as plt

    runs = load(results, gpu, ["main_rtx4090"])
    th = thresholds()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    missing = []
    for col, task in enumerate(["cartpole", "ant"]):
        for row, (xkey, xlabel) in enumerate([("wall_s", "wall-clock time (s, log)"), ("env_steps", "env steps (log)")]):
            ax = axes[row, col]
            got = False
            for algo, impl, lab in METHODS:
                rs = [r for r in runs if r["task"] == task and r["algo"] == algo and r["impl"] == impl]
                if not rs:
                    if row == 0:
                        missing.append(f"{task}/{lab}")
                    continue
                curves = []
                for r in rs:
                    pts = [(x[xkey], x["reward_mean"]) for x in r["rows"] if x.get("reward_mean") is not None and x[xkey] > 0]
                    if len(pts) >= 2:
                        curves.append(tuple(map(np.asarray, zip(*pts))))
                mc = A._mean_curve(curves) if curves else None
                if not mc:
                    continue
                got = True
                g, m, lo, hi = mc
                color = A.ALGO_COLOR[algo]
                ls = (0, (4, 2)) if impl.startswith("naive") else "-"
                ax.fill_between(g, lo, hi, color=color, alpha=0.15, linewidth=0)
                ax.plot(g, m, color=color, linestyle=ls, label=f"{lab} (n={len(curves)})")
            if not got:
                placeholder(ax, f"{gpu}/main_rtx4090 {task}")
                continue
            ax.set_xscale("log")
            A._plain_log_ticks(ax)
            tau = th[task]["tau"]
            ax.axhline(tau, color=A.INK2, linestyle=(0, (4, 3)), linewidth=1)
            ax.annotate(f"τ = {tau:g}", (0.01, tau), xycoords=("axes fraction", "data"), xytext=(0, 4),
                        textcoords="offset points", color=A.INK2, fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("episode return")
            ax.set_title(f"{task}: return vs {'wall-clock' if row == 0 else 'samples'}", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=5)
    F.save(fig, "fig_curves", missing)


# ---------------------------------------------------------------------------------------------- 3 waterfall
def _bench(results: Path, gpu: str) -> list[dict]:
    f = results / gpu / "bench" / "planner.json"
    return json.loads(f.read_text())["runs"] if f.exists() else []


def _headline(bench: list[dict], level: str, task="cartpole") -> float | None:
    v = [r["planner_steps_per_s"] for r in bench if r["level"] == level and r["task"] == task
         and all(r.get(k) == v for k, v in HEADLINE.items())]
    return max(v) if v else None


def fig_waterfall(F: Fig, results: Path, gpu="rtx4090"):
    import matplotlib.pyplot as plt

    runs = [r for r in load(results, gpu, ["step5_ablation"]) if r["task"] == "cartpole" and r["algo"] == "pets"]
    tau = thresholds()["cartpole"]["tau"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [3, 2]})
    missing = []
    med = {}
    for lv in LEVELS:
        t = [time_to(r, tau) for r in runs if r["impl"].replace("+tuned", "") == lv]
        if t:
            med[lv] = med_ci(t)
        else:
            missing.append(f"ablation/{lv}")
    present = [lv for lv in LEVELS if lv in med and np.isfinite(med[lv][0])]
    if len(present) >= 2:
        prev = None
        top = max(med[lv][2] for lv in present)
        for i, lv in enumerate(present):
            m, lo, hi = med[lv]
            label = f"{m:.1f} s · {med[present[0]][0] / m:.1f}×"
            if prev is None or lv == present[-1]:  # full bars: start and end of the waterfall
                ax.bar(i, m, color=A.SERIES[1] if lv != present[-1] else A.SERIES[2], width=0.6)
                ax.annotate(label, (i, hi), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=8, color=A.INK)
            else:  # floating bar = time removed by this optimisation; result written under it
                saved = prev - m
                ax.bar(i, saved, bottom=m, color=A.SERIES[0] if saved >= 0 else A.SERIES[7], width=0.6)
                if abs(saved) > 0.06 * top:
                    ax.annotate(f"−{saved:.1f} s" if saved >= 0 else f"+{-saved:.1f} s", (i, m + saved / 2),
                                ha="center", va="center", fontsize=8, color="#ffffff")
                ax.annotate(label, (i, lo), xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                            fontsize=8, color=A.INK)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=A.INK, capsize=3, linewidth=1)
            if prev is not None:
                ax.plot([i - 1.3, i + 0.3], [prev, prev], color=A.INK2, linewidth=0.8, linestyle=(0, (2, 2)))
            prev = m
        ax.set_xticks(range(len(present)), [LEVEL_LABEL[lv] for lv in present], fontsize=8)
        ax.set_ylabel("wall-clock to τ (s, median, 95% CI)")
        ax.set_ylim(0, max(med[lv][2] for lv in present) * 1.25)
    else:
        placeholder(ax, f"{gpu}/step5_ablation (PETS cartpole, 6 levels)")
    ax.set_title("PETS-Cartpole: wall-clock to τ, one optimisation at a time", loc="left")
    bench = _bench(results, gpu)
    vals = [(lv, _headline(bench, lv)) for lv in LEVELS[:5]]
    vals = [(lv, v) for lv, v in vals if v]
    if vals:
        bx.bar(range(len(vals)), [v for _, v in vals], color=A.SERIES[0], width=0.6)
        for i, (_, v) in enumerate(vals):
            bx.annotate(f"{v / vals[0][1]:.1f}×", (i, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8)
        bx.set_yscale("log")
        plain_log_y(bx)
        bx.set_xticks(range(len(vals)), [LEVEL_LABEL[lv] for lv, _ in vals], fontsize=8)
        bx.set_ylabel("planner steps/s (log)")
    else:
        missing.append("bench/planner.json")
        placeholder(bx, f"{gpu}/bench/planner.json")
    bx.set_title("planner throughput (microbenchmark)", loc="left")
    F.save(fig, "fig_waterfall", missing)


# ---------------------------------------------------------------------------------------------- 4 crossover
def _sim_ms(runs) -> float | None:
    h = sum((x.get("host_ms") or {}).get("sim", 0.0) for r in runs for x in r["rows"])
    c = sum((x.get("calls") or {}).get("sim", 0) for r in runs for x in r["rows"])
    return h / c if c else None


def crossover_series(runs, task: str, tau: float) -> dict:
    """{label: {k: [times]}} for PPO (default config) at each N and for the fast model-based methods
    (tuned config only: ensemble/planner sweeps from step5_scaling are excluded)."""
    out: dict[str, dict[int, list[float]]] = {}
    for r in runs:
        if r["task"] != task:
            continue
        kn = A.knobs(r)
        k = int(kn["sim_substep_mult"])
        if r["algo"] == "ppo" and r["impl"] == "reference":
            lab = f"PPO N={kn['num_envs']}"
        elif r["algo"] in ("pets", "mbpo") and r["impl"] == "fast+tuned" and r["exp"] != "step5_scaling":
            lab = f"{r['algo'].upper()} fast"
        else:
            continue
        out.setdefault(lab, {}).setdefault(k, []).append(time_to(r, tau))
    return out


def fig_crossover(F: Fig, results: Path, gpu="rtx4090"):
    import matplotlib.pyplot as plt

    runs = load(results, gpu, CROSS_EXPS)
    th = thresholds()
    panels = [("cartpole", 1.0)] + [("ant", f) for f in ANT_FRACS]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), constrained_layout=True)
    missing, kstar = [], {}
    for ax, (task, frac) in zip(axes, panels):
        tau = th[task]["tau"] * frac
        ser = crossover_series(runs, task, tau)
        ks_all = sorted({k for s in ser.values() for k in s})
        if len(ks_all) < 2:
            missing.append(f"{task}@{frac:g}τ")
            placeholder(ax, f"{gpu}/step5_crossover_{task} (k = 1..16)")
            ax.set_title(f"{task}, {frac:g}·τ" if frac != 1 else task, loc="left")
            continue
        ppo_labels = sorted([s for s in ser if s.startswith("PPO")], key=lambda s: int(s.split("=")[1]))
        mb_labels = [s for s in ("PETS fast", "MBPO fast") if s in ser]
        shades = np.linspace(0.45, 1.0, max(1, len(ppo_labels)))
        cap = max((t for s in ser.values() for v in s.values() for t in v if np.isfinite(t)), default=1.0) * 1.6
        for lab in ppo_labels + mb_labels:
            ks = sorted(ser[lab])
            st = [med_ci(ser[lab][k]) for k in ks]
            if lab.startswith("PPO"):
                color, alpha = A.ALGO_COLOR["ppo"], shades[ppo_labels.index(lab)]
            else:
                color, alpha = A.ALGO_COLOR[lab.split()[0].lower()], 1.0
            m = np.array([s[0] for s in st])
            fin = np.isfinite(m)
            ax.plot(np.array(ks)[fin], m[fin], marker="o", color=color, alpha=alpha, label=lab)
            for k, (mm, lo, hi) in zip(ks, st):
                if np.isfinite(mm):
                    ax.plot([k, k], [lo, min(hi, cap)], color=color, alpha=alpha * 0.6, linewidth=1)
                else:  # most seeds never reached tau: open marker at the top
                    ax.plot([k], [cap], marker="^", markerfacecolor="none", color=color, alpha=alpha)
        # crossover of each model-based method against the strongest PPO at each k
        if ppo_labels and mb_labels:
            for lab in mb_labels:  # pairwise: the k values where this method and some PPO were both run
                ks = sorted(k for k in ser[lab] if any(k in ser[s] for s in ppo_labels))
                if len(ks) < 2:
                    continue
                best_ppo = [min(med_ci(ser[s][k])[0] for s in ppo_labels if k in ser[s]) for k in ks]
                t_mb = [med_ci(ser[lab][k])[0] for k in ks]
                ksx = crossover_k(ks, best_ppo, t_mb)
                kstar[f"{task}@{frac:g}τ/{lab}"] = ksx
                if ksx:
                    ax.axvline(ksx, color=A.ALGO_COLOR[lab.split()[0].lower()], linestyle=(0, (2, 2)), linewidth=1)
                    ax.axvspan(ksx, max(ks_all) * 1.4, color=A.ALGO_COLOR[lab.split()[0].lower()], alpha=0.06)
                    ax.annotate(f"k* ≈ {ksx:.1f}", (ksx, 0.98), xycoords=("data", "axes fraction"), xytext=(3, -3),
                                textcoords="offset points", va="top", fontsize=8, color=A.INK2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        plain_log_y(ax)
        ax.set_xticks(ks_all, [str(k) for k in ks_all])
        ax.set_xlabel("physics substeps per control step, relative (k)")
        ax.set_ylabel(f"wall-clock to {frac:g}·τ (s, median)" if frac != 1 else "wall-clock to τ (s, median)")
        ax.set_title(f"{task}, {frac:g}·τ" if frac != 1 else task, loc="left")
        ppo_ref = [r for r in runs if r["task"] == task and r["algo"] == "ppo" and r["impl"] == "reference"
                   and A.knobs(r)["num_envs"] == 4096]
        sim = {k: _sim_ms([r for r in ppo_ref if A.knobs(r)["sim_substep_mult"] == k]) for k in ks_all}
        if any(sim.values()):
            top = ax.secondary_xaxis("top")
            top.set_xticks(ks_all, [f"{sim[k]:.1f}" if sim[k] else "" for k in ks_all], fontsize=8)
            top.set_xlabel("sim step, ms (PPO, N=4096)", fontsize=8, color=A.INK2)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Crossover: when does a learned model beat parallel simulation on wall-clock?", x=0.01, ha="left",
                 fontweight="bold")
    F.save(fig, "fig_crossover", missing)
    return kstar


# ---------------------------------------------------------------------------------------------- 5 cross-GPU
def fig_xgpu(F: Fig, results: Path):
    import matplotlib.pyplot as plt

    th = thresholds()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    missing = []
    gpus = [g for g in GPUS if (results / g).is_dir()]
    # a: planner throughput per level, normalised to the 4090
    ax = axes[0, 0]
    base = {lv: _headline(_bench(results, "rtx4090"), lv) for lv in LEVELS[:5]}
    width = 0.8 / max(1, len(gpus))
    got = False
    for gi, g in enumerate(gpus):
        b = _bench(results, g)
        if not b:
            missing.append(f"{g}/bench")
            continue
        vals = [(_headline(b, lv) or np.nan) / (base[lv] or np.nan) for lv in LEVELS[:5]]
        if np.all(np.isnan(vals)):
            continue
        got = True
        ax.bar(np.arange(5) + gi * width, vals, width, color=GPU_COLOR[g], label=GPU_LABEL[g])
    if got:
        ax.axhline(1, color=A.INK2, linewidth=0.8)
        ax.set_xticks(np.arange(5) + width * (len(gpus) - 1) / 2, [LEVEL_LABEL[lv] for lv in LEVELS[:5]], fontsize=8)
        ax.set_ylabel("planner throughput vs RTX 4090")
        ax.legend(fontsize=8)
    else:
        placeholder(ax, "<gpu>/bench/planner.json")
    ax.set_title("a  planner throughput (launch-bound levels track the CPU)", loc="left")
    # b, c: wall-clock to tau per method; d: $ per solved task (cartpole)
    data = {}
    for g in gpus:
        exps = ["main_rtx4090"] if g == "rtx4090" else ["xgpu"]
        runs = [r for r in load(results, g, exps) if r["seed"] in (0, 1, 2)]
        if not runs:
            missing.append(f"{g}/{exps[0]}")
        data[g] = runs
    for ax, task, frac in [(axes[0, 1], "cartpole", 1.0), (axes[1, 0], "ant", ANT_FRACS[0])]:
        tau = th[task]["tau"] * frac
        meths = [mm for mm in METHODS if task == "cartpole" or mm[1] != "naive+tuned"]
        got = False
        for gi, g in enumerate(gpus):
            for mi, (algo, impl, lab) in enumerate(meths):
                t = [time_to(r, tau) for r in data.get(g, []) if r["task"] == task and r["algo"] == algo and r["impl"] == impl]
                if not t:
                    continue
                m, lo, hi = med_ci(t)
                x = mi + gi * width
                if not np.isfinite(m):  # most seeds never reached the threshold
                    ax.annotate("not\nreached", (x, 0.02), xycoords=("data", "axes fraction"), ha="center",
                                va="bottom", fontsize=7, color=GPU_COLOR[g])
                    continue
                got = True
                ax.bar(x, m, width, color=GPU_COLOR[g], label=GPU_LABEL[g] if mi == 0 or not got else None)
                ax.errorbar(x, m, yerr=[[m - lo], [max(0, min(hi, m * 10) - m)]], color=A.INK, capsize=2, linewidth=1)
        if got:
            ax.set_yscale("log")
            plain_log_y(ax)
            ax.set_xticks(np.arange(len(meths)) + width * (len(gpus) - 1) / 2, [mm[2] for mm in meths], fontsize=8)
            ax.set_ylabel(f"wall-clock to {frac:g}·τ (s, log)" if frac != 1 else "wall-clock to τ (s, log)")
            h, l = ax.get_legend_handles_labels()
            uniq = dict(zip(l, h))
            ax.legend(uniq.values(), uniq.keys(), fontsize=8)
        else:
            placeholder(ax, f"xgpu runs ({task})")
        ax.set_title(f"{'b' if task == 'cartpole' else 'c'}  {task}: wall-clock to {frac:g}·τ" if frac != 1
                     else "b  cartpole: wall-clock to τ", loc="left")
    ax = axes[1, 1]
    tau = th["cartpole"]["tau"]
    got = False
    for gi, g in enumerate(gpus):
        for mi, (algo, impl, lab) in enumerate(METHODS):
            rs = [r for r in data.get(g, []) if r["task"] == "cartpole" and r["algo"] == algo and r["impl"] == impl]
            usd = []
            for r in rs:
                dph = ((r["meta"].get("sys") or {}).get("vast") or {}).get("dph")
                w = time_to(r, tau)
                if dph and np.isfinite(w):
                    usd.append(w / 3600 * dph)
            if usd:
                got = True
                ax.bar(mi + gi * width, float(np.median(usd)), width, color=GPU_COLOR[g])
    if got:
        ax.set_yscale("log")
        plain_log_y(ax)
        ax.set_xticks(np.arange(len(METHODS)) + width * (len(gpus) - 1) / 2, [mm[2] for mm in METHODS], fontsize=8)
        ax.set_ylabel("$ per solved task (median, log)")
    else:
        placeholder(ax, "xgpu runs with Vast price")
    ax.set_title("d  cartpole: cost per solved task", loc="left")
    cpus = []
    for g in gpus:
        c = next((A.host_cpu(r, results / g) for r in data.get(g, [])), None)
        if c:
            cpus.append(f"{GPU_LABEL[g]}: {c}")
    fig.suptitle("Cross-GPU: same code, same configs" + (f"   (host CPU: {'; '.join(cpus)})" if cpus else ""),
                 x=0.01, ha="left", fontsize=9, color=A.INK2, fontweight="normal")
    if len([g for g in gpus if data.get(g)]) < 3:
        missing.append("fewer than 3 GPUs")
    F.save(fig, "fig_xgpu", missing)


# ---------------------------------------------------------------------------------------------- scaling
def fig_scaling(F: Fig, results: Path, gpu="rtx4090"):
    import matplotlib.pyplot as plt

    runs = load(results, gpu, ["step5_scaling", "main_rtx4090"])
    th = thresholds()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    missing = []
    ax = axes[0]
    got = False
    for task, ls in (("cartpole", "-"), ("ant", (0, (4, 2)))):
        by = {}
        for r in runs:
            if r["task"] == task and r["algo"] == "ppo" and r["impl"] == "reference" and A.knobs(r)["sim_substep_mult"] == 1:
                by.setdefault(A.knobs(r)["num_envs"], []).append(time_to(r, th[task]["tau"]))
        ns = sorted(by)
        if len(ns) >= 2:
            got = True
            st = [med_ci(by[n]) for n in ns]
            m = np.array([s[0] for s in st])
            ax.plot(np.array(ns)[np.isfinite(m)], m[np.isfinite(m)], marker="o", color=A.ALGO_COLOR["ppo"], linestyle=ls,
                    label=task)
    if got:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log"); plain_log_y(ax)
        ax.set_xlabel("parallel environments N")
        ax.set_ylabel("wall-clock to τ (s, median)")
        ax.legend(fontsize=8)
    else:
        missing.append("ppo num_envs")
        placeholder(ax, f"{gpu}/step5_scaling (PPO N sweep)")
    ax.set_title("PPO: env count", loc="left")
    tau = th["cartpole"]["tau"]
    pets = [r for r in runs if r["task"] == "cartpole" and r["algo"] == "pets" and r["impl"] == "fast+tuned"]
    for ax, key, xlabel, title in ((axes[1], "ensemble_size", "ensemble size E", "PETS fast: ensemble size"),
                                   (axes[2], "kh", "planner samples × horizon (K·H)", "PETS fast: samples × horizon")):
        by = {}
        for r in pets:
            kn = A.knobs(r)
            if key == "ensemble_size":
                if (kn["mppi_samples"], kn["mppi_horizon"]) != (HEADLINE["K"], HEADLINE["H"]):
                    continue
                x = kn["ensemble_size"]
            else:
                if kn["ensemble_size"] != HEADLINE["E"]:
                    continue
                x = (kn["mppi_samples"], kn["mppi_horizon"])
            by.setdefault(x, []).append(time_to(r, tau))
        if len(by) < 2:
            missing.append(f"pets {key}")
            placeholder(ax, f"{gpu}/step5_scaling (PETS {key})")
            ax.set_title(title, loc="left")
            continue
        xs = sorted(by, key=lambda v: v if key == "ensemble_size" else v[0] * v[1])
        st = [med_ci(by[x]) for x in xs]
        xv = [x if key == "ensemble_size" else x[0] * x[1] for x in xs]
        m = np.array([s[0] for s in st])
        ax.plot(np.array(xv)[np.isfinite(m)], m[np.isfinite(m)], marker="o", color=A.ALGO_COLOR["pets"],
                linestyle="-" if key == "ensemble_size" else "none")
        for x, xx, (mm, lo, hi) in zip(xs, xv, st):
            if np.isfinite(mm):
                ax.plot([xx, xx], [lo, hi], color=A.ALGO_COLOR["pets"], alpha=0.5, linewidth=1)
                if key == "kh":
                    ax.annotate(f"{x[0]}×{x[1]}", (xx, mm), xytext=(4, 4), textcoords="offset points", fontsize=7)
        if key == "kh":
            ax.set_xscale("log")
        ax.set_yscale("log"); plain_log_y(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("wall-clock to τ (s, median)")
        ax.set_title(title, loc="left")
    F.save(fig, "fig_scaling", missing)


# ---------------------------------------------------------------------------------------------- main
GATE_FIGS = ["fig_breakdown", "fig_curves", "fig_waterfall", "fig_crossover", "fig_xgpu"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "paper" / "figs")
    p.add_argument("--draft", action="store_true", help="stamp DRAFT on every figure")
    a = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    A._style()
    matplotlib.rcParams["axes.axisbelow"] = True  # grid behind bars
    a.out.mkdir(parents=True, exist_ok=True)
    F = Fig(a.out, a.draft)
    fig_breakdown(F, a.results)
    fig_curves(F, a.results)
    fig_waterfall(F, a.results)
    kstar = fig_crossover(F, a.results)
    fig_xgpu(F, a.results)
    fig_scaling(F, a.results)
    lines = ["| figure | gate | status | missing |", "|---|---|---|---|"]
    for name, miss in F.status.items():
        lines.append(f"| {name} | {'yes' if name in GATE_FIGS else '-'} | {'complete' if not miss else 'partial'} | "
                     f"{', '.join(miss) or '—'} |")
    exists = all((a.out / f"{n}.png").exists() for n in GATE_FIGS)
    complete = all(not F.status[n] for n in GATE_FIGS)
    lines.append(f"\n**Gate 5: {'MET' if exists and complete else 'NOT MET'}**: all five gate figures "
                 f"{'exist with data' if complete else 'exist, some panels are placeholders' if exists else 'missing'}.")
    if kstar:
        lines.append("\nCrossover k* (model-based faster than the best PPO for k >= k*):")
        lines += [f"- {k}: {'%.2f' % v if v else 'no crossing in the sampled k range'}" for k, v in kstar.items()]
    md = "\n".join(lines)
    (a.out / "status.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nfigures -> {a.out}")


if __name__ == "__main__":
    main()
