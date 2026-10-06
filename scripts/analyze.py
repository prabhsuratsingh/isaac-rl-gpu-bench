"""Aggregate pulled results into the protocol's metrics, tables and figures (runs locally, no GPU).

    python scripts/analyze.py results/rtx4090                       # all experiments under that GPU
    python scripts/analyze.py results/rtx4090 --exps step1_ppo step1b_pets
    python scripts/analyze.py results/rtx4090 --derive-thresholds   # print tau per the protocol rule

Outputs (results/<gpu>/analysis/):
    runs.csv             one row per run: final reward, wall/samples to tau, throughput, cost, phase shares
    summary.csv / .md    per (task, algo, impl): mean +- 95% bootstrap CI over seeds, solved count
    curves_<task>.png    reward vs wall-clock and vs env steps (log x), mean + CI band + per-seed lines, tau
    time_breakdown.png   share of wall time per phase (from the runs' own timers, not the profiler)

Time-to-threshold = first log row where reward_mean >= tau and the next row is too (protocol section 4).
Runs that never get there are censored: counted in `solved`, excluded from the time statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"analysis", "checks", "profiles", "sanity", "step1_profile"}
PHASE_ORDER = ["sim", "plan", "model_train", "model_rollout", "policy_update", "policy_act", "storage", "buffer",
               "random_act", "other"]
LEARN_FRAC = 0.10  # "learns" (step-2 gate): CI lower bound > random return AND mean >= 10% of PPO's final return

# Reference categorical palette (dataviz skill references/palette.md), fixed order, light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ALGO_COLOR = {"ppo": SERIES[0], "pets": SERIES[1], "mbpo": SERIES[2], "dreamer": SERIES[3]}
PHASE_COLOR = {p: SERIES[i] for i, p in enumerate(["sim", "plan", "model_train", "policy_update", "policy_act",
                                                     "model_rollout"])}
PHASE_COLOR["other"] = "#b5b3ad"


# ---------------------------------------------------------------------------------------------- loading
def load_runs(gpu_dir: Path, exps: list[str] | None) -> list[dict]:
    runs = []
    for exp_dir in sorted(p for p in gpu_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if (exps and exp_dir.name not in exps) or (not exps and (exp_dir.name in SKIP_DIRS or "syncdebug" in exp_dir.name)):
            continue
        for run_dir in sorted(p for p in exp_dir.iterdir() if (p / "metrics.jsonl").exists()):
            rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl") if l.strip()]
            summ = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
            meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}
            if not rows:
                continue
            runs.append({"exp": exp_dir.name, "dir": run_dir, "rows": rows, "summary": summ, "meta": meta,
                         "algo": rows[0]["algo"], "impl": rows[0]["impl"], "task": rows[0]["task"], "seed": rows[0]["seed"]})
    return runs


def host_cpu(run: dict, gpu_dir: Path) -> str | None:
    """Host CPU of a run: recorded by sysinfo since step 1c; older runs via the pulled Vast instance record."""
    sys_ = run["meta"].get("sys", {}) or {}
    cpu = (sys_.get("cpu") or {}).get("model") or (sys_.get("vast") or {}).get("host_cpu")
    if cpu:
        return cpu
    cid = (sys_.get("vast") or {}).get("container_id")
    f = gpu_dir / "_instance" / str(cid) / "vast_instance.json"
    if cid and f.exists():
        return json.loads(f.read_text()).get("cpu_name")
    return None


def final_reward(rows, k=5):
    r = [x["reward_mean"] for x in rows if x.get("reward_mean") is not None]
    return float(np.mean(r[-k:])) if r else np.nan


def crossing(rows, tau):
    r = [x for x in rows if x.get("reward_mean") is not None]
    for a, b in zip(r, r[1:]):
        if a["reward_mean"] >= tau and b["reward_mean"] >= tau:
            return a["wall_s"], a["env_steps"]
    return None, None


def phase_shares(rows) -> dict:
    tot = Counter()
    for x in rows:
        for k, v in (x.get("host_ms") or {}).items():
            tot[k] += v
    wall_ms = rows[-1]["wall_s"] * 1e3
    shares = {k: v / wall_ms for k, v in tot.items()}
    shares["other"] = max(0.0, 1.0 - sum(shares.values()))
    return shares


def bootstrap_ci(x, n=10000, alpha=0.05, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return x.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)


# ---------------------------------------------------------------------------------------------- tables
def run_table(runs, thresholds, gpu_dir: Path | None = None) -> pd.DataFrame:
    out = []
    for r in runs:
        tau = (thresholds.get(r["task"]) or {}).get("tau")
        wall_tau, steps_tau = crossing(r["rows"], tau) if tau is not None else (None, None)
        dph = (r["meta"].get("sys", {}).get("vast", {}) or {}).get("dph")
        rows = r["rows"]
        sps = [x["env_steps_per_s"] for x in rows[len(rows) // 2:] if x.get("env_steps_per_s")]
        rec = {
            "exp": r["exp"], "task": r["task"], "algo": r["algo"], "impl": r["impl"], "seed": r["seed"],
            "status": r["summary"].get("status"), "env_steps": rows[-1]["env_steps"], "wall_s": rows[-1]["wall_s"],
            "final_reward": final_reward(rows), "tau": tau, "solved": wall_tau is not None,
            "wall_to_tau_s": wall_tau, "steps_to_tau": steps_tau,
            "gpu_h_to_tau": wall_tau / 3600 if wall_tau is not None else None,
            "usd_to_tau": wall_tau / 3600 * dph if (wall_tau is not None and dph) else None,
            "env_steps_per_s": float(np.median(sps)) if sps else None,
            "model_rollouts_per_s": float(np.median([x["model_rollouts_per_s"] for x in rows if x.get("model_rollouts_per_s")]))
            if any(x.get("model_rollouts_per_s") for x in rows) else None,
            "gpu": (r["meta"].get("sys", {}).get("gpu", {}) or {}).get("name"),
            "cpu": host_cpu(r, gpu_dir) if gpu_dir else None,
            "git_sha": (r["meta"].get("sys", {}) or {}).get("git_sha"),
        }
        rec.update({f"share_{k}": v for k, v in phase_shares(rows).items()})
        out.append(rec)
    return pd.DataFrame(out)


def random_reference(df: pd.DataFrame) -> dict:
    """Mean final return of the uniform-random policy per task (algo=random runs), if present."""
    r = df[df.algo == "random"]
    return {t: float(g.final_reward.mean()) for t, g in r.groupby("task")}


def summary_table(df: pd.DataFrame, thresholds: dict | None = None, random_ref: dict | None = None) -> pd.DataFrame:
    out = []
    for (task, algo, impl), g in df.groupby(["task", "algo", "impl"]):
        rec = {"task": task, "algo": algo, "impl": impl, "seeds": len(g), "solved": int(g.solved.sum())}
        for col in ["final_reward", "wall_to_tau_s", "steps_to_tau", "usd_to_tau", "env_steps_per_s"]:
            m, lo, hi = bootstrap_ci(g[col].tolist())
            rec[col], rec[col + "_lo"], rec[col + "_hi"] = m, lo, hi
        ppo_final = ((thresholds or {}).get(task) or {}).get("ppo_final_mean")
        rnd = (random_ref or {}).get(task)
        rec["random_return"], rec["learn_bar"] = rnd, (LEARN_FRAC * ppo_final if ppo_final else None)
        if algo != "random" and rnd is not None and ppo_final:
            lo = rec["final_reward_lo"] if np.isfinite(rec["final_reward_lo"]) else rec["final_reward"]
            rec["learns"] = bool(lo > rnd and rec["final_reward"] >= LEARN_FRAC * ppo_final)
        else:
            rec["learns"] = None
        out.append(rec)
    return pd.DataFrame(out)


def gate_markdown(s: pd.DataFrame, easy: str = "cartpole", loco: str = "ant") -> str:
    """Step-2 gate: all methods learn on the easy task; at least PPO and PETS learn on locomotion."""
    def learns(task, algo):
        r = s[(s.task == task) & (s.algo == algo)]
        return None if r.empty else bool(r.learns.fillna(False).any())

    lines = ["| task | method | final return | random | learn bar (10% PPO) | learns |", "|---|---|---|---|---|---|"]
    for _, r in s[s.algo != "random"].iterrows():
        lines.append(f"| {r.task} | {r.algo} ({r.impl}) | {_fmt(r.final_reward, r.final_reward_lo, r.final_reward_hi)} | "
                     f"{'—' if r.random_return is None else f'{r.random_return:.3g}'} | "
                     f"{'—' if r.learn_bar is None else f'{r.learn_bar:.3g}'} | "
                     f"{'—' if r.learns is None else ('yes' if r.learns else 'no')} |")
    easy_ok = all(learns(easy, m) for m in ("ppo", "pets", "mbpo"))
    loco_ok = all(learns(loco, m) for m in ("ppo", "pets"))
    mbpo_loco = learns(loco, "mbpo")
    lines.append("")
    lines.append(f"- all three learn on {easy}: **{'yes' if easy_ok else 'no'}**")
    lines.append(f"- PPO and PETS learn on {loco}: **{'yes' if loco_ok else 'no'}**")
    lines.append(f"- MBPO on {loco}: {'learns' if mbpo_loco else 'does not learn -> dropped on locomotion (gate rule)'}")
    lines.append(f"\n**Gate: {'MET' if easy_ok and loco_ok else 'NOT MET'}** (learns = 95% CI lower bound of the final "
                 f"return above the random policy's return, and mean >= {LEARN_FRAC:.0%} of PPO's final return)")
    return "\n".join(lines)


def _fmt(m, lo, hi, digits=3):
    if m is None or not np.isfinite(m):
        return "—"
    f = lambda v: f"{v:,.{digits}g}" if abs(v) < 1e5 else f"{v:.2e}"  # noqa: E731
    return f"{f(m)} [{f(lo)}, {f(hi)}]" if np.isfinite(lo) else f(m)


def _tex_num(m, lo, hi):
    if m is None or not np.isfinite(m):
        return "--"

    def f(v):
        a = abs(v)
        if a >= 1e5:
            e = int(np.floor(np.log10(a)))
            return f"{v / 10**e:.2f}\\times10^{{{e}}}"
        if a >= 1000:
            return f"{v:,.0f}".replace(",", "{,}")
        return f"{v:.3g}"

    return f"${f(m)}$ {{\\scriptsize [${f(lo)}$, ${f(hi)}$]}}" if np.isfinite(lo) else f"${f(m)}$"


def summary_latex(s: pd.DataFrame) -> str:
    """booktabs tabular, \\input-able from docs/paper/findings.tex (needs \\usepackage{booktabs})."""
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"task & method & solved & final return & wall-clock to $\tau$ (s) & env steps to $\tau$ & \$ to $\tau$ \\",
             r"\midrule"]
    for _, r in s.iterrows():
        cells = [r.task, f"{r.algo.upper()} ({r.impl})", f"{r.solved}/{r.seeds}"] + [
            _tex_num(r[c], r[c + "_lo"], r[c + "_hi"]) for c in ["final_reward", "wall_to_tau_s", "steps_to_tau", "usd_to_tau"]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def summary_markdown(s: pd.DataFrame) -> str:
    lines = ["| task | method | solved | final reward | wall-clock to τ (s) | env steps to τ | $ to τ | env steps/s |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.task} | {r.algo} ({r.impl}) | {r.solved}/{r.seeds} | "
                     + " | ".join(_fmt(r[c], r[c + '_lo'], r[c + '_hi']) for c in
                                  ["final_reward", "wall_to_tau_s", "steps_to_tau", "usd_to_tau", "env_steps_per_s"]) + " |")
    lines.append("\nMean [95% bootstrap CI] over seeds; time statistics over solved seeds only (censored runs excluded).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    })


def _mean_curve(seed_curves, log=True, n=200):
    """seed_curves: list of (x[], y[]) -> grid, mean, lo, hi over the x-range covered by all seeds."""
    lo_x = max(c[0][0] for c in seed_curves)
    hi_x = min(c[0][-1] for c in seed_curves)
    if hi_x <= lo_x:
        return None
    grid = np.geomspace(lo_x, hi_x, n) if log else np.linspace(lo_x, hi_x, n)
    Y = np.stack([np.interp(grid, x, y) for x, y in seed_curves])
    m = Y.mean(0)
    if len(seed_curves) > 1:
        rng = np.random.default_rng(0)
        boots = Y[rng.integers(0, len(Y), size=(2000, len(Y)))].mean(1)
        lo, hi = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
    else:
        lo = hi = m
    return grid, m, lo, hi


def _plain_log_ticks(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def fmt(v, _):
        if v >= 1e6:
            return f"{v / 1e6:g}M"
        if v >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"

    lo, hi = ax.get_xlim()
    decades = np.log10(hi / lo) if lo > 0 else 9
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if decades <= 2.5 else (1.0,)))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_curves(runs, thresholds, out_dir: Path, gpu_label: str):
    import matplotlib.pyplot as plt

    for task in sorted({r["task"] for r in runs}):
        tr = [r for r in runs if r["task"] == task]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for ax, xkey, xlabel in [(axes[0], "wall_s", "wall-clock time (s, log)"), (axes[1], "env_steps", "env steps (log)")]:
            groups = {}
            for r in tr:
                groups.setdefault((r["algo"], r["impl"]), []).append(r)
            for (algo, impl), rs in sorted(groups.items()):
                if algo == "random":  # drawn as a reference line instead
                    continue
                color = ALGO_COLOR.get(algo, SERIES[4])
                curves = []
                for r in rs:
                    pts = [(x[xkey], x["reward_mean"]) for x in r["rows"] if x.get("reward_mean") is not None and x[xkey] > 0]
                    if len(pts) < 2:
                        continue
                    x, y = map(np.asarray, zip(*pts))
                    curves.append((x, y))
                    ax.plot(x, y, color=color, alpha=0.25, linewidth=1)
                mc = _mean_curve(curves) if curves else None
                label = f"{algo.upper()} ({impl}, n={len(curves)})"
                if mc:
                    g, m, lo, hi = mc
                    ax.fill_between(g, lo, hi, color=color, alpha=0.18, linewidth=0)
                    ax.plot(g, m, color=color, label=label)
                    ax.annotate(algo.upper(), (g[-1], m[-1]), xytext=(4, 0), textcoords="offset points", color=INK2,
                                fontsize=9, va="center")
                elif curves:
                    ax.plot([], [], color=color, label=label)
            tau = (thresholds.get(task) or {}).get("tau")
            ax.set_xscale("log")
            _plain_log_ticks(ax)
            if tau is not None:
                ax.axhline(tau, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
                ax.annotate(f"τ = {tau:g}", (0.01, tau), xycoords=("axes fraction", "data"), xytext=(0, 4),
                            textcoords="offset points", color=INK2, fontsize=9)
            rnd = [final_reward(r["rows"]) for r in tr if r["algo"] == "random"]
            if rnd:  # uniform-random policy: the floor of the "learns" criterion
                ax.axhline(float(np.mean(rnd)), color=INK2, linestyle=(0, (1, 2)), linewidth=1)
                ax.annotate("random", (0.99, float(np.mean(rnd))), xycoords=("axes fraction", "data"), xytext=(0, 4),
                            textcoords="offset points", ha="right", color=INK2, fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("episode return (recent episodes)")
        axes[0].set_title(f"{task}: reward vs wall-clock ({gpu_label})", loc="left")
        axes[1].set_title(f"{task}: reward vs samples", loc="left")
        # one shared legend under both panels: never collides with τ or the curve-end labels
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=max(1, len(labels)))
        fig.savefig(out_dir / f"curves_{task}.png", dpi=200)
        plt.close(fig)


def plot_breakdown(df: pd.DataFrame, out_dir: Path, gpu_label: str):
    import matplotlib.pyplot as plt

    df = df[df.algo != "random"]
    share_cols = [c for c in df.columns if c.startswith("share_")]
    g = df.groupby(["task", "algo", "impl"])[share_cols].mean().fillna(0.0)
    # phases under 3% everywhere (storage, buffer, random_act) fold into "other": one gray, not several
    phases = [p for p in PHASE_ORDER if p in PHASE_COLOR and p != "other" and f"share_{p}" in g.columns
              and g[f"share_{p}"].max() > 0.03]
    minor = [c[6:] for c in share_cols if c[6:] not in phases]
    g["share_other"] = g.get("share_other", 0) + g[[f"share_{p}" for p in minor if p != "other"]].sum(axis=1)
    if "other" not in phases:
        phases.append("other")
    labels = [f"{t} · {a.upper()} ({i})" for t, a, i in g.index]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(g) + 1.4), constrained_layout=True)
    left = np.zeros(len(g))
    ypos = np.arange(len(g))[::-1]
    for p in phases:
        vals = g[f"share_{p}"].to_numpy()
        ax.barh(ypos, vals, left=left, height=0.62, color=PHASE_COLOR.get(p, "#b5b3ad"), edgecolor=SURFACE, linewidth=2,
                label=p.replace("_", " "))
        for y, l, v in zip(ypos, left, vals):
            if v >= 0.08:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", color="#ffffff", fontsize=8.5, fontweight="bold")
        left += vals
    ax.set_yticks(ypos, labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Where the wall-clock goes ({gpu_label}, mean over seeds)", loc="left")
    ax.legend(ncol=len(phases), loc="upper left", bbox_to_anchor=(0, -0.08 if len(g) > 3 else -0.15), fontsize=9)
    fig.savefig(out_dir / "time_breakdown.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def main():
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252 (no 'τ')
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("gpu_dir", type=Path)
    p.add_argument("--exps", nargs="*")
    p.add_argument("--thresholds", type=Path, default=ROOT / "configs" / "thresholds.yaml")
    p.add_argument("--derive-thresholds", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--tau-frac", type=float,
                   help="secondary threshold: tau = FRAC x PPO final return (e.g. 0.8); outputs get a _frac<FRAC> suffix")
    p.add_argument("--paper", metavar="TAG", help="also copy figures + LaTeX table to docs/paper/generated/<gpu>_<TAG>_*")
    a = p.parse_args()

    runs = load_runs(a.gpu_dir, a.exps)
    if not runs:
        sys.exit(f"no runs found under {a.gpu_dir}")
    if a.derive_thresholds:
        out = {}
        for task in sorted({r["task"] for r in runs if r["algo"] == "ppo"}):
            finals = [final_reward(r["rows"]) for r in runs if r["algo"] == "ppo" and r["task"] == task]
            out[task] = {"tau": round(0.9 * float(np.mean(finals)), 2), "ppo_final_mean": round(float(np.mean(finals)), 2),
                         "seeds": len(finals)}
        print(yaml.safe_dump(out, sort_keys=False))
        return

    thresholds = yaml.safe_load(a.thresholds.read_text()) if a.thresholds.exists() else {}
    suffix = ""
    if a.tau_frac:  # secondary thresholds are derived from the same frozen PPO reference returns
        thresholds = {t: {**v, "tau": round(a.tau_frac * v["ppo_final_mean"], 2)} for t, v in thresholds.items()}
        suffix = f"_frac{a.tau_frac:.2f}"
    out_dir = a.gpu_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    if not a.no_plots and not suffix:
        for old in out_dir.glob("*.png"):  # figures are per invocation: never ship a stale task's plot
            old.unlink()
    df = run_table(runs, thresholds, a.gpu_dir)
    for task, g in df.groupby("task"):
        cpus = sorted(g.cpu.fillna("unknown").unique())
        if len(cpus) > 1:  # protocol section 8: one comparison = one host
            print(f"WARNING: {task} mixes host CPUs {cpus}; wall-clock comparisons across them are confounded "
                  f"(launch-bound code is CPU-sensitive). Compare within one instance.", file=sys.stderr)
    df.to_csv(out_dir / f"runs{suffix}.csv", index=False)
    s = summary_table(df, thresholds, random_reference(df))
    s.to_csv(out_dir / f"summary{suffix}.csv", index=False)
    md = summary_markdown(s)
    if not suffix and s.learns.notna().any():
        gate = gate_markdown(s)
        (out_dir / "gate_step2.md").write_text(gate + "\n", encoding="utf-8")
        print(gate + "\n")
    if suffix:
        md = f"Threshold: tau = {a.tau_frac:g} x PPO final return\n\n" + md
    (out_dir / f"summary{suffix}.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    if not a.no_plots and not suffix:
        _style()
        gpu_label = df.gpu.dropna().iloc[0].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "") if df.gpu.notna().any() else a.gpu_dir.name
        plot_curves(runs, thresholds, out_dir, gpu_label)
        plot_breakdown(df, out_dir, gpu_label)
        print(f"\nfigures -> {out_dir}")
    (out_dir / f"summary{suffix}.tex").write_text(summary_latex(s), encoding="utf-8")
    if a.paper:
        import shutil

        gen = ROOT / "docs" / "paper" / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        prefix = f"{a.gpu_dir.name}_{a.paper}"
        for f in ([] if suffix else list(out_dir.glob("*.png"))) + [out_dir / f"summary{suffix}.tex"]:
            shutil.copy2(f, gen / f"{prefix}_{f.name}")
        print(f"paper artefacts -> {gen} ({prefix}_*)")


if __name__ == "__main__":
    main()
