"""Step-5 figure script: every figure renders from a fake results tree, missing data gives placeholders,
and the crossover interpolation finds a known crossing (CPU only)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import figures as FG  # noqa: E402


def _run(root: Path, gpu, exp, run_id, algo, impl, task, seed, t_tau, *, k=1, n=4096, E=5, K=200, H=36, dph=0.5):
    """A run whose return ramps linearly and crosses `t_tau` seconds in (never, if t_tau is None)."""
    d = root / gpu / exp / run_id
    d.mkdir(parents=True)
    tau = FG.thresholds()[task]["tau"]
    end = (t_tau or 100.0) * 2
    rows = []
    for i in range(1, 41):
        w = end * i / 40
        r = tau * 1.05 * w / t_tau if t_tau else tau * 0.2
        rows.append({"algo": algo, "impl": impl, "task": task, "seed": seed, "wall_s": w, "env_steps": i * 1000,
                     "reward_mean": min(r, tau * 1.1), "host_ms": {"sim": 5.0 * k}, "calls": {"sim": 1}})
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    (d / "summary.json").write_text(json.dumps({"status": "ok", "num_envs": n}))
    (d / "run_meta.json").write_text(json.dumps({"sys": {"vast": {"dph": dph}, "cpu": {"model": "Fake CPU"}}}))
    (d / "config.json").write_text(json.dumps({"task": {"sim_substep_mult": k}, "algo": {
        "num_envs": n, "model": {"ensemble_size": E}, "mppi": {"samples": K, "horizon": H}}}))


def _tree(root: Path):
    for task in ("cartpole", "ant"):
        for s in range(3):
            _run(root, "rtx4090", "main_rtx4090", f"ppo_{task}_s{s}", "ppo", "reference", task, s, 10 + s)
            for algo in ("pets", "mbpo"):
                for v, t in (("naive", 80), ("fast", 9 if algo == "pets" else 27)):
                    _run(root, "rtx4090", "main_rtx4090", f"{algo}_{v}_{task}_s{s}", algo, f"{v}+tuned", task, s,
                         t + s, n=32)
        # crossover: PPO slows ~k, MBPO barely (27 at k=1): crossing between k=2 and k=4
        for k in (2, 4, 8):
            for s in range(3):
                _run(root, "rtx4090", f"step5_crossover_{task}" + ("_fixed" if task == "ant" else ""), f"ppo_k{k}_{task}_s{s}", "ppo", "reference", task, s,
                     10 * k, k=k)
                _run(root, "rtx4090", f"step5_crossover_{task}" + ("_fixed" if task == "ant" else ""), f"mbpo_k{k}_{task}_s{s}", "mbpo", "fast+tuned", task,
                     s, 25 + k, k=k, n=32)
    for lv, t in zip(FG.LEVELS, (80, 40, 25, 15, 12, 9)):
        for s in range(3):
            _run(root, "rtx4090", "step5_ablation", f"{lv}_s{s}", "pets", f"{lv}+tuned", "cartpole", s, t + s / 10, n=32)
    for s in range(2):
        for n in (1024, 16384):
            _run(root, "rtx4090", "step5_scaling", f"ppo_n{n}_s{s}", "ppo", "reference", "cartpole", s, 5 + n / 1000, n=n)
        for E in (1, 3):
            _run(root, "rtx4090", "step5_scaling", f"pets_e{E}_s{s}", "pets", "fast+tuned", "cartpole", s, 7 + E, E=E, n=32)
        for K, H in ((100, 36), (200, 12)):
            _run(root, "rtx4090", "step5_scaling", f"pets_k{K}h{H}_s{s}", "pets", "fast+tuned", "cartpole", s, 9,
                 K=K, H=H, n=32)
    for g in ("a100", "h100"):
        for task in ("cartpole", "ant"):
            for s in range(3):
                for algo, impl, _ in FG.METHODS:
                    _run(root, g, "xgpu", f"{algo}_{impl[:4]}_{task}_s{s}", algo, impl, task, s, 12)
    for g, mult in (("rtx4090", 1.0), ("a100", 0.8), ("h100", 1.3)):
        b = root / g / "bench"
        b.mkdir(parents=True, exist_ok=True)
        runs = [{"level": lv, "task": "cartpole", **FG.HEADLINE, "planner_steps_per_s": 300 * (i + 1) * mult}
                for i, lv in enumerate(FG.LEVELS[:5])]
        (b / "planner.json").write_text(json.dumps({"runs": runs}))
    prof = root / "rtx4090" / "profiles"
    prof.mkdir(parents=True)
    for task in ("cartpole", "ant"):
        for name in [f"prof_ppo_{task}"] + [f"prof_{a}_{v}_{task}" for a in ("pets", "mbpo") for v in ("naive", "fast")]:
            (prof / f"{name}.breakdown.json").write_text(json.dumps({
                "_total": {"window_ms": 1000.0, "gpu_util": 0.4},
                "sim": {"gpu_busy_ms": 100.0, "calls": 200}, "plan": {"gpu_busy_ms": 300.0, "calls": 200}}))


def _main(monkeypatch, results, out, *extra):
    monkeypatch.setattr(sys, "argv", ["figures.py", "--results", str(results), "--out", str(out), *extra])
    FG.main()
    return (out / "status.md").read_text(encoding="utf-8")


def test_all_figures_render_and_gate_met_with_full_data(tmp_path, monkeypatch):
    _tree(tmp_path / "results")
    status = _main(monkeypatch, tmp_path / "results", tmp_path / "figs")
    for name in FG.GATE_FIGS + ["fig_scaling"]:
        assert (tmp_path / "figs" / f"{name}.png").stat().st_size > 10_000
        assert (tmp_path / "figs" / f"{name}.pdf").exists()
    assert "Gate 5: MET" in status, status
    assert "cartpole@1τ/MBPO fast: 2." in status or "cartpole@1τ/MBPO fast: 3." in status, status


def test_missing_data_gives_placeholders_not_crashes(tmp_path, monkeypatch):
    (tmp_path / "results" / "rtx4090").mkdir(parents=True)
    status = _main(monkeypatch, tmp_path / "results", tmp_path / "figs", "--draft")
    for name in FG.GATE_FIGS:
        assert (tmp_path / "figs" / f"{name}.png").exists()
    assert "Gate 5: NOT MET" in status
    assert "partial" in status


def test_crossover_interpolation():
    ks = [1, 2, 4, 8]
    ppo = [10, 20, 40, 80]  # doubles with k
    mb = [30, 30, 30, 30]   # flat: equal at k = 3 (log2 interpolation between 2 and 4 -> 2**1.585 ~= 3.0)
    assert FG.crossover_k(ks, ppo, mb) == pytest.approx(3.0, rel=0.02)
    assert FG.crossover_k(ks, [1, 1, 1, 1], mb) is None            # PPO always faster
    assert FG.crossover_k(ks, [40, 40, 40, 40], mb) == 1.0         # model-based faster from the start
    assert FG.crossover_k(ks, [10, 20, math.inf, math.inf], mb) == pytest.approx(2 ** 1.5)  # censored bracket


def test_median_ci_handles_censored_seeds():
    m, lo, hi = FG.med_ci([5.0, 6.0, math.inf])
    assert m == 6.0 and lo <= m and math.isinf(hi)
    assert math.isinf(FG.med_ci([5.0, math.inf, math.inf])[0])
    assert np.isnan(FG.med_ci([])[0])
