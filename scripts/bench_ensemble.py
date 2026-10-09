"""Microbenchmark: naive vs batched ensemble, and one full MPPI plan() with each (GPU, no simulator).

    python scripts/bench_ensemble.py                    # Ant-like sizes; writes results/bench/ensemble.json

Times use torch.cuda.synchronize around fixed repetition counts after warmup. The `plan` number is one
full planner call at the tuned step-2 PETS-Ant setting (B=32, K=800, H=12, 2 iterations, E=5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gpubench.algos.pets.ensemble_batched import BatchedEnsemble  # noqa: E402
from gpubench.algos.pets.ensemble_naive import NaiveEnsemble  # noqa: E402
from gpubench.algos.pets.mppi_naive import NaiveMPPI  # noqa: E402
from gpubench.common import sysinfo  # noqa: E402


def timed(fn, reps: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", type=int, default=36)
    p.add_argument("--act", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--E", type=int, default=5)
    p.add_argument("--batch", type=int, default=1024, help="model-training batch per member")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--samples", type=int, default=800)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--out", default=str(ROOT / "results" / "bench" / "ensemble.json"))
    a = p.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    naive = NaiveEnsemble(a.obs, a.act, a.E, a.hidden, a.layers).to(dev)
    batched = BatchedEnsemble(a.obs, a.act, a.E, a.hidden, a.layers).to(dev).load_from_naive(naive)
    cost = lambda s, u, s2: (s2[..., 0], torch.zeros(s2.shape[:-1], dtype=torch.bool, device=s2.device))  # noqa: E731
    n_part = a.envs * a.samples // a.E
    o, u = torch.randn(a.E, n_part, a.obs, device=dev), torch.randn(a.E, n_part, a.act, device=dev)
    ob, ub = torch.randn(a.E, a.batch, a.obs, device=dev), torch.randn(a.E, a.batch, a.act, device=dev)
    w = torch.ones(a.E, a.batch, device=dev)
    res = {"sizes": vars(a), "sys": {"gpu": sysinfo.collect()["gpu"].get("name"), "cpu": sysinfo.collect()["cpu"].get("model")}}
    for name, m in [("naive", naive), ("batched", batched)]:
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)

        def train_step():
            loss, _ = m.loss(ob, ub, ob, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred_ms = timed(lambda: m.predict(o, u), 50)
        train_ms = timed(train_step, 50)
        mp = NaiveMPPI(m, cost, a.envs, a.act, dev, horizon=a.horizon, samples=a.samples, iterations=2)
        obs0 = torch.randn(a.envs, a.obs, device=dev)
        plan_ms = timed(lambda: mp.plan(obs0), 20)
        res[name] = {"predict_ms": round(pred_ms, 3), "train_step_ms": round(train_ms, 3), "plan_ms": round(plan_ms, 3),
                     "planner_steps_per_s": round(a.envs / plan_ms * 1e3, 1),
                     "model_rollouts_per_s": round(2 * a.envs * a.samples * a.horizon / plan_ms * 1e3, 1)}
    res["speedup"] = {k: round(res["naive"][k] / res["batched"][k], 2) for k in ("predict_ms", "train_step_ms", "plan_ms")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
