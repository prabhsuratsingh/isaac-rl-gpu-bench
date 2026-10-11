"""Planner microbenchmark for the step-3 ablation (GPU, no simulator).

    python scripts/bench_planner.py --out results/bench/planner.json

For every level, one plan() call at the tuned PETS-Cartpole setting and at Ant sizes, plus sweeps:
    E in {1,3,5,7}; samples x horizon in a grid; envs B in {8,32,128}.
Levels (cumulative): naive, batched, fused (Triton), graph (CUDA graphs), bf16.
Reference points (benchmark only): vmap (torch.func ensemble), compiled (torch.compile'd step), and an
FP8 GEMM probe (torch._scaled_mm) on one hidden layer.
Per call: CUDA-event latency (p50/p95); throughput = envs / latency (planner steps/s) and
iterations * envs * samples * horizon / latency (model rollouts/s).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gpubench.algos.pets.ensemble_batched import BatchedEnsemble  # noqa: E402
from gpubench.algos.pets.ensemble_naive import NaiveEnsemble  # noqa: E402
from gpubench.algos.pets.mppi_naive import NaiveMPPI  # noqa: E402
from gpubench.common import sysinfo  # noqa: E402
from gpubench.envs.costs import ANT_ACTION_SLICE, make_cost  # noqa: E402

DEV = "cuda"
COSTS = {
    "cartpole": (4, 1, SimpleNamespace(rew_scale_alive=1.0, rew_scale_terminated=-2.0, rew_scale_pole_pos=-1.0,
                                       rew_scale_cart_vel=-0.01, rew_scale_pole_vel=-0.005, max_cart_pos=3.0), None, 200),
    "ant": (36, 8, SimpleNamespace(heading_weight=0.5, up_weight=0.1, actions_cost_scale=0.005, energy_cost_scale=0.05,
                                   alive_reward_scale=0.5, death_cost=-2.0, termination_height=0.31, decimation=2),
            ANT_ACTION_SLICE, 256),
}
LEVELS = ["naive", "batched", "fused", "graph", "bf16"]


# ---------------------------------------------------------------------------------------------- reference impls
class VmapEnsemble(torch.nn.Module):
    """torch.func.vmap over the naive members' stacked parameters ("vmap" option of the batched level)."""

    def __init__(self, naive: NaiveEnsemble):
        super().__init__()
        from torch.func import functional_call, stack_module_state

        self.naive = naive
        self.E, self.obs_dim, self.action_obs_slice = naive.E, naive.obs_dim, naive.action_obs_slice
        self.params, self.buffers_ = stack_module_state(list(naive.members))
        base = naive.members[0]

        def f(p, b, x):
            return functional_call(base, (p, b), (x,))

        self.vf = torch.vmap(f)

    def predict(self, obs, act, sample=True, eps=None):
        n = self.naive
        x = (torch.cat([obs, act], -1) - n.in_mean) / n.in_std
        mean, logvar = self.vf(self.params, self.buffers_, x)
        noise = torch.randn_like(mean) if eps is None else eps
        nxt = obs + (mean + noise * (0.5 * logvar).exp()) * n.out_std + n.out_mean
        if self.action_obs_slice is not None:
            nxt[..., self.action_obs_slice] = act
        return nxt


class CompiledMPPI(NaiveMPPI):
    """Reference: the per-step model+cost+accumulate function compiled by torch.compile (Inductor/Triton)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        model, cost = self.model, self.cost_fn

        def step(s, a_t, ret, alive):
            s_next = model.predict(s, a_t)
            r, term = cost(s, a_t, s_next)
            return s_next, ret + alive * r, alive * (~term).to(alive.dtype)

        self._step = torch.compile(step, dynamic=False)

    @torch.no_grad()
    def evaluate(self, obs, actions, model_noise=None):
        B, K, H, A = actions.shape
        P = self.P
        s = self._to_members(obs[:, None, :].expand(B, K * P, obs.shape[-1]).contiguous())
        a_all = self._to_members(actions[:, :, None].expand(B, K, P, H, A).reshape(B, K * P, H * A)).view(self.E, -1, H, A)
        ret = torch.zeros(s.shape[:2], device=s.device)
        alive = torch.ones_like(ret)
        for t in range(H):
            s, ret, alive = self._step(s, a_all[:, :, t].contiguous(), ret, alive)
        return self._from_members(ret).view(B, K, P).mean(-1)


# ---------------------------------------------------------------------------------------------- bench core
def make_planner(level, task, E, K, H, B, hidden, layers=4, r=1, iters=2):
    from gpubench.algos.pets.graphs import GraphedPlanner
    from gpubench.algos.pets.mppi_fused import FusedMPPI

    do, da, cfg, sl, _ = COSTS[task]
    torch.manual_seed(0)
    naive = NaiveEnsemble(do, da, E, hidden, layers, sl).to(DEV)
    cost = make_cost(task, cfg)
    kw = dict(horizon=H, samples=K, iterations=iters, noise_std=0.3, action_repeat=r)
    if level == "naive":
        return NaiveMPPI(naive, cost, B, da, DEV, **kw)
    if level == "vmap":
        return NaiveMPPI(VmapEnsemble(naive), cost, B, da, DEV, **kw)
    batched = BatchedEnsemble(do, da, E, hidden, layers, sl).to(DEV).load_from_naive(naive)
    if level == "batched":
        return NaiveMPPI(batched, cost, B, da, DEV, **kw)
    if level == "compiled":
        return CompiledMPPI(batched, cost, B, da, DEV, **kw)
    dtype = torch.bfloat16 if level == "bf16" else torch.float32
    p = FusedMPPI(batched, cost, B, da, DEV, dtype=dtype, **kw)
    return p if level == "fused" else GraphedPlanner(p)


def bench(level, task, E=5, K=200, H=36, B=32, r=3, reps=30, warmup=5, hidden=None):
    do = COSTS[task][0]
    hidden = hidden or COSTS[task][4]
    K = E * math.ceil(K / E)  # samples must be divisible by the ensemble size
    if H % r:
        r = 1
    p = make_planner(level, task, E, K, H, B, hidden, r=r)
    obs = torch.randn(B, do, device=DEV) * 0.2
    t0 = time.perf_counter()
    for _ in range(warmup):  # Triton JIT, CUDA-graph capture, torch.compile all happen here
        p.plan(obs)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - t0
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    t0 = time.perf_counter()
    for s, e in ev:
        s.record()
        p.plan(obs)
        e.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / reps * 1e3
    lat = sorted(s.elapsed_time(e) for s, e in ev)
    p50, p95 = lat[len(lat) // 2], lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    rollouts = 2 * B * K * H
    return {"level": level, "task": task, "E": E, "K": K, "H": H, "B": B, "r": r, "hidden": hidden,
            "p50_ms": round(p50, 4), "p95_ms": round(p95, 4), "wall_ms": round(wall_ms, 4),
            "planner_steps_per_s": round(B / wall_ms * 1e3, 1), "model_rollouts_per_s": round(rollouts / wall_ms * 1e3, 1),
            "setup_s": round(setup_s, 2)}


def fp8_probe(M=6400, k=256, n=256, E=5, reps=50):
    """One hidden layer for all members: fp32 bmm vs bf16 bmm vs fp8 _scaled_mm (per member). Error vs fp32."""
    out = {}
    x = torch.randn(E, M, k, device=DEV)
    w = torch.randn(E, k, n, device=DEV) / math.sqrt(k)

    def t(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    ref = torch.bmm(x, w)
    out["fp32_ms"] = round(t(lambda: torch.bmm(x, w)), 4)
    xb, wb = x.bfloat16(), w.bfloat16()
    out["bf16_ms"] = round(t(lambda: torch.bmm(xb, wb)), 4)
    out["bf16_rel_err"] = float(((torch.bmm(xb, wb).float() - ref).norm() / ref.norm()).item())
    try:
        f8 = torch.float8_e4m3fn
        sx = (x.abs().amax() / 448.0).float().reshape(1)
        sw = (w.abs().amax() / 448.0).float().reshape(1)
        x8 = (x / sx).to(f8)
        w8 = (w / sw).to(f8)
        w8t = [w8[e].t().contiguous().t() for e in range(E)]  # column-major operand required by _scaled_mm

        def fp8():
            return torch.stack([torch._scaled_mm(x8[e], w8t[e], scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
                                for e in range(E)])

        out["fp8_ms"] = round(t(fp8), 4)
        out["fp8_rel_err"] = float(((fp8().float() - ref).norm() / ref.norm()).item())
    except Exception as exc:  # unsupported GPU / torch build
        out["fp8_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "bench" / "planner.json"))
    ap.add_argument("--quick", action="store_true", help="headline configs only")
    a = ap.parse_args()
    info = sysinfo.collect()
    res = {"sys": {"gpu": info["gpu"].get("name"), "cpu": info["cpu"].get("model"), "torch": info["packages"].get("torch"),
                   "triton": info["packages"].get("triton"), "git_sha": info.get("git_sha")}, "runs": []}

    def run(level, task, **kw):
        try:
            r = bench(level, task, **kw)
        except Exception as exc:
            r = {"level": level, "task": task, **kw, "error": f"{type(exc).__name__}: {exc}"[:400]}
        res["runs"].append(r)
        print(json.dumps(r), flush=True)
        torch.cuda.empty_cache()

    # headline: the tuned PETS-Cartpole planner (H=36, r=3, K=200, B=32) and Ant sizes (H=12, K=800)
    for level in LEVELS + ["vmap", "compiled"]:
        run(level, "cartpole", E=5, K=200, H=36, B=32, r=3)
        run(level, "ant", E=5, K=800, H=12, B=32, r=1)
    if not a.quick:
        for level in LEVELS:
            for E in (1, 3, 5, 7):
                run(level, "cartpole", E=E, K=200, H=36, B=32, r=3, reps=20)
            for K, H in ((200, 12), (200, 36), (400, 25), (800, 12), (1600, 30)):
                run(level, "cartpole", E=5, K=K, H=H, B=32, r=1, reps=20)
            for B in (8, 128):
                run(level, "cartpole", E=5, K=200, H=36, B=B, r=3, reps=20)
    res["fp8_probe"] = fp8_probe()
    print("fp8 probe:", json.dumps(res["fp8_probe"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
