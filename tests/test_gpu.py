"""GPU-only equivalence tests for the step-3 ablation levels (run on the instance: pytest -m gpu).

Each level is checked against the previous one with *external* noise, so the planners consume identical
random draws:
    naive == batched == fused == graph   (fp32: tight tolerance)
    bf16 vs fp32                         (reported relative error, loose tolerance)
plus the Triton step kernel against the torch cost functions for both tasks, including terminations.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (run on the instance)")]

from gpubench.algos.pets.ensemble_batched import BatchedEnsemble  # noqa: E402
from gpubench.algos.pets.ensemble_naive import NaiveEnsemble  # noqa: E402
from gpubench.algos.pets.mppi_naive import NaiveMPPI  # noqa: E402
from gpubench.envs.costs import ANT_ACTION_SLICE, make_cost  # noqa: E402

DEV = "cuda"
CARTPOLE = SimpleNamespace(rew_scale_alive=1.0, rew_scale_terminated=-2.0, rew_scale_pole_pos=-1.0,
                           rew_scale_cart_vel=-0.01, rew_scale_pole_vel=-0.005, max_cart_pos=3.0)
ANT = SimpleNamespace(heading_weight=0.5, up_weight=0.1, actions_cost_scale=0.005, energy_cost_scale=0.05,
                      alive_reward_scale=0.5, death_cost=-2.0, termination_height=0.31, decimation=2)
TASKS = {"cartpole": (4, 1, CARTPOLE, None), "ant": (36, 8, ANT, ANT_ACTION_SLICE)}


def _models(task, E=5, hidden=64, layers=3, seed=0):
    do, da, cfg, sl = TASKS[task]
    torch.manual_seed(seed)
    naive = NaiveEnsemble(do, da, E, hidden, layers, sl).to(DEV)
    with torch.no_grad():  # non-trivial normaliser so the folded first layer is exercised
        naive.in_mean.normal_(0, 0.3)
        naive.in_std.uniform_(0.5, 1.5)
        naive.out_mean.normal_(0, 0.01)
        naive.out_std.uniform_(0.01, 0.1)
    batched = BatchedEnsemble(do, da, E, hidden, layers, sl).to(DEV).load_from_naive(naive)
    return naive, batched, make_cost(task, cfg), do, da


def _planner(cls, model, cost, da, **kw):
    base = dict(horizon=12, samples=50, iterations=2, noise_std=0.4, noise_decay=0.8, action_repeat=3)
    base.update(kw)
    return cls(model, cost, 8, da, DEV, **base)


def _noise(p, do, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_part = p.B * p.K * p.P // p.E
    return {"act": [torch.randn(p.B, p.K, p.Hc, p.A, device=DEV, generator=g) for _ in range(p.iters)],
            "model": [[torch.randn(p.E, n_part, do, device=DEV, generator=g) for _ in range(p.H)] for _ in range(p.iters)]}


@pytest.mark.parametrize("task", ["cartpole", "ant"])
def test_triton_step_matches_torch_cost(task):
    from gpubench.algos.pets.triton_kernels import launch_step

    _, batched, cost, do, da = _models(task)
    E, N = 5, 300
    torch.manual_seed(3)
    s = torch.randn(E, N, do, device=DEV) * 0.5
    if task == "cartpole":  # force some terminations
        s[:, :40, 2] = 3.5
        s[:, 40:80, 0] = 2.0
    else:
        s[:, :40, 0] = 0.1
        s[:, 40:80, 12:20] = 1.5  # joints at limit
    a_all = torch.rand(E, N, 2, da, device=DEV) * 2 - 1
    x = torch.cat([s, a_all[:, :, 0]], -1)
    head = torch.randn(E, N, 2 * do, device=DEV)
    noise = torch.randn(E, N, do, device=DEV)
    # torch reference: same math as BatchedEnsemble.predict + cost_fn + accumulation
    mean, lvr = head.chunk(2, -1)
    lv = batched.max_logvar - torch.nn.functional.softplus(batched.max_logvar - lvr)
    lv = batched.min_logvar + torch.nn.functional.softplus(lv - batched.min_logvar)
    nxt = s + (mean + noise * (0.5 * lv).exp()) * batched.out_std + batched.out_mean
    if batched.action_obs_slice is not None:
        nxt[..., batched.action_obs_slice] = a_all[:, :, 0]
    r_ref, t_ref = cost(s, a_all[:, :, 0], nxt)
    ret = torch.full((E * N,), 0.5, device=DEV)
    alive = torch.ones(E * N, device=DEV)
    alive[::7] = 0.0
    ret_ref = ret + alive * r_ref.reshape(-1)
    alive_ref = alive * (~t_ref.reshape(-1)).float()
    s_out, x_out = torch.empty_like(s), torch.empty_like(x)
    seed = torch.zeros(1, device=DEV, dtype=torch.int64)
    launch_step(head, batched.max_logvar, batched.min_logvar, batched.out_mean, batched.out_std, s, s_out, x_out,
                a_all[:, :, 0], a_all[:, :, 1], ret, alive, seed, 0, task, cost.params,
                ANT_ACTION_SLICE.start if task == "ant" else -1, noise=noise)
    torch.testing.assert_close(s_out, nxt, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_out[..., :do], nxt, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_out[..., do:], a_all[:, :, 1])
    torch.testing.assert_close(ret, ret_ref, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(alive, alive_ref)
    assert t_ref.any() and (~t_ref).any()


@pytest.mark.parametrize("task", ["cartpole", "ant"])
@pytest.mark.parametrize("elite", [None])  # elite refit: near-tied returns can swap one elite across levels
def test_levels_agree_with_external_noise(task, elite):
    from gpubench.algos.pets.mppi_fused import FusedMPPI

    naive, batched, cost, do, da = _models(task)
    obs = torch.randn(8, do, device=DEV) * 0.3
    planners = {"naive": _planner(NaiveMPPI, naive, cost, da, elite_frac=elite),
                "batched": _planner(NaiveMPPI, batched, cost, da, elite_frac=elite),
                "fused": _planner(FusedMPPI, batched, cost, da, elite_frac=elite)}
    acts = {k: [] for k in planners}
    for step in range(4):  # several calls: exercises warm start, phase cycling and the plan shift
        noise = _noise(planners["naive"], do, seed=step)
        for k, p in planners.items():
            acts[k].append(p.plan(obs, noise=noise))
    for k in ("batched", "fused"):
        for a_ref, a in zip(acts["naive"], acts[k]):
            torch.testing.assert_close(a, a_ref, atol=2e-4, rtol=1e-3, msg=f"{k} vs naive")
        torch.testing.assert_close(planners[k].mean, planners["naive"].mean, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("task", ["cartpole", "ant"])
def test_graph_replays_match_eager_fused(task):
    from gpubench.algos.pets.graphs import GraphedPlanner
    from gpubench.algos.pets.mppi_fused import FusedMPPI

    _, batched, cost, do, da = _models(task)
    eager = _planner(FusedMPPI, batched, cost, da)
    graphed = GraphedPlanner(_planner(FusedMPPI, batched, cost, da))
    graphed.p.seed_t.copy_(eager.seed_t)
    obs = torch.randn(8, do, device=DEV) * 0.3
    for step in range(7):  # > 2 x action_repeat: every phase graph replayed, seeds advance identically
        a_e = eager.plan(obs)
        a_g = graphed.plan(obs)
        torch.testing.assert_close(a_g, a_e, atol=1e-5, rtol=1e-5)
        if step == 3:  # model update between calls (in place, as Adam does) must be seen by the replay
            with torch.no_grad():
                batched.W[1].mul_(0.9)
                batched.out_std.mul_(1.1)
        if step == 4:
            done = torch.zeros(8, dtype=torch.bool, device=DEV)
            done[:3] = True
            eager.reset(done)
            graphed.reset(done)
    assert graphed.captures == 1
    graphed.sigma = 0.2
    assert abs(graphed.p.sigma_t.item() - 0.2) < 1e-7


@pytest.mark.parametrize("task", ["cartpole", "ant"])
def test_bf16_error_is_small(task):
    """bf16 planning GEMMs vs fp32 on identical noise: report and bound the return error."""
    from gpubench.algos.pets.mppi_fused import FusedMPPI

    _, batched, cost, do, da = _models(task)
    p32 = _planner(FusedMPPI, batched, cost, da)
    p16 = _planner(FusedMPPI, batched, cost, da, dtype=torch.bfloat16)
    obs = torch.randn(8, do, device=DEV) * 0.3
    noise = _noise(p32, do, seed=5)
    fine = p32._to_fine(torch.rand(8, p32.K, p32.Hc, da, device=DEV) * 2 - 1)
    W32, b32 = p32._folded_weights()
    W16, b16 = p16._folded_weights()
    R32 = p32.evaluate_fused(obs, fine, 0, W32, b32, noise["model"][0])
    R16 = p16.evaluate_fused(obs, fine, 0, W16, b16, noise["model"][0])
    rel = ((R16 - R32).abs().mean() / R32.abs().mean().clamp_min(1e-6)).item()
    print(f"\nbf16 vs fp32 mean relative return error ({task}): {rel:.2e}")
    assert rel < 0.01  # documented tolerance: returns within 1% on fixed sequences
    assert not math.isnan(rel)


def test_fused_elite_refit_runs():
    from gpubench.algos.pets.mppi_fused import FusedMPPI

    _, batched, cost, do, da = _models("cartpole")
    p = _planner(FusedMPPI, batched, cost, da, elite_frac=0.2, action_repeat=1)
    obs = torch.randn(8, do, device=DEV) * 0.3
    for _ in range(3):
        a = p.plan(obs)
    assert torch.isfinite(a).all() and a.abs().max() <= 1.0
