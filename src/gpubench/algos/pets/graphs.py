"""CUDA-graph wrapper for FusedMPPI (step-3 ablation levels `graph` and `bf16`).

The whole plan() (all MPPI iterations x the full horizon: sampling, H x (GEMMs + fused step), softmax
update, warm-start shift, seed increment) is captured once per action-repeat phase and replayed with a
single launch. Why replays stay correct:
  * inputs are static buffers: obs is copied into `obs_static`; the warm-start mean, sigma and RNG seed
    live in FusedMPPI tensors that the graph reads and updates in place;
  * model weights and normaliser buffers are only ever updated in place (Adam steps, copy_), so a
    replay always uses the current model, and the folded first-layer weights are recomputed inside
    the graph;
  * reset(done) and the sigma setter act on the same persistent tensors outside the graph.
Capture happens lazily on the first plan() (after the first model fit), after a side-stream warmup that
compiles the Triton kernels and sizes cuBLAS workspaces; planner state is restored afterwards.
"""

from __future__ import annotations

import torch

from .mppi_fused import FusedMPPI


class GraphedPlanner:
    def __init__(self, planner: FusedMPPI, warmup: int = 3):
        self.p = planner
        self.warmup = warmup
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.obs_static = torch.zeros(planner.B, planner.model.obs_dim, device=planner.device)
        self.captures = 0

    # pass-throughs used by run.py
    def __getattr__(self, name):
        return getattr(self.p, name)

    @property
    def sigma(self):
        return self.p.sigma

    @sigma.setter
    def sigma(self, v):
        self.p.sigma = v

    def reset(self, env_mask=None):
        self.p.reset(env_mask)

    @property
    def rollouts_per_plan(self) -> int:
        return self.p.rollouts_per_plan

    @torch.no_grad()
    def capture(self):
        p = self.p
        saved_mean, saved_seed, saved_phase = p.mean.clone(), p.seed_t.clone(), p._phase
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for ph in range(p.r):
                for _ in range(self.warmup):
                    p.plan_body(self.obs_static, ph)
        torch.cuda.current_stream().wait_stream(side)
        pool = None
        for ph in range(p.r):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                p.plan_body(self.obs_static, ph)
            pool = g.pool()
            self.graphs[ph] = g
        p.mean.copy_(saved_mean)
        p.seed_t.copy_(saved_seed)
        p._phase = saved_phase
        self.captures += 1

    @torch.no_grad()
    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.graphs:
            self.capture()
        self.obs_static.copy_(obs)
        self.graphs[self.p._phase].replay()
        self.p.advance_phase()
        return self.p.act_out.clone()
