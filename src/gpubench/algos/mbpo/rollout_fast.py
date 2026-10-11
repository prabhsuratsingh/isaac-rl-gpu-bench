"""Fast MBPO branched rollouts (MBPO `fast`): batched ensemble, Triton step, bf16 GEMMs, CUDA graphs.

Naive rollout, per step:
    evaluate ALL E members on ALL n particles, then gather one member per particle (TS-1)
    -> E x the needed model FLOPs, plus about 20 small elementwise launches and a Python loop over members.
Fast rollout, per step:
    * TS-1 by random permutation: particles are shuffled (argsort of uniform noise) into E equal chunks
      and chunk e is propagated by member e. Every particle's member is still uniform per step (sampled
      without replacement across particles), and each member does only n/E rows, i.e. E x fewer FLOPs.
    * one batched GEMM chain (bf16, normalisation folded into the first layer);
    * the PETS Triton step kernel: soft-bounded log-variance, in-kernel sampling, next state, task reward
      and termination. Resetting `ret`/`alive` each step makes the kernel return the per-step reward
      (ret) and termination (alive == 0).
    * transitions go straight into the model buffer at a device-side ring pointer.
The whole k-step rollout (start-state sampling, actor, model, kernel, buffer writes) is captured as one
CUDA graph per rollout length k. The actor and model are updated in place by their optimisers, so
replays always use the current networks.
"""

from __future__ import annotations

import torch

from gpubench.algos.pets.buffer import GpuReplayBuffer
from gpubench.algos.pets.triton_kernels import launch_step


class FastRollout:
    def __init__(self, model, sac, cost_fn, real: GpuReplayBuffer, model_buf: GpuReplayBuffer, n: int,
                 dtype: torch.dtype = torch.bfloat16):
        E = model.E
        if n % E:
            raise ValueError(f"rollout_batch ({n}) must be divisible by the ensemble size ({E})")
        self.model, self.sac, self.cost, self.real, self.mbuf = model, sac, cost_fn, real, model_buf
        self.n, self.E, self.np, self.dtype = n, E, n // E, dtype
        dev = real.device
        do, A = model.obs_dim, model.act_dim
        self.do, self.A = do, A
        sl = model.action_obs_slice
        self.act_lo = sl.start if sl is not None else -1
        self.s_out = torch.zeros(E, self.np, do, device=dev)
        self.x_dummy = torch.zeros(E, self.np, do + A, device=dev, dtype=dtype)
        self.ret = torch.zeros(n, device=dev)
        self.alive_step = torch.ones(n, device=dev)
        self.real_size_t = torch.ones((), device=dev)
        self.seed_t = torch.randint(0, 2**31 - 1, (1,), device=dev, dtype=torch.int64)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None
        # test hooks: deterministic actor + external model noise / permutation
        self.deterministic_actor = False
        self.noise: list[torch.Tensor] | None = None
        self.perm: list[torch.Tensor] | None = None

    @torch.no_grad()
    def body(self, k: int):
        E, n, do, A = self.E, self.n, self.do, self.A
        dev = self.ret.device
        Ws, bs = self.model.folded_weights(self.dtype)
        idx = (torch.rand(n, device=dev) * self.real_size_t).long()
        s = self.real.obs[idx]
        alive = torch.ones(n, device=dev)
        for t in range(k):
            a = self.sac.act(s, deterministic=self.deterministic_actor)
            perm = self.perm[t] if self.perm is not None else torch.argsort(torch.rand(n, device=dev))
            sp = s[perm].view(E, self.np, do)
            ap = a[perm].view(E, self.np, A)
            alive = alive[perm]
            head = self.model.head_from_x(torch.cat([sp, ap], -1).to(self.dtype), Ws, bs)
            self.ret.zero_()
            self.alive_step.fill_(1.0)
            launch_step(head, self.model.max_logvar, self.model.min_logvar, self.model.out_mean, self.model.out_std,
                        sp, self.s_out, self.x_dummy, ap, None, self.ret, self.alive_step, self.seed_t,
                        rng_offset=t * n * do, task=self.cost.kind, params=self.cost.params, act_lo=self.act_lo,
                        noise=None if self.noise is None else self.noise[t])
            s2 = self.s_out.view(n, do).clone()
            term = self.alive_step < 0.5
            self.mbuf.add_at(sp.reshape(n, do), ap.reshape(n, A), s2, alive > 0, term, self.ret)
            alive = alive * self.alive_step
            s = s2
        self.seed_t.add_(1)

    def __call__(self, k: int) -> int:
        """k-step rollouts from n real start states into the model buffer. Returns the transition count."""
        self.real_size_t.fill_(float(self.real.size))
        self.mbuf.sync_ptr()
        if k not in self.graphs:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # this eager call is the round's real rollout (and the warmup)
                self.body(k)
            torch.cuda.current_stream().wait_stream(side)
            ptr_after = self.mbuf.ptr_t.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self.body(k)  # recorded, not executed
            self.pool = g.pool()
            self.graphs[k] = g
            self.mbuf.ptr_t.copy_(ptr_after)
        else:
            self.graphs[k].replay()
        self.mbuf.advance(self.n * k)
        return self.n * k
