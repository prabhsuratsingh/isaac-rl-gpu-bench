"""Fused MPPI planner (step-3 ablation level `fused`; with bf16 weights, level `bf16`).

Same algorithm and particle layout as `NaiveMPPI` (it subclasses it); only the execution changes:
  * ensemble = BatchedEnsemble (one baddbmm per layer for all members);
  * input normalisation folded into the first layer's weights once per plan() (no per-step normalise);
  * one Triton kernel per horizon step for everything after the GEMMs (triton_kernels.pets_step_kernel),
    which also writes the next model input in place (no torch.cat per step);
  * action sequences sampled by one Triton kernel per MPPI iteration;
  * every buffer preallocated and every update in place, so the whole plan() can be captured as a
    CUDA graph (graphs.GraphedPlanner). The plan body depends on the action-repeat phase, so it takes
    the phase as an argument (one graph per phase).

Randomness comes from in-kernel Philox streams keyed by a device seed tensor (incremented per plan),
so results are statistically, not bitwise, equivalent to the torch planners. With `noise=...` both
planners consume the same external noise and agree numerically (tests/test_gpu.py).
"""

from __future__ import annotations

import torch

from .mppi_naive import NaiveMPPI
from .triton_kernels import launch_sample, launch_step


class FusedMPPI(NaiveMPPI):
    def __init__(self, model, cost_fn, num_envs: int, act_dim: int, device, dtype: torch.dtype = torch.float32, **kw):
        super().__init__(model, cost_fn, num_envs, act_dim, device, **kw)
        if not hasattr(model, "W"):
            raise TypeError("FusedMPPI needs the BatchedEnsemble (stacked weights)")
        if getattr(cost_fn, "kind", None) is None:
            raise TypeError("cost_fn must expose .kind/.params (envs/costs.py)")
        self.dtype = dtype
        self.task, self.cparams = cost_fn.kind, cost_fn.params
        sl = model.action_obs_slice
        self.act_lo = sl.start if sl is not None else -1
        E, do, A = self.E, model.obs_dim, act_dim
        self.Np = self.B * self.K * self.P // E  # particles per member
        f32 = dict(device=device, dtype=torch.float32)
        self.s_buf = [torch.zeros(E, self.Np, do, **f32) for _ in range(2)]
        self.x_buf = [torch.zeros(E, self.Np, do + A, device=device, dtype=dtype) for _ in range(2)]
        self.ret = torch.zeros(E * self.Np, **f32)
        self.alive = torch.ones(E * self.Np, **f32)
        self.coarse = torch.zeros(self.B, self.K, self.Hc, A, **f32)
        self.a_all = torch.zeros(E, self.Np, self.H, A, **f32)
        self.act_out = torch.zeros(self.B, A, **f32)
        self.sigma_t = torch.tensor(float(kw.get("noise_std", 0.5)), **f32)
        self.seed_t = torch.randint(0, 2**31 - 1, (1,), device=device, dtype=torch.int64)

    # sigma is a device scalar so a captured graph sees updates (run.py anneals it between calls)
    @property
    def sigma(self):
        return self.sigma_t

    @sigma.setter
    def sigma(self, v):
        if hasattr(self, "sigma_t"):
            self.sigma_t.fill_(float(v))

    # ---- model ---------------------------------------------------------------------------------
    def _folded_weights(self):
        return self.model.folded_weights(self.dtype)

    @staticmethod
    def _head(x, Ws, bs):
        from .ensemble_batched import BatchedEnsemble

        return BatchedEnsemble.head_from_x(x, Ws, bs)

    # ---- planning ------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_fused(self, obs, fine, it: int, Ws, bs, model_noise=None):
        """obs [B, do], fine [B, K, H, A] -> expected return [B, K]."""
        B, K, H, A = fine.shape
        P, E = self.P, self.E
        s0 = self._to_members(obs[:, None, :].expand(B, K * P, obs.shape[-1]).contiguous())
        a = fine[:, :, None].expand(B, K, P, H, A).reshape(B, K * P, H * A)
        self.a_all.copy_(self._to_members(a).view(E, -1, H, A))
        do = s0.shape[-1]
        self.s_buf[0].copy_(s0)
        self.x_buf[0][..., :do].copy_(s0)
        self.x_buf[0][..., do:].copy_(self.a_all[:, :, 0])
        self.ret.zero_()
        self.alive.fill_(1.0)
        rows = E * self.Np
        for t in range(H):
            i_in, i_out = t % 2, (t + 1) % 2
            head = self._head(self.x_buf[i_in], Ws, bs)
            launch_step(head, self.model.max_logvar, self.model.min_logvar, self.model.out_mean, self.model.out_std,
                        self.s_buf[i_in], self.s_buf[i_out], self.x_buf[i_out], self.a_all[:, :, t],
                        self.a_all[:, :, t + 1] if t < H - 1 else None, self.ret, self.alive, self.seed_t,
                        rng_offset=(it * H + t) * rows * do, task=self.task, params=self.cparams, act_lo=self.act_lo,
                        noise=None if model_noise is None else model_noise[t])
        return self._from_members(self.ret.view(E, self.Np)).view(B, K, P).mean(-1)

    @torch.no_grad()
    def plan_body(self, obs, phase: int, noise: dict | None = None):
        """One plan() at a fixed action-repeat phase. Writes self.act_out; updates mean/seed in place."""
        Ws, bs = self._folded_weights()
        sigma = self.sigma_t
        for i in range(self.iters):
            launch_sample(self.mean, sigma, self.seed_t, self.coarse, rng_offset=2**30 + i * self.coarse.numel(),
                          lo=self.lo, hi=self.hi, noise=None if noise is None else noise["act"][i])
            saved = self._phase
            self._phase = phase
            fine = self._to_fine(self.coarse)
            self._phase = saved
            R = self.evaluate_fused(obs, fine, i, Ws, bs, None if noise is None else noise["model"][i])
            z = (R - R.max(dim=1, keepdim=True).values) / (self.lam * (R.std(dim=1, keepdim=True) + 1e-6))
            w = torch.softmax(z, dim=1)
            self.mean.copy_((w[:, :, None, None] * self.coarse).sum(1))
            if self.n_elite:
                top = R.topk(self.n_elite, dim=1).indices
                elites = self.coarse.gather(1, top[:, :, None, None].expand(-1, -1, self.Hc, self.A))
                sigma = (elites.std(dim=1).clamp_min(self.min_std)[:, None] * self.decay).contiguous()
            else:
                sigma = sigma * self.decay
        self.act_out.copy_(self.mean[:, 0])
        if phase == self.r - 1:  # shift the coarse plan in place
            self.mean[:, :-1].copy_(self.mean[:, 1:].clone())
            self.mean[:, -1].zero_()
        self.seed_t.add_(1)

    def advance_phase(self):
        self._phase = (self._phase + 1) % self.r

    @torch.no_grad()
    def plan(self, obs: torch.Tensor, noise: dict | None = None) -> torch.Tensor:
        self.plan_body(obs, self._phase, noise)
        self.advance_phase()
        return self.act_out.clone()

    def reset(self, env_mask: torch.Tensor | None = None):
        if env_mask is None:
            self.mean.zero_()
        else:
            self.mean.mul_((~env_mask).to(self.mean.dtype).view(-1, 1, 1))
