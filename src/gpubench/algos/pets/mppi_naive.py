"""Naive MPPI planner over a learned ensemble, batched over B real envs.

For each real env: K action sequences of length H, each propagated by P particles (TS-inf: a particle
keeps its ensemble member for the whole horizon). Cost = -sum of task reward from envs/costs.py on the
predicted states; rewards after a predicted termination are masked out.

"Naive" = Python loop over the horizon and (inside the ensemble) over the members, fp32, no CUDA graphs.
All tensors stay on the GPU; nothing here syncs with the host.

Particle layout (so that member chunks are contiguous without boolean indexing, which would sync):
    [B, K*P] -> view [B, K*P/E, E] -> permute [E, B, K*P/E] -> reshape [E, N/E]   (N = B*K*P)
i.e. particle j of an env uses member j % E, so the P particles of one sequence spread over members.

Optional knobs (defaults reproduce the step-1 planner exactly):
  action_repeat r  plan H/r coarse actions, each held for r model steps. Fewer decision variables and the
                   same model-step budget; the coarse plan shifts by one segment every r env steps.
  noise_decay d    sigma_i = sigma * d**i over the MPPI iterations of one plan() (coarse-to-fine).
  sigma            settable between calls (run.py anneals it over training).
"""

from __future__ import annotations

import torch


class NaiveMPPI:
    def __init__(self, model, cost_fn, num_envs: int, act_dim: int, device, horizon: int = 25, samples: int = 400,
                 particles: int = 1, iterations: int = 2, temperature: float = 0.5, noise_std: float = 0.5,
                 action_low: float = -1.0, action_high: float = 1.0, action_repeat: int = 1, noise_decay: float = 1.0):
        self.model, self.cost_fn = model, cost_fn
        self.B, self.A, self.H, self.K, self.P = num_envs, act_dim, horizon, samples, particles
        self.E = model.E
        if (self.K * self.P) % self.E:
            raise ValueError(f"samples*particles ({self.K * self.P}) must be divisible by ensemble size ({self.E})")
        if horizon % action_repeat:
            raise ValueError(f"horizon ({horizon}) must be divisible by action_repeat ({action_repeat})")
        self.iters, self.lam, self.sigma = iterations, temperature, noise_std
        self.r, self.decay = int(action_repeat), float(noise_decay)
        self.Hc = self.H // self.r  # coarse decision steps
        self.lo, self.hi = action_low, action_high
        self.device = device
        self.mean = torch.zeros(self.B, self.Hc, self.A, device=device)
        self._phase = 0  # env steps already executed of the current first coarse segment (host int, no sync)

    # --- layout helpers --------------------------------------------------------------------------
    def _to_members(self, x: torch.Tensor) -> torch.Tensor:
        """[B, K*P, d] -> [E, B*K*P/E, d]"""
        B, KP, d = x.shape
        return x.view(B, KP // self.E, self.E, d).permute(2, 0, 1, 3).reshape(self.E, -1, d)

    def _from_members(self, x: torch.Tensor) -> torch.Tensor:
        """[E, B*K*P/E, ...] -> [B, K*P, ...]  (inverse of _to_members)"""
        tail = x.shape[2:]
        x = x.reshape(self.E, self.B, -1, *tail)
        return x.permute(1, 2, 0, *range(3, 3 + len(tail))).reshape(self.B, -1, *tail)

    # --- planning --------------------------------------------------------------------------------
    def reset(self, env_mask: torch.Tensor | None = None):
        if env_mask is None:
            self.mean.zero_()
        else:  # zero the warm-start of envs that just reset (no host sync: arithmetic mask)
            self.mean.mul_((~env_mask).to(self.mean.dtype).view(-1, 1, 1))

    @torch.no_grad()
    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs [B, do], actions [B, K, H, A] -> expected return [B, K]"""
        B, K, H, A = actions.shape
        P = self.P
        s = obs[:, None, :].expand(B, K * P, obs.shape[-1])
        s = self._to_members(s.contiguous())
        a_all = actions[:, :, None].expand(B, K, P, H, A).reshape(B, K * P, H, A)
        a_all = self._to_members(a_all.reshape(B, K * P, H * A)).view(self.E, -1, H, A)
        ret = torch.zeros(s.shape[:2], device=s.device)
        alive = torch.ones_like(ret)
        for t in range(H):  # <- the naive part (H x E small launches)
            a_t = a_all[:, :, t]
            s_next = self.model.predict(s, a_t)
            r, term = self.cost_fn(s, a_t, s_next)
            ret = ret + alive * r
            alive = alive * (~term).to(alive.dtype)
            s = s_next
        ret = self._from_members(ret).view(B, K, P).mean(-1)
        return ret

    def _to_fine(self, coarse: torch.Tensor) -> torch.Tensor:
        """[B, K, Hc, A] -> [B, K, H, A]: hold each coarse action r steps, skipping the part of the first
        segment already executed (and holding the last action to keep H fixed)."""
        if self.r == 1:
            return coarse
        fine = coarse.repeat_interleave(self.r, dim=2)
        if self._phase:
            fine = torch.cat([fine[:, :, self._phase:], fine[:, :, -1:].expand(-1, -1, self._phase, -1)], dim=2)
        return fine

    @torch.no_grad()
    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [B, do] -> action [B, A]; warm-starts from the previous plan."""
        sigma = self.sigma
        for _ in range(self.iters):
            eps = torch.randn(self.B, self.K, self.Hc, self.A, device=self.device) * sigma
            eps[:, 0] = 0.0  # always evaluate the current mean sequence
            coarse = (self.mean[:, None] + eps).clamp(self.lo, self.hi)
            R = self.evaluate(obs, self._to_fine(coarse))  # [B, K]
            z = (R - R.max(dim=1, keepdim=True).values) / (self.lam * (R.std(dim=1, keepdim=True) + 1e-6))
            w = torch.softmax(z, dim=1)  # scale-free temperature: lam in units of return std
            self.mean = (w[:, :, None, None] * coarse).sum(1)
            sigma = sigma * self.decay
        act = self.mean[:, 0].clone()
        self._phase += 1
        if self._phase == self.r:  # first coarse segment fully executed: shift the plan
            self._phase = 0
            self.mean = torch.cat([self.mean[:, 1:], torch.zeros_like(self.mean[:, :1])], dim=1)
        return act

    @property
    def rollouts_per_plan(self) -> int:
        """Model transitions (particle-steps) evaluated per plan() call."""
        return self.iters * self.B * self.K * self.P * self.H
