"""Batched probabilistic ensemble: all E members evaluated with one batched GEMM per layer.

Drop-in replacement for `NaiveEnsemble` (same interface, same math, same initialisation distribution).
The naive version runs E separate nn.Sequential members in a Python loop, E x layers small GEMMs plus
their launches. Here every layer is a single `torch.baddbmm` over stacked weights [E, in, out]. This is
the first systems optimisation of the Axis-D ablation ("batched ensemble"), pulled forward for the
PETS-on-Ant retry.

    predict(obs[E, N, do], act[E, N, da]) -> next_obs[E, N, do]     (member i handles slice i)
    loss(obs[E, M, do], act[E, M, da], next_obs[E, M, do], w[E, M]) -> (loss, mse)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ensemble_naive import _weighted_mean_std


class BatchedEnsemble(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, ensemble_size: int = 5, hidden: int = 200, layers: int = 4,
                 action_obs_slice: slice | None = None):
        super().__init__()
        self.E, self.obs_dim, self.act_dim = ensemble_size, obs_dim, act_dim
        self.action_obs_slice = action_obs_slice
        dims = [obs_dim + act_dim] + [hidden] * layers + [2 * obs_dim]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for i, o in zip(dims[:-1], dims[1:]):
            w = torch.empty(ensemble_size, i, o)
            b = torch.empty(ensemble_size, 1, o)
            bound = 1.0 / math.sqrt(i)
            for e in range(ensemble_size):  # same distribution as nn.Linear's default init
                nn.init.kaiming_uniform_(w[e].T, a=math.sqrt(5))
            nn.init.uniform_(b, -bound, bound)
            self.W.append(nn.Parameter(w))
            self.b.append(nn.Parameter(b))
        self.max_logvar = nn.Parameter(torch.full((ensemble_size, 1, obs_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((ensemble_size, 1, obs_dim), -10.0))
        for name, dim in [("in_mean", obs_dim + act_dim), ("in_std", obs_dim + act_dim), ("out_mean", obs_dim), ("out_std", obs_dim)]:
            self.register_buffer(name, torch.zeros(dim) if "mean" in name else torch.ones(dim))

    @torch.no_grad()
    def fit_normalizer(self, obs, act, next_obs, valid):
        inp = torch.cat([obs, act], -1)
        m, s = _weighted_mean_std(inp, valid)
        self.in_mean.copy_(m)
        self.in_std.copy_(s)
        m, s = _weighted_mean_std(next_obs - obs, valid)
        self.out_mean.copy_(m)
        self.out_std.copy_(s)

    def _forward(self, obs, act):
        """obs [E, N, do], act [E, N, da] -> (mean, logvar) [E, N, do], normalised delta space."""
        x = (torch.cat([obs, act], -1) - self.in_mean) / self.in_std
        n = len(self.W)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, w)
            if i < n - 1:
                x = F.silu(x)
        mean, logvar = x.chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def predict(self, obs: torch.Tensor, act: torch.Tensor, sample: bool = True, eps: torch.Tensor | None = None) -> torch.Tensor:
        mean, logvar = self._forward(obs, act)
        noise = torch.randn_like(mean) if eps is None else eps
        delta_n = mean + noise * (0.5 * logvar).exp() if sample else mean
        nxt = obs + delta_n * self.out_std + self.out_mean
        if self.action_obs_slice is not None:
            nxt = nxt.clone()
            nxt[..., self.action_obs_slice] = act
        return nxt

    def loss(self, obs, act, next_obs, w) -> tuple[torch.Tensor, torch.Tensor]:
        target = ((next_obs - obs) - self.out_mean) / self.out_std
        mean, logvar = self._forward(obs, act)
        nll = (((mean - target) ** 2) * torch.exp(-logvar) + logvar).mean(-1)  # [E, M]
        wsum = w.sum().clamp_min(1.0)
        total = (nll * w).sum() / wsum + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum()) / wsum
        mse = ((((mean - target) ** 2).mean(-1)) * w).sum() / wsum
        return total, mse

    @torch.no_grad()
    def eval_mse(self, obs, act, next_obs, w) -> torch.Tensor:
        target = ((next_obs - obs) - self.out_mean) / self.out_std
        E = self.E
        mean, _ = self._forward(obs.unsqueeze(0).expand(E, -1, -1), act.unsqueeze(0).expand(E, -1, -1))
        pred = mean.mean(0)
        return ((((pred - target) ** 2).mean(-1)) * w).sum() / w.sum().clamp_min(1.0)

    @torch.no_grad()
    def load_from_naive(self, naive) -> "BatchedEnsemble":
        """Copy weights from a NaiveEnsemble with the same shape (equivalence tests, warm starts)."""
        for e, member in enumerate(naive.members):
            linears = [m for m in member.net if isinstance(m, nn.Linear)]
            for l, lin in enumerate(linears):
                self.W[l][e].copy_(lin.weight.T)
                self.b[l][e, 0].copy_(lin.bias)
            self.max_logvar[e, 0].copy_(member.max_logvar)
            self.min_logvar[e, 0].copy_(member.min_logvar)
        for name in ("in_mean", "in_std", "out_mean", "out_std"):
            getattr(self, name).copy_(getattr(naive, name))
        return self
