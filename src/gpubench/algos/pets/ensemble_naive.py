"""Naive probabilistic ensemble (PETS, Chua et al. 2018) in plain PyTorch.

"Naive" is deliberate and is the Axis-D baseline: E independent nn.Sequential members, evaluated in a
Python loop (E small GEMMs + E x layers kernel launches per call). The fused version (batched bmm /
vmap / Triton) will replace `NaiveEnsemble` behind the same interface:

    fit_normalizer(obs, act, next_obs, valid)
    predict(obs[E, N, do], act[E, N, da]) -> next_obs[E, N, do]     (member i handles slice i)
    loss(obs[E, M, do], act[E, M, da], next_obs[E, M, do], w[E, M]) -> scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weighted_mean_std(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    w = w.unsqueeze(-1)
    n = w.sum().clamp_min(1.0)
    mean = (x * w).sum(0) / n
    var = (((x - mean) ** 2) * w).sum(0) / n
    return mean, var.clamp_min(eps**2).sqrt()


class GaussianMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, 2 * out_dim))
        self.net = nn.Sequential(*mods)
        self.max_logvar = nn.Parameter(torch.full((out_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((out_dim,), -10.0))

    def forward(self, x):
        mean, logvar = self.net(x).chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


class NaiveEnsemble(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, ensemble_size: int = 5, hidden: int = 200, layers: int = 4,
                 action_obs_slice: slice | None = None):
        super().__init__()
        self.E, self.obs_dim, self.act_dim = ensemble_size, obs_dim, act_dim
        self.members = nn.ModuleList(GaussianMLP(obs_dim + act_dim, obs_dim, hidden, layers) for _ in range(ensemble_size))
        self.action_obs_slice = action_obs_slice
        for name, dim in [("in_mean", obs_dim + act_dim), ("in_std", obs_dim + act_dim), ("out_mean", obs_dim), ("out_std", obs_dim)]:
            self.register_buffer(name, torch.zeros(dim) if "mean" in name else torch.ones(dim))

    @torch.no_grad()
    def fit_normalizer(self, obs, act, next_obs, valid):
        inp = torch.cat([obs, act], -1)
        self.in_mean.copy_(_weighted_mean_std(inp, valid)[0])
        self.in_std.copy_(_weighted_mean_std(inp, valid)[1])
        m, s = _weighted_mean_std(next_obs - obs, valid)
        self.out_mean.copy_(m)
        self.out_std.copy_(s)

    def _member(self, i: int, obs, act):
        x = (torch.cat([obs, act], -1) - self.in_mean) / self.in_std
        return self.members[i](x)

    def predict(self, obs: torch.Tensor, act: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """obs [E, N, do], act [E, N, da] -> next_obs [E, N, do]; member i propagates obs[i]."""
        outs = []
        for i in range(self.E):  # <- the naive part
            mean, logvar = self._member(i, obs[i], act[i])
            delta_n = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
            outs.append(obs[i] + delta_n * self.out_std + self.out_mean)
        nxt = torch.stack(outs, 0)
        if self.action_obs_slice is not None:  # obs contains the last action: known exactly, don't predict it
            nxt[..., self.action_obs_slice] = act
        return nxt

    def loss(self, obs, act, next_obs, w) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian NLL on normalised deltas, per-member batches [E, M, *]. Returns (loss, mse)."""
        target = ((next_obs - obs) - self.out_mean) / self.out_std
        total = obs.new_zeros(())
        mse = obs.new_zeros(())
        wsum = w.sum().clamp_min(1.0)
        for i in range(self.E):
            mean, logvar = self._member(i, obs[i], act[i])
            nll = (((mean - target[i]) ** 2) * torch.exp(-logvar) + logvar).mean(-1)
            total = total + (nll * w[i]).sum()
            mse = mse + ((((mean - target[i]) ** 2).mean(-1)) * w[i]).sum()
            m = self.members[i]
            total = total + 0.01 * (m.max_logvar.sum() - m.min_logvar.sum())
        return total / wsum, mse / wsum

    @torch.no_grad()
    def eval_mse(self, obs, act, next_obs, w) -> torch.Tensor:
        """One-step normalised-delta MSE of the ensemble mean on the same batch for all members."""
        target = ((next_obs - obs) - self.out_mean) / self.out_std
        preds = torch.stack([self._member(i, obs, act)[0] for i in range(self.E)], 0).mean(0)
        return ((((preds - target) ** 2).mean(-1)) * w).sum() / w.sum().clamp_min(1.0)
