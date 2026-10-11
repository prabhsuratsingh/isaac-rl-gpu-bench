"""Soft Actor-Critic (Haarnoja et al. 2018), plain PyTorch, entirely on the GPU.

tanh-squashed Gaussian actor, twin Q critics with Polyak-averaged targets, automatic entropy temperature
(target entropy -|A|). `update` returns detached loss tensors: nothing in the update path calls .item(),
so it can run inside `torch.cuda.set_sync_debug_mode("error")`.

Every loss takes a per-sample weight `w` (0/1): MBPO uses it to drop real transitions whose next_obs is
an auto-reset observation (time-outs), without boolean indexing.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(i: int, o: int, hidden: int, layers: int) -> nn.Sequential:
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, layers):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden, layers)

    def forward(self, obs, deterministic: bool = False):
        mean, log_std = self.net(obs).chunk(2, -1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)
        # log N(u; mean, std) - sum log(1 - tanh(u)^2), the latter in a numerically stable form
        logp = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, layers):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden, layers)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden, layers)

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SAC(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, layers: int = 2, actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4, alpha_lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 init_alpha: float = 0.1, target_entropy: float | None = None):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden, layers)
        self.critic = TwinQ(obs_dim, act_dim, hidden, layers)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        self.gamma, self.tau = gamma, tau
        self.target_entropy = -float(act_dim) if target_entropy is None else target_entropy
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @torch.no_grad()
    def set_obs_norm(self, mean: torch.Tensor, std: torch.Tensor):
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std.clamp_min(1e-3))

    def _n(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        return self.actor(self._n(obs), deterministic)[0]

    def update(self, obs, act, rew, next_obs, done, w, set_to_none: bool = True) -> dict:
        """One gradient step on critic, actor and temperature. All inputs [N, ...] on device; w in {0,1}.
        set_to_none=False keeps gradient tensors alive and zeroes them in place (required inside a CUDA
        graph, sac_fast.GraphedSACUpdater)."""
        o, no = self._n(obs), self._n(next_obs)
        alpha = self.log_alpha.exp().detach()
        wsum = w.sum().clamp_min(1.0)

        with torch.no_grad():
            na, nlogp = self.actor(no)
            tq1, tq2 = self.critic_targ(no, na)
            target = rew + self.gamma * (1.0 - done) * (torch.min(tq1, tq2) - alpha * nlogp)
        q1, q2 = self.critic(o, act)
        q_loss = (((q1 - target) ** 2 + (q2 - target) ** 2) * w).sum() / wsum
        self.opt_critic.zero_grad(set_to_none=set_to_none)
        q_loss.backward()
        self.opt_critic.step()

        a_pi, logp = self.actor(o)
        q1p, q2p = self.critic(o, a_pi)
        pi_loss = ((alpha * logp - torch.min(q1p, q2p)) * w).sum() / wsum
        self.opt_actor.zero_grad(set_to_none=set_to_none)
        pi_loss.backward()
        self.opt_actor.step()

        alpha_loss = -(self.log_alpha * ((logp.detach() + self.target_entropy) * w)).sum() / wsum
        self.opt_alpha.zero_grad(set_to_none=set_to_none)
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.lerp_(p, self.tau)
        return {"q_loss": q_loss.detach(), "pi_loss": pi_loss.detach(), "alpha": alpha,
                "entropy": -(logp.detach() * w).sum() / wsum}
