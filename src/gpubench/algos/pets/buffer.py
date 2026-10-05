"""Preallocated on-GPU transition ring buffer. add() / sample() never sync with the host."""

from __future__ import annotations

import torch


class GpuReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros(self.capacity, obs_dim, device=device)
        self.act = torch.zeros(self.capacity, act_dim, device=device)
        self.next_obs = torch.zeros(self.capacity, obs_dim, device=device)
        # 1.0 where next_obs is the true successor (not an auto-reset observation)
        self.valid = torch.zeros(self.capacity, device=device)
        self.ptr = 0  # host-side bookkeeping only; never read back from the device
        self.size = 0

    def add(self, obs, act, next_obs, valid):
        n = obs.shape[0]
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.next_obs[idx] = next_obs
        self.valid[idx] = valid.to(self.valid.dtype)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        return idx

    def sample_idx(self, shape) -> torch.Tensor:
        return torch.randint(0, self.size, shape, device=self.device)

    def recent_idx(self, n: int) -> torch.Tensor:
        """Indices of the last n inserted transitions (for out-of-sample model evaluation)."""
        n = min(n, self.size)
        return (self.ptr - n + torch.arange(n, device=self.device)) % self.capacity

    def get(self, idx):
        return self.obs[idx], self.act[idx], self.next_obs[idx], self.valid[idx]

    def all(self):
        s = slice(0, self.size)
        return self.obs[s], self.act[s], self.next_obs[s], self.valid[s]
