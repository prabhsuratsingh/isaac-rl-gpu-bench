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
        # 1.0 where the episode truly terminated (not a time-out). A terminated transition is usable for TD
        # learning even though its next_obs is a post-reset observation: the target never reads it.
        self.term = torch.zeros(self.capacity, device=device)
        self.rew = torch.zeros(self.capacity, device=device)  # used by MBPO's SAC; PETS recomputes rewards
        self.ptr = 0  # host-side bookkeeping only; never read back from the device
        self.size = 0

    def add(self, obs, act, next_obs, valid, term=None, rew=None):
        n = obs.shape[0]
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.next_obs[idx] = next_obs
        self.valid[idx] = valid.to(self.valid.dtype)
        # Optional fields: index_fill_ takes the scalar as a kernel argument. `self.rew[idx] = 0.0` would wrap
        # 0.0 in a CPU tensor and copy it host->device, which is a sync (caught by sync_debug=error in step 2).
        if term is None:
            self.term.index_fill_(0, idx, 0.0)
        else:
            self.term[idx] = term.to(self.term.dtype)
        if rew is None:
            self.rew.index_fill_(0, idx, 0.0)
        else:
            self.rew[idx] = rew.to(self.rew.dtype)
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

    def get_batch(self, idx) -> dict:
        return {"obs": self.obs[idx], "act": self.act[idx], "next_obs": self.next_obs[idx], "valid": self.valid[idx],
                "term": self.term[idx], "rew": self.rew[idx]}

    def all(self):
        s = slice(0, self.size)
        return self.obs[s], self.act[s], self.next_obs[s], self.valid[s]
