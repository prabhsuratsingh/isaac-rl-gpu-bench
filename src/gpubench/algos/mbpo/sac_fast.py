"""CUDA-graphed SAC updates (MBPO `fast`). On Ant, naive MBPO spent 94% of its wall-clock here.

One SAC update is about 150 small kernels (three MLP forward/backward passes, three Adam steps, Polyak
averaging), so at batch 256 it is launch-bound. This captures U = updates_per_iter consecutive updates,
including the real/model minibatch sampling, as ONE CUDA graph per environment iteration:
  * indices are drawn in-graph from device size tensors: floor(rand(n) * size);
  * the actor, critic and alpha optimisers are rebuilt with capturable=True before any update;
  * gradients are allocated during warmup and zeroed in place (SAC.update(set_to_none=False));
  * loss statistics land in static tensors, read only at log time.
The batch composition (n_real real + n_model model transitions) is static, so capture happens on the
first update after the model buffer is non-empty (MBPO rolls out before its first SAC update).
"""

from __future__ import annotations

import torch

from gpubench.algos.pets.buffer import GpuReplayBuffer

from .sac import SAC


def make_capturable(sac: SAC):
    """Replace SAC's Adam optimisers with capturable ones (same hyperparameters; call before any step)."""
    for name, params in (("opt_actor", sac.actor.parameters()), ("opt_critic", sac.critic.parameters()),
                         ("opt_alpha", [sac.log_alpha])):
        old = getattr(sac, name)
        g = old.param_groups[0]
        setattr(sac, name, torch.optim.Adam(params, lr=g["lr"], betas=g["betas"], eps=g["eps"],
                                            weight_decay=g["weight_decay"], capturable=True))


class GraphedSACUpdater:
    def __init__(self, sac: SAC, real: GpuReplayBuffer, model_buf: GpuReplayBuffer, batch: int, real_ratio: float,
                 updates: int, warmup: int = 3):
        make_capturable(sac)
        self.sac, self.real, self.mbuf = sac, real, model_buf
        self.n_real = max(1, int(round(batch * real_ratio)))
        self.n_model = batch - self.n_real
        self.updates, self.warmup = int(updates), int(warmup)
        dev = real.device
        self.real_size_t = torch.ones((), device=dev)
        self.model_size_t = torch.ones((), device=dev)
        self.stats = {k: torch.zeros((), device=dev) for k in ("q_loss", "pi_loss", "alpha", "entropy")}
        self.fixed_idx: tuple[torch.Tensor, torch.Tensor] | None = None  # test hook: static (real_idx, model_idx)
        self.graph: torch.cuda.CUDAGraph | None = None

    def _batch(self):
        dev = self.real_size_t.device
        if self.fixed_idx is not None:
            ir, im = self.fixed_idx
        else:
            ir = (torch.rand(self.n_real, device=dev) * self.real_size_t).long()
            im = (torch.rand(self.n_model, device=dev) * self.model_size_t).long()
        br, bm = self.real.get_batch(ir), self.mbuf.get_batch(im)
        br["w"] = (br["valid"] + br["term"]).clamp(max=1.0)  # real: time-outs unusable, terminations fine
        bm["w"] = bm["valid"]
        return {k: torch.cat([br[k], bm[k]]) for k in ("obs", "act", "rew", "next_obs", "term", "w")}

    def _one(self):
        b = self._batch()
        out = self.sac.update(b["obs"], b["act"], b["rew"], b["next_obs"], b["term"], b["w"], set_to_none=False)
        for k, v in out.items():
            self.stats[k].copy_(v)

    def _many(self):
        for _ in range(self.updates):
            self._one()

    def update(self) -> int:
        """Run `updates` SAC updates (one graph replay after the first call). Returns updates done."""
        self.real_size_t.fill_(float(self.real.size))
        self.model_size_t.fill_(float(max(1, self.mbuf.size)))
        if self.graph is None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warmup = this iteration's real updates (+ Adam state, grads)
                for _ in range(self.warmup):
                    self._one()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._many()
            self.graph = g
            return self.warmup
        self.graph.replay()
        return self.updates

    def stats_view(self) -> dict:
        return dict(self.stats)
