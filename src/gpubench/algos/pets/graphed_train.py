"""CUDA-graphed dynamics-model training (PETS `fast` and MBPO `fast`).

After step 3, model training was about half of a PETS iteration and still launch-bound (about 13% GPU-busy):
each gradient step is a few dozen small kernels plus Adam. Here one full step

    sample a bootstrap minibatch per member -> loss -> backward -> Adam step

is captured once as a CUDA graph and replayed `steps` times.
  * Minibatch indices are drawn inside the graph: idx = floor(rand(E, batch) * size), with the buffer size
    in a device tensor (updated before each round), so a replay never syncs.
  * Adam is created with capturable=True (its step counters live on the GPU).
  * Gradients are allocated during capture and overwritten by every replay (whole-network capture,
    torch.cuda.graphs docs).
  * fit_normalizer runs before the replays and updates the normaliser buffers in place.
Holdout early stopping is not supported here (it needs host syncs); the tuned configs do not use it.
"""

from __future__ import annotations

import torch

from .buffer import GpuReplayBuffer


class GraphedTrainer:
    def __init__(self, model, buf: GpuReplayBuffer, batch: int, lr: float, weight_decay: float = 0.0, warmup: int = 3):
        self.model, self.buf, self.batch, self.warmup = model, buf, int(batch), int(warmup)
        dev = buf.device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, capturable=True)
        self.size_t = torch.ones((), device=dev)
        self.loss_t = torch.zeros((), device=dev)
        self.mse_t = torch.zeros((), device=dev)
        self.fixed_idx: torch.Tensor | None = None  # test hook: a static [E, batch] index tensor
        self.graph: torch.cuda.CUDAGraph | None = None

    def _idx(self):
        if self.fixed_idx is not None:
            return self.fixed_idx
        return (torch.rand(self.model.E, self.batch, device=self.size_t.device) * self.size_t).long()

    def _step(self):
        o, a, no, v = self.buf.get(self._idx())
        loss, mse = self.model.loss(o, a, no, v)
        loss.backward()
        self.opt.step()
        self.loss_t.copy_(loss.detach())
        self.mse_t.copy_(mse.detach())

    def _capture(self):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warmup: Adam state, cuBLAS workspaces, allocator
            for _ in range(self.warmup):
                self.opt.zero_grad(set_to_none=True)
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        self.opt.zero_grad(set_to_none=True)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        self.graph = g
        return self.warmup  # warmup steps were real training steps

    def train(self, steps: int):
        """One retraining round: refit the normaliser, then `steps` gradient steps. Returns (loss, mse, steps)."""
        self.model.fit_normalizer(*self.buf.all()[:4])
        self.model.train()
        self.size_t.fill_(float(self.buf.size))
        done = 0
        if self.graph is None:
            done = self._capture()
        for _ in range(max(0, steps - done)):
            self.graph.replay()
        self.model.eval()
        return self.loss_t.clone(), self.mse_t.clone(), max(steps, done)
