"""Low-overhead phase timing.

Every phase (sim, plan, model_train, policy_act, policy_update, ...) is bracketed by
  * a pair of CUDA events  -> GPU time of the work enqueued inside the phase
  * an NVTX range          -> the same phase shows up by name in Nsight Systems
  * perf_counter           -> host wall time spent inside the phase

CUDA events are only synchronised at `flush()` (once per log interval), so timing adds no per-step
host<->device syncs. GPU ms is the time between the two events on the current stream: for phases that
are GPU-bound it equals device busy time; for host-bound phases (Python overhead, launch latency) the
host wall time is the more honest number. Both are logged.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class PhaseTimer:
    def __init__(self, device: torch.device | str | None = None, enabled: bool = True, nvtx: bool = True):
        dev = torch.device(device) if device is not None else None
        self.cuda = enabled and torch.cuda.is_available() and (dev is None or dev.type == "cuda")
        self.enabled = enabled
        self.nvtx = nvtx and self.cuda
        self._pending: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._gpu_ms: dict[str, float] = defaultdict(float)
        self._host_s: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        if self.nvtx:
            torch.cuda.nvtx.range_push(name)
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._host_s[name] += time.perf_counter() - t0
            self._calls[name] += 1
            if self.cuda:
                end.record()
                self._pending[name].append((start, end))
            if self.nvtx:
                torch.cuda.nvtx.range_pop()

    def wrap(self, name: str, fn):
        """Return `fn` wrapped so that every call is timed as phase `name`."""

        def wrapped(*args, **kwargs):
            with self.phase(name):
                return fn(*args, **kwargs)

        wrapped.__wrapped__ = fn
        return wrapped

    def flush(self) -> dict:
        """Synchronise pending events and return cumulative-since-last-flush stats, then reset."""
        if self.cuda and self._pending:
            torch.cuda.synchronize()
            for name, pairs in self._pending.items():
                self._gpu_ms[name] += sum(s.elapsed_time(e) for s, e in pairs)
            self._pending.clear()
        out = {
            "gpu_ms": {k: round(v, 3) for k, v in self._gpu_ms.items()},
            "host_ms": {k: round(v * 1e3, 3) for k, v in self._host_s.items()},
            "calls": dict(self._calls),
        }
        self._gpu_ms.clear()
        self._host_s.clear()
        self._calls.clear()
        return out


@contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class ProfilerWindow:
    """Opens the cudaProfilerApi capture range for Nsight between two step counts.

    Used with `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`, so the
    profile covers only a steady-state window (no Isaac Sim startup, no JIT warmup).
    """

    def __init__(self, start: int | None, length: int | None):
        self.start = start
        self.stop = None if start is None or length is None else start + length
        self.active = False
        self.done = False

    def step(self, i: int) -> bool:
        """Call once per iteration with the iteration index. Returns True once the window has closed."""
        if self.start is None or self.done:
            return self.done
        if not self.active and i >= self.start:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            self.active = True
        elif self.active and i >= self.stop:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            self.active = False
            self.done = True
        return self.done


def gpu_memory_stats(device=None, reset: bool = True) -> dict:
    if not torch.cuda.is_available():
        return {}
    out = {
        "mem_alloc_max_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "mem_reserved_max_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
    }
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return out
