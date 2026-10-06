"""Shared metrics harness: every algorithm emits its JSONL rows through `Reporter.emit`, so the schema is
identical by construction (docs/protocol.md section 6).

Common keys in every row (null when not applicable):
    iteration, env_steps, env_steps_per_s,
    reward_mean, ep_len_mean, episodes, episodes_total, reward_window,
    model_rollouts_per_s, planner_steps_per_s, policy_updates_per_s,
    gpu_ms{}, host_ms{}, calls{}, mem_alloc_max_mb, mem_reserved_max_mb
plus algorithm-specific values flattened as "<algo>/<key>" (e.g. "pets/model_holdout_mse").

`emit` also enforces the per-run wall-clock cap (`algo.max_wall_s`, used by the tuning budget): it
returns True once the cap is reached, and every runner stops at that point.
"""

from __future__ import annotations

import math

import torch

from .timing import gpu_memory_stats

COMMON_KEYS = (
    "iteration", "env_steps", "env_steps_per_s",
    "reward_mean", "ep_len_mean", "episodes", "episodes_total", "reward_window",
    "model_rollouts_per_s", "planner_steps_per_s", "policy_updates_per_s",
    "gpu_ms", "host_ms", "calls",
)


def _scalar(v):
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class Reporter:
    def __init__(self, logger, timer, algo: str, num_envs: int, max_wall_s: float | None = None, verbose: bool = True,
                 stop_tau: float | None = None):
        self.logger, self.timer, self.algo = logger, timer, algo
        self.num_envs = num_envs
        self.max_wall_s = float(max_wall_s) if max_wall_s else None
        self.stop_tau = float(stop_tau) if stop_tau is not None else None
        self.verbose = verbose
        self._last_wall = logger.wall_s
        self._last_steps = 0
        self._above = 0
        self.rows = 0
        self.stopped_by_cap = False
        self.stopped_at_tau = False

    @classmethod
    def from_cfg(cls, cfg: dict, logger, timer, algo: str, num_envs: int, verbose: bool = True) -> "Reporter":
        a = cfg["algo"]
        return cls(logger, timer, algo, num_envs, max_wall_s=a.get("max_wall_s"), verbose=verbose,
                   stop_tau=a.get("stop_tau"))

    def rate(self, count: float, phase: str, t: dict) -> float | None:
        """count / host seconds spent in `phase` during this interval."""
        s = t["host_ms"].get(phase, 0.0) / 1e3
        return count / s if s > 0 and count else None

    def emit(self, env_steps: int, iteration: int, episode_stats: dict, *, timing: dict | None = None,
             model_rollouts: float = 0, model_rollout_phase: str = "plan", planner_steps: float = 0,
             policy_updates: float = 0, policy_update_phase: str = "policy_update", extras: dict | None = None) -> bool:
        """Write one row. Counts are totals for this interval; rates are computed against phase time."""
        t = timing if timing is not None else self.timer.flush()
        wall = self.logger.wall_s
        d_wall = max(wall - self._last_wall, 1e-9)
        row = {
            "iteration": iteration,
            "env_steps": int(env_steps),
            "env_steps_per_s": (env_steps - self._last_steps) / d_wall,
            "reward_mean": episode_stats.get("reward_mean"),
            "ep_len_mean": episode_stats.get("ep_len_mean"),
            "episodes": episode_stats.get("episodes"),
            "episodes_total": episode_stats.get("episodes_total"),
            "reward_window": episode_stats.get("reward_window"),
            "model_rollouts_per_s": self.rate(model_rollouts, model_rollout_phase, t),
            "planner_steps_per_s": self.rate(planner_steps, "plan", t),
            "policy_updates_per_s": self.rate(policy_updates, policy_update_phase, t),
            **{k: t[k] for k in ("gpu_ms", "host_ms", "calls")},
            **gpu_memory_stats(),
        }
        for k, v in (episode_stats or {}).items():  # e.g. reward_mean_interval
            if k not in row:
                row[k] = v
        for k, v in (extras or {}).items():
            row[f"{self.algo}/{k}"] = _scalar(v)
        self.logger.log(row)
        self._last_wall, self._last_steps = wall, env_steps
        self.rows += 1
        if self.verbose:
            r = row["reward_mean"]
            print(f"[{self.algo}] it {iteration} steps {int(env_steps)} R {r if r is None else round(r, 2)} "
                  f"sps {row['env_steps_per_s']:.0f} wall {wall:.0f}s", flush=True)
        if self.stop_tau is not None:  # same crossing rule as the analysis: two consecutive rows >= tau
            r = row["reward_mean"]
            self._above = self._above + 1 if (r is not None and r >= self.stop_tau) else 0
            if self._above >= 2:
                self.stopped_at_tau = True
                return True
        if self.max_wall_s is not None and wall >= self.max_wall_s:
            self.stopped_by_cap = True
            return True
        return False

    def summary(self) -> dict:
        return {"rows": self.rows, "stopped_by_wall_cap": self.stopped_by_cap, "max_wall_s": self.max_wall_s,
                "stopped_at_tau": self.stopped_at_tau, "stop_tau": self.stop_tau}
