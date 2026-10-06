"""Uniform-random policy: the "beats random" reference of the step-2 gate (docs/protocol.md section 4).

Runs until `min_episodes` episodes have finished (and at least `min_env_steps`), logging through the
shared Reporter like every other method. The reward window is widened to `min_episodes`, so the final
row averages the whole sample rather than the last min(100, num_envs) episodes.
"""

from __future__ import annotations

import torch

from gpubench.common.reporter import Reporter


def run(cfg: dict, logger, venv, env_cfg, timer):
    a = cfg["algo"]
    B, da = venv.num_envs, venv.act_dim
    rep = Reporter.from_cfg(cfg, logger, timer, "random", B)
    venv.reset()
    it = 0
    finished = 0
    while True:
        with timer.phase("policy_act"):
            act = torch.rand(B, da, device=venv.device) * 2 - 1
        with timer.phase("sim"):
            venv.step(act)
        it += 1
        if it % a["log_every_iters"] == 0:
            stats = venv.pop_episode_stats()
            finished = stats["episodes_total"]
            done = finished >= a["min_episodes"] and it * B >= a["min_env_steps"]
            if rep.emit(it * B, it, stats) or done:
                break
    return {"env_steps": it * B, "iterations": it, "episodes": finished, "num_envs": B, **rep.summary()}
