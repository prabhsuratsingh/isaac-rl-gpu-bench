"""MBPO (Janner et al. 2019), naive implementation: SAC trained mostly on short branched model rollouts.

Loop, per vectorised env step (B real envs):
    act (SAC, stochastic; uniform for seed_steps) -> env.step -> real buffer
    every train_every transitions:   retrain the ensemble on real data (same model/loss as PETS)
    every rollout_every transitions: branch rollout_batch model rollouts of length k from real states
                                     with the current policy -> model buffer
    every step: round(utd * B) SAC updates on batches of real_ratio real + (1 - real_ratio) model data

Choices that keep the comparison with PETS fair:
  * the same NaiveEnsemble, normaliser, training routine and known reward/termination functions
    (envs/costs.py). The model learns dynamics only.
  * all buffers are preallocated on the GPU; nothing in our loop syncs with the host (sync_debug).
  * TS-1 propagation: every rollout particle draws a fresh ensemble member each step. The naive version
    evaluates every member on every particle and gathers (E x the FLOPs, no boolean indexing).
Real time-outs (next_obs = post-reset obs) get weight 0 for both model and SAC. Real terminations stay
usable for SAC (the TD target masks next_obs).
"""

from __future__ import annotations

import math

import torch

from gpubench.algos.pets.buffer import GpuReplayBuffer
from gpubench.algos.pets.run import build_model, model_train_steps, sync_guard, train_model
from gpubench.common.reporter import Reporter
from gpubench.common.timing import ProfilerWindow
from gpubench.envs.costs import make_cost

from .sac import SAC


def rollout_length(sched, env_steps: int) -> int:
    """MBPO's linear schedule [k0, k1, step0, step1] (env steps), or a constant int."""
    if isinstance(sched, (int, float)):
        return int(sched)
    k0, k1, s0, s1 = sched
    f = min(1.0, max(0.0, (env_steps - s0) / max(1, s1 - s0)))
    return int(round(k0 + f * (k1 - k0)))


@torch.no_grad()
def branched_rollout(model, sac: SAC, cost_fn, real: GpuReplayBuffer, model_buf: GpuReplayBuffer, n: int, k: int):
    """n rollouts of length k from states drawn uniformly from the real buffer. Returns the transition count."""
    idx = real.sample_idx((n,))
    s = real.obs[idx]
    alive = torch.ones(n, device=s.device)
    ar = torch.arange(n, device=s.device)
    for _ in range(k):
        a = sac.act(s)
        E = model.E
        preds = model.predict(s.unsqueeze(0).expand(E, -1, -1), a.unsqueeze(0).expand(E, -1, -1))  # [E, n, do]
        s2 = preds[torch.randint(0, E, (n,), device=s.device), ar]  # TS-1: random member per particle
        r, term = cost_fn(s, a, s2)
        model_buf.add(s, a, s2, alive > 0, term, r)  # transitions after a predicted termination get weight 0
        alive = alive * (~term).to(alive.dtype)
        s = s2
    return n * k


def _sample(buf: GpuReplayBuffer, n: int, real: bool) -> dict:
    b = buf.get_batch(buf.sample_idx((n,)))
    # real data: time-outs (valid=0, term=0) are unusable; terminations are fine for TD
    b["w"] = (b["valid"] + b["term"]).clamp(max=1.0) if real else b["valid"]
    return b


def sac_update(sac: SAC, real: GpuReplayBuffer, model_buf: GpuReplayBuffer, batch: int, real_ratio: float) -> dict:
    n_real = max(1, int(round(batch * real_ratio))) if model_buf.size > 0 else batch
    parts = [_sample(real, n_real, True)]
    if model_buf.size > 0 and batch - n_real > 0:
        parts.append(_sample(model_buf, batch - n_real, False))
    b = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return sac.update(b["obs"], b["act"], b["rew"], b["next_obs"], b["term"], b["w"])


def run(cfg: dict, logger, venv, env_cfg, timer):
    a, device = cfg["algo"], venv.device
    cost_fn = make_cost(cfg["task"]["cost"], env_cfg)
    B, do, da = venv.num_envs, venv.obs_dim, venv.act_dim
    m, s = a["model"], a["sac"]

    model, model_opt = build_model(m, do, da, cfg["task"]["cost"], device, a.get("impl", "naive"))
    sac = SAC(do, da, hidden=s["hidden"], layers=s["layers"], actor_lr=s["actor_lr"], critic_lr=s["critic_lr"],
              alpha_lr=s["alpha_lr"], gamma=s["gamma"], tau=s["tau"], init_alpha=s["init_alpha"]).to(device)
    real = GpuReplayBuffer(a["buffer_capacity"], do, da, device)
    k_max = rollout_length(a["rollout_len"], 10**12)
    model_buf = GpuReplayBuffer(a["rollout_batch"] * max(1, k_max) * a["model_retain_rounds"], do, da, device)
    rep = Reporter.from_cfg(cfg, logger, timer, "mbpo", B)

    total_iters = math.ceil(a["total_env_steps"] / B)
    seed_iters = math.ceil(a["seed_steps"] / B)
    train_every = max(1, a["train_every"] // B)
    rollout_every = max(1, a["rollout_every"] // B)
    log_every = max(1, a["log_every"] // B)
    updates_per_iter = max(1, int(round(a["utd"] * B)))
    sync_mode = a.get("sync_debug", "off")
    prof = ProfilerWindow(cfg["profile"]["start"], cfg["profile"]["length"])
    if prof.start is not None:
        total_iters = min(total_iters, prof.stop + 1)

    obs = venv.reset()
    last_it, it = 0, -1
    trained = False
    n_rollouts = n_updates = 0
    stats: dict = {}
    sac_stats: dict = {}

    for it in range(total_iters):
        if prof.step(it):
            break
        steps_now = it * B
        if it >= seed_iters:
            # ---- model training ----------------------------------------------------------------
            if not trained or (it - seed_iters) % train_every == 0:
                with timer.phase("model_train"):
                    holdout = model.eval_mse(*real.get(real.recent_idx(train_every * B))) if trained else None
                    steps = model_train_steps(m, real.size, first=not trained)
                    loss, mse = train_model(model, model_opt, real, steps, m["batch_size"])
                    sac.set_obs_norm(model.in_mean[:do], model.in_std[:do])
                stats.update(model_loss=loss, model_train_mse=mse, model_holdout_mse=holdout)
                trained = True
            # ---- branched model rollouts --------------------------------------------------------
            if (it - seed_iters) % rollout_every == 0:
                k = rollout_length(a["rollout_len"], steps_now)
                with sync_guard(sync_mode), timer.phase("model_rollout"):
                    n_rollouts += branched_rollout(model, sac, cost_fn, real, model_buf, a["rollout_batch"], k)
                stats["rollout_len"] = k

        # ---- act ------------------------------------------------------------------------------
        with sync_guard(sync_mode), timer.phase("policy_act"):
            act = (torch.rand(B, da, device=device) * 2 - 1) if it < seed_iters else sac.act(obs)
        with timer.phase("sim"):
            next_obs, rew, term, trunc, _ = venv.step(act)
        with sync_guard(sync_mode), timer.phase("buffer"):
            done = term | trunc
            real.add(obs, act, next_obs, ~done, term, rew)
        obs = next_obs

        # ---- policy updates ---------------------------------------------------------------------
        if it >= seed_iters:
            with sync_guard(sync_mode), timer.phase("policy_update"):
                for _ in range(updates_per_iter):
                    sac_stats = sac_update(sac, real, model_buf, s["batch_size"], a["real_ratio"])
            n_updates += updates_per_iter

        # ---- log --------------------------------------------------------------------------------
        is_last = it == total_iters - 1
        if (it + 1) % log_every == 0 or (is_last and it + 1 - last_it >= max(1, log_every // 4)):
            stop = rep.emit((it + 1) * B, it + 1, venv.pop_episode_stats(),
                            model_rollouts=n_rollouts, model_rollout_phase="model_rollout",
                            policy_updates=n_updates, extras={**stats, **sac_stats, "model_buffer_size": model_buf.size})
            last_it, n_rollouts, n_updates = it + 1, 0, 0
            if stop:
                break

    return {"env_steps": (it + 1) * B, "iterations": it + 1, "num_envs": B, "updates_per_iter": updates_per_iter,
            **rep.summary()}
