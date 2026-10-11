"""PETS training loop (naive implementation): collect with MPPI -> retrain ensemble -> repeat.

Real-env interaction runs B Isaac Lab envs in parallel, each planned for by the batched MPPI.
Budgets are in *total env transitions* (summed over the B envs), the same unit PPO is measured in.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

from gpubench.common.reporter import Reporter
from gpubench.common.timing import ProfilerWindow
from gpubench.envs.costs import ANT_ACTION_SLICE, make_cost

from .buffer import GpuReplayBuffer
from .ensemble_batched import BatchedEnsemble
from .ensemble_naive import NaiveEnsemble
from .mppi_naive import NaiveMPPI

ACTION_OBS_SLICES = {"ant": ANT_ACTION_SLICE}


@contextmanager
def sync_guard(mode: str | int):
    """torch.cuda.set_sync_debug_mode around *our* code: 'warn' / 'error' on any host<->device sync."""
    if not mode or mode == "off" or not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(mode)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


# step-3 ablation levels (cumulative): which ensemble each level uses; the planner is chosen in build_planner
IMPL_LEVELS = ("naive", "batched", "fused", "graph", "bf16")
ENSEMBLES = {"naive": NaiveEnsemble, "batched": BatchedEnsemble, "fused": BatchedEnsemble, "graph": BatchedEnsemble,
             "bf16": BatchedEnsemble}


def build_model(m: dict, obs_dim: int, act_dim: int, cost_kind: str, device, impl: str = "naive"):
    if impl not in ENSEMBLES:
        raise NotImplementedError(f"ensemble impl '{impl}' (available: {sorted(ENSEMBLES)})")
    model = ENSEMBLES[impl](obs_dim, act_dim, m["ensemble_size"], m["hidden"], m["layers"],
                            ACTION_OBS_SLICES.get(cost_kind)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=m["lr"], weight_decay=m.get("weight_decay", 0.0))
    return model, opt


def build_planner(impl: str, model, cost_fn, num_envs: int, act_dim: int, device, mp: dict):
    """naive/batched -> NaiveMPPI (torch ops); fused -> FusedMPPI (Triton step kernels);
    graph -> fused captured as CUDA graphs; bf16 -> graph with bf16 planning GEMMs."""
    kw = dict(horizon=mp["horizon"], samples=mp["samples"], particles=mp["particles"], iterations=mp["iterations"],
              temperature=mp["temperature"], noise_std=mp["noise_std"], action_repeat=mp.get("action_repeat", 1),
              noise_decay=mp.get("noise_decay", 1.0), elite_frac=mp.get("elite_frac"), min_std=mp.get("min_std", 0.05))
    if impl in ("naive", "batched"):
        return NaiveMPPI(model, cost_fn, num_envs, act_dim, device, **kw)
    from .mppi_fused import FusedMPPI  # Triton: imported only when used (not available on CPU-only hosts)

    dtype = torch.bfloat16 if impl == "bf16" else torch.float32
    planner = FusedMPPI(model, cost_fn, num_envs, act_dim, device, dtype=dtype, **kw)
    if impl == "fused":
        return planner
    from .graphs import GraphedPlanner

    return GraphedPlanner(planner)


def train_model(model, opt, buf: GpuReplayBuffer, steps: int, batch: int, early_stop: dict | None = None):
    """`steps` gradient steps on bootstrap batches (one independent batch per member).

    early_stop = {holdout_frac, holdout_max, check_every, patience}: hold out a random split of the
    buffer this round, check its MSE every `check_every` steps and stop after `patience` checks without
    improvement. The checks read a scalar back (a host sync), but only inside the model_train phase,
    never in the act loop. Returns (loss, mse, steps_done).
    """
    model.fit_normalizer(*buf.all()[:4])
    model.train()
    loss = mse = None
    n = buf.size
    pool = hold = None
    if early_stop:
        n_hold = min(int(early_stop.get("holdout_max", 20000)), max(1, int(n * early_stop.get("holdout_frac", 0.1))))
        perm = torch.randperm(n, device=buf.device)
        hold, pool = perm[:n_hold], perm[n_hold:]
    best, bad, done = float("inf"), 0, 0
    for step in range(steps):
        if pool is not None:
            idx = pool[torch.randint(0, pool.numel(), (model.E, batch), device=buf.device)]
        else:
            idx = buf.sample_idx((model.E, batch))  # independent bootstrap batch per member
        o, a, no, v = buf.get(idx)
        loss, mse = model.loss(o, a, no, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        done = step + 1
        if early_stop and done % int(early_stop.get("check_every", 100)) == 0:
            h = model.eval_mse(*buf.get(hold)).item()
            if h < best - 1e-6:
                best, bad = h, 0
            else:
                bad += 1
                if bad >= int(early_stop.get("patience", 5)):
                    break
    model.eval()
    return loss.detach(), mse.detach(), done


def model_train_steps(m: dict, buf_size: int, first: bool) -> int:
    """Fixed gradient-step budget per round, or `train_epochs_per_round` passes over the buffer
    (capped by `max_train_steps` when set)."""
    if first:
        steps = int(m["first_train_steps"])
    elif m.get("train_epochs_per_round"):
        steps = max(1, int(m["train_epochs_per_round"] * buf_size / m["batch_size"]))
    else:
        steps = int(m["train_steps"])
    if m.get("max_train_steps"):
        steps = min(steps, int(m["max_train_steps"]))
    return steps


def early_stop_cfg(m: dict) -> dict | None:
    if not m.get("early_stop"):
        return None
    return {k: m[k] for k in ("holdout_frac", "holdout_max", "check_every", "patience") if m.get(k) is not None}


def annealed(v0: float, v1: float | None, steps: int, anneal_steps: int | None) -> float:
    if v1 is None or not anneal_steps:
        return v0
    f = min(1.0, steps / anneal_steps)
    return v0 + f * (v1 - v0)


def run(cfg: dict, logger, venv, env_cfg, timer):
    a, device = cfg["algo"], venv.device
    cost_kind = cfg["task"]["cost"]
    cost_fn = make_cost(cost_kind, env_cfg)
    B, do, da = venv.num_envs, venv.obs_dim, venv.act_dim

    model, opt = build_model(a["model"], do, da, cost_kind, device, a.get("impl", "naive"))
    mp = a["mppi"]
    planner = build_planner(a.get("impl", "naive"), model, cost_fn, B, da, device, mp)
    buf = GpuReplayBuffer(a["buffer_capacity"], do, da, device)
    rep = Reporter.from_cfg(cfg, logger, timer, "pets", B)

    total_iters = math.ceil(a["total_env_steps"] / B)
    seed_iters = math.ceil(a["seed_steps"] / B)
    train_every = max(1, a["train_every"] // B)
    log_every = max(1, a["log_every"] // B)
    sync_mode = a.get("sync_debug", "off")
    prof = ProfilerWindow(cfg["profile"]["start"], cfg["profile"]["length"])
    if prof.start is not None:
        total_iters = min(total_iters, prof.stop + 1)

    obs = venv.reset()
    last_it = 0
    model_stats: dict = {}
    trained = False
    plan_calls = 0
    it = -1

    for it in range(total_iters):
        if prof.step(it):
            break
        # ---- model training ------------------------------------------------------------------
        if it >= seed_iters and (not trained or (it - seed_iters) % train_every == 0):
            with timer.phase("model_train"):
                holdout = None
                if trained:  # out-of-sample: transitions collected since the last training round
                    ridx = buf.recent_idx(train_every * B)
                    holdout = model.eval_mse(*buf.get(ridx))
                steps = model_train_steps(a["model"], buf.size, first=not trained)
                loss, mse, done = train_model(model, opt, buf, steps, a["model"]["batch_size"], early_stop_cfg(a["model"]))
            model_stats = {"model_loss": loss, "model_train_mse": mse, "model_holdout_mse": holdout, "model_train_steps": done}
            trained = True

        # ---- act ------------------------------------------------------------------------------
        planner.sigma = annealed(mp["noise_std"], mp.get("noise_std_final"), it * B, mp.get("noise_anneal_steps"))
        with sync_guard(sync_mode):
            if not trained:
                with timer.phase("random_act"):
                    act = torch.rand(B, da, device=device) * 2 - 1
            else:
                with timer.phase("plan"):
                    act = planner.plan(obs)
                plan_calls += 1

        with timer.phase("sim"):
            next_obs, rew, term, trunc, _ = venv.step(act)

        with sync_guard(sync_mode):
            with timer.phase("buffer"):
                done = term | trunc
                buf.add(obs, act, next_obs, ~done, term)
                planner.reset(done)
        obs = next_obs

        # ---- log (every `log_every`; the final partial interval only if sizeable) ----------------
        is_last = it == total_iters - 1
        if (it + 1) % log_every == 0 or (is_last and it + 1 - last_it >= max(1, log_every // 4)):
            stop = rep.emit((it + 1) * B, it + 1, venv.pop_episode_stats(),
                            model_rollouts=plan_calls * planner.rollouts_per_plan, model_rollout_phase="plan",
                            planner_steps=plan_calls * B,
                            extras={**model_stats, "noise_std": planner.sigma})
            last_it, plan_calls = it + 1, 0
            if stop:
                break

    return {"env_steps": (it + 1) * B, "iterations": it + 1, "rollouts_per_plan": planner.rollouts_per_plan,
            "num_envs": B, **rep.summary()}
