"""PETS training loop (naive implementation): collect with MPPI -> retrain ensemble -> repeat.

Real-env interaction runs B Isaac Lab envs in parallel, each planned for by the batched MPPI.
Budgets are in *total env transitions* (summed over the B envs), the same unit PPO is measured in.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

from gpubench.common.timing import ProfilerWindow, gpu_memory_stats
from gpubench.envs.costs import ANT_ACTION_SLICE, make_cost

from .buffer import GpuReplayBuffer
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


def build_model(a: dict, obs_dim: int, act_dim: int, cost_kind: str, device):
    m = a["model"]
    impl = a.get("impl", "naive")
    if impl != "naive":
        raise NotImplementedError(f"PETS impl '{impl}' lands in the systems-ablation step")
    model = NaiveEnsemble(obs_dim, act_dim, m["ensemble_size"], m["hidden"], m["layers"], ACTION_OBS_SLICES.get(cost_kind)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=m["lr"], weight_decay=m.get("weight_decay", 0.0))
    return model, opt


def train_model(model, opt, buf: GpuReplayBuffer, steps: int, batch: int):
    model.fit_normalizer(*buf.all())
    model.train()
    loss = mse = None
    for _ in range(steps):
        idx = buf.sample_idx((model.E, batch))  # independent bootstrap batch per member
        o, a, no, v = buf.get(idx)
        loss, mse = model.loss(o, a, no, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return loss.detach(), mse.detach()


def run(cfg: dict, logger, venv, env_cfg, timer):
    a, device = cfg["algo"], venv.device
    cost_kind = cfg["task"]["cost"]
    cost_fn = make_cost(cost_kind, env_cfg)
    B, do, da = venv.num_envs, venv.obs_dim, venv.act_dim

    model, opt = build_model(a, do, da, cost_kind, device)
    mp = a["mppi"]
    planner = NaiveMPPI(model, cost_fn, B, da, device, horizon=mp["horizon"], samples=mp["samples"],
                        particles=mp["particles"], iterations=mp["iterations"], temperature=mp["temperature"],
                        noise_std=mp["noise_std"])
    buf = GpuReplayBuffer(a["buffer_capacity"], do, da, device)

    total_iters = math.ceil(a["total_env_steps"] / B)
    seed_iters = math.ceil(a["seed_steps"] / B)
    train_every = max(1, a["train_every"] // B)
    log_every = max(1, a["log_every"] // B)
    sync_mode = a.get("sync_debug", "off")
    prof = ProfilerWindow(cfg["profile"]["start"], cfg["profile"]["length"])
    if prof.start is not None:
        total_iters = min(total_iters, prof.stop + 1)

    obs = venv.reset()
    last = {"wall": logger.wall_s, "it": 0}
    model_stats: dict = {}
    trained = False
    plan_calls = 0

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
                steps = a["model"]["train_steps"] if trained else a["model"]["first_train_steps"]
                loss, mse = train_model(model, opt, buf, steps, a["model"]["batch_size"])
            model_stats = {"model_loss": loss, "model_train_mse": mse, "model_holdout_mse": holdout}
            trained = True

        # ---- act ------------------------------------------------------------------------------
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
                buf.add(obs, act, next_obs, ~done)
                planner.reset(done)
        obs = next_obs

        # ---- log --------------------------------------------------------------------------------
        # log every `log_every`; the final partial interval only if it is a sizeable one
        is_last = it == total_iters - 1
        if (it + 1) % log_every == 0 or (is_last and it + 1 - last["it"] >= max(1, log_every // 4)):
            t = timer.flush()
            wall = logger.wall_s
            d_wall = max(wall - last["wall"], 1e-9)
            d_steps = (it + 1 - last["it"]) * B
            plan_s = t["host_ms"].get("plan", 0.0) / 1e3
            row = {
                "iteration": it + 1,
                "env_steps": (it + 1) * B,
                "env_steps_per_s": d_steps / d_wall,
                "model_rollouts_per_s": plan_calls * planner.rollouts_per_plan / plan_s if plan_s > 0 else None,
                "planner_steps_per_s": plan_calls * B / plan_s if plan_s > 0 else None,
                **venv.pop_episode_stats(),
                **{k: (v.item() if torch.is_tensor(v) else v) for k, v in model_stats.items()},
                **t,
                **gpu_memory_stats(),
            }
            logger.log(row)
            print(f"[pets] it {it + 1}/{total_iters} steps {row['env_steps']} R {row.get('reward_mean', float('nan')):.2f} "
                  f"sps {row['env_steps_per_s']:.0f} holdout {row.get('model_holdout_mse')}", flush=True)
            last = {"wall": wall, "it": it + 1}
            plan_calls = 0

    return {"env_steps": (it + 1) * B, "iterations": it + 1,
            "rollouts_per_plan": planner.rollouts_per_plan, "num_envs": B}
