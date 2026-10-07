"""PPO reference: rsl_rl OnPolicyRunner + Isaac Lab's shipped agent config for the task.

We do not modify the algorithm. The runner is subclassed only to
  * time the phases (sim = env.step, policy_act = actor forward/sampling, policy_update = PPO update)
    with CUDA events + NVTX, by wrapping the bound methods (no change to the training loop),
  * write our JSONL row (shared Reporter) from rsl_rl's own log() hook,
  * open/close the Nsight capture window and stop at the wall-clock cap (`algo.max_wall_s`).
"""

from __future__ import annotations

import statistics

from gpubench.common.reporter import Reporter
from gpubench.common.timing import ProfilerWindow


class _StopTraining(Exception):
    """Raised from the log hook to leave rsl_rl's learn() loop early (wall-clock cap / profile window)."""


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _apply_overrides(obj, overrides: dict):
    """Set dotted (`algorithm.learning_rate`) or nested ({algorithm: {learning_rate: ..}}) agent-cfg fields."""
    for key, val in _flatten(overrides).items():
        target = obj
        *path, last = key.split(".")
        for p in path:
            target = getattr(target, p)
        if not hasattr(target, last):
            raise AttributeError(f"agent cfg has no field {key!r}")
        setattr(target, last, val)


def make_runner_cls():
    from rsl_rl.runners import OnPolicyRunner

    class BenchRunner(OnPolicyRunner):
        def attach(self, reporter: Reporter, timer, prof: ProfilerWindow):
            self.reporter, self.bench_timer, self.prof = reporter, timer, prof
            self.env.step = timer.wrap("sim", self.env.step)
            for attr, phase in [("act", "policy_act"), ("process_env_step", "storage"),
                                ("compute_returns", "policy_update"), ("update", "policy_update")]:
                if hasattr(self.alg, attr):  # tolerate minor rsl_rl API drift
                    setattr(self.alg, attr, timer.wrap(phase, getattr(self.alg, attr)))
                else:
                    print(f"[ppo] warning: rsl_rl algorithm has no '{attr}'; phase '{phase}' will be incomplete")

        def log(self, locs: dict, width: int = 80, pad: int = 35):
            super().log(locs, width, pad)  # rsl_rl console + tensorboard; updates tot_timesteps
            it = locs["it"]
            rew, lens = locs["rewbuffer"], locs["lenbuffer"]
            stats = {"reward_mean": statistics.mean(rew) if len(rew) else None,
                     "ep_len_mean": statistics.mean(lens) if len(lens) else None,
                     "episodes": None, "episodes_total": None, "reward_window": getattr(rew, "maxlen", None)}
            extras = {"collection_s": locs["collection_time"], "learn_s": locs["learn_time"],
                      **{f"loss_{k}": v for k, v in (locs.get("loss_dict") or {}).items()}}
            # gradient steps this iteration (epochs x minibatches), so policy_updates_per_s compares across algos
            n_updates = getattr(self.alg, "num_learning_epochs", 0) * getattr(self.alg, "num_mini_batches", 0)
            stop = self.reporter.emit(self.tot_timesteps, it + 1, stats, policy_updates=n_updates, extras=extras)
            if self.prof.step(it + 1) or stop:
                raise _StopTraining

    return BenchRunner


def run(cfg: dict, logger, env, env_cfg, timer):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    a = cfg["algo"]
    agent_cfg = load_cfg_from_registry(cfg["task"]["gym_id"], a.get("agent_entry_point", "rsl_rl_cfg_entry_point"))
    agent_cfg.seed = cfg["seed"]
    agent_cfg.device = cfg["device"]
    _apply_overrides(agent_cfg, a.get("agent_overrides"))
    if a.get("max_iterations"):
        agent_cfg.max_iterations = int(a["max_iterations"])
    if not a.get("save_checkpoints", False):
        # rsl_rl saves model_<it>.pt every save_interval iterations (1.6 GB over the step-2 Ant tuning);
        # the benchmark never reloads them. The final model is still saved when learn() completes.
        agent_cfg.save_interval = 10**9
    prof = ProfilerWindow(cfg["profile"]["start"], cfg["profile"]["length"])
    if prof.start is not None:  # profile runs only need the window
        agent_cfg.max_iterations = prof.stop + 1

    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    reporter = Reporter.from_cfg(cfg, logger, timer, "ppo", venv.num_envs, verbose=False)
    runner = make_runner_cls()(venv, agent_cfg.to_dict(), log_dir=str(logger.dir / "rsl_rl"), device=agent_cfg.device)
    runner.attach(reporter, timer, prof)
    logger._write_json("agent_cfg.json", agent_cfg.to_dict())

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except _StopTraining:
        pass
    return {
        "env_steps": runner.tot_timesteps,
        "iterations": runner.current_learning_iteration,
        "num_envs": venv.num_envs,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        **reporter.summary(),
    }
