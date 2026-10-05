"""PPO reference: rsl_rl OnPolicyRunner + Isaac Lab's shipped agent config for the task.

We do not modify the algorithm. The runner is subclassed only to
  * time the phases (sim = env.step, policy_act = actor forward/sampling, policy_update = PPO update)
    with CUDA events + NVTX, by wrapping the bound methods (no change to the training loop),
  * write our JSONL row from rsl_rl's own log() hook,
  * open/close the Nsight capture window.
"""

from __future__ import annotations

import statistics

from gpubench.common.timing import ProfilerWindow, gpu_memory_stats


def _apply_overrides(obj, overrides: dict):
    for key, val in (overrides or {}).items():
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
        def attach(self, logger, timer, prof: ProfilerWindow, num_envs: int):
            self.bench_logger, self.bench_timer, self.prof = logger, timer, prof
            self.bench_num_envs = num_envs
            self._last_wall, self._last_steps = logger.wall_s, 0
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
            t = self.bench_timer.flush()
            wall = self.bench_logger.wall_s
            d_steps = self.tot_timesteps - self._last_steps
            rew, lens = locs["rewbuffer"], locs["lenbuffer"]
            row = {
                "iteration": it + 1,
                "env_steps": self.tot_timesteps,
                "env_steps_per_s": d_steps / max(wall - self._last_wall, 1e-9),
                "reward_mean": statistics.mean(rew) if len(rew) else None,
                "ep_len_mean": statistics.mean(lens) if len(lens) else None,
                "collection_s": locs["collection_time"],
                "learn_s": locs["learn_time"],
                **{f"loss_{k}": v for k, v in (locs.get("loss_dict") or {}).items()},
                **t,
                **gpu_memory_stats(),
            }
            self.bench_logger.log(row)
            self._last_wall, self._last_steps = wall, self.tot_timesteps
            self.prof.step(it + 1)

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
    prof = ProfilerWindow(cfg["profile"]["start"], cfg["profile"]["length"])
    if prof.start is not None:  # profile runs only need the window
        agent_cfg.max_iterations = prof.stop + 1

    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = make_runner_cls()(venv, agent_cfg.to_dict(), log_dir=str(logger.dir / "rsl_rl"), device=agent_cfg.device)
    runner.attach(logger, timer, prof, venv.num_envs)
    logger._write_json("agent_cfg.json", agent_cfg.to_dict())

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    return {
        "env_steps": runner.tot_timesteps,
        "iterations": agent_cfg.max_iterations,
        "num_envs": venv.num_envs,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
    }
