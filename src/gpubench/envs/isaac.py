"""Isaac Lab plumbing shared by every algorithm.

Import order matters: Isaac Sim must be launched (`launch_app`) before anything imports `isaaclab.*`
beyond `isaaclab.app`. `scripts/train.py` does that first thing.
"""

from __future__ import annotations

import torch


def launch_app(device: str = "cuda:0", headless: bool = True, **kwargs):
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=headless, device=device, **kwargs)
    return launcher.app


def shutdown(app, code: int = 0, env=None):
    """End the process with a meaningful exit code.

    `SimulationApp.close()` terminates the interpreter itself with status 0, so nothing after it runs and
    a failed run would look successful. Flush output first; only use the regular Kit shutdown on success.
    """
    import os
    import sys

    if env is not None:
        try:
            env.close()
        except Exception as e:  # noqa: BLE001
            print(f"[shutdown] env.close failed: {e}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if code == 0 and app is not None:
        app.close()  # does not return
    os._exit(code)


def make_env_cfg(task: dict, device: str, seed: int, num_envs: int | None = None):
    """Registered Isaac Lab env cfg + our overrides (num_envs, sim-cost knob)."""
    import isaaclab_tasks  # noqa: F401  (registers the gym ids)
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task["gym_id"], device=device, num_envs=num_envs or task.get("num_envs"))
    env_cfg.seed = seed
    # Sim-cost knob for the crossover analysis: k x more physics substeps per control step at the same
    # control frequency (dt / k, decimation * k). Dynamics change slightly (finer integration); the MDP
    # seen by the agent (control dt, episode length, obs/act) is unchanged.
    k = int(task.get("sim_substep_mult", 1))
    if k != 1:
        env_cfg.sim.dt = env_cfg.sim.dt / k
        env_cfg.decimation = env_cfg.decimation * k
        env_cfg.sim.render_interval = env_cfg.decimation
    for key, val in (task.get("env_cfg_overrides") or {}).items():
        obj = env_cfg
        *path, last = key.split(".")
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, last, val)
    return env_cfg


def make_env(task: dict, device: str, seed: int, num_envs: int | None = None):
    import gymnasium as gym

    env_cfg = make_env_cfg(task, device, seed, num_envs)
    env = gym.make(task["gym_id"], cfg=env_cfg)
    return env, env_cfg


class GpuVecEnv:
    """Minimal torch-in / torch-out view of a Direct Isaac Lab env for model-based agents.

    * obs, actions, rewards, dones stay CUDA tensors (no .cpu(), no .item()).
    * Isaac Lab auto-resets done envs inside step(), so `next_obs` for a done env is the *post-reset*
      observation. `valid` marks transitions whose next_obs is a true successor state (not done).
      Model-based learners train only on valid transitions (protocol: 'auto-reset masking').
    * Episode returns are accumulated on-device; `pop_episode_stats()` is the only sync point.
    * `reward_mean` is over the last `window` finished episodes, the same definition rsl_rl uses for PPO
      (its 100-episode rewbuffer), so both methods' curves and time-to-threshold are comparable.
    """

    def __init__(self, env, clip_actions: float | None = 1.0, window: int = 100):
        self.env = env
        self.u = env.unwrapped
        self.num_envs = self.u.num_envs
        self.device = self.u.device
        self.obs_dim = int(self.u.single_observation_space["policy"].shape[-1])
        self.act_dim = int(self.u.single_action_space.shape[-1])
        self.clip = clip_actions
        z = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self._ret, self._len = z(), z()
        self._done_ret_sum = torch.zeros((), device=self.device)
        self._done_len_sum = torch.zeros((), device=self.device)
        self._done_count = torch.zeros((), device=self.device)
        self._obs = None
        # ring of the last `window` finished episodes (return, length); filled without host syncs
        self.window = window
        self._ep_ret = torch.zeros(window, device=self.device)
        self._ep_len = torch.zeros(window, device=self.device)
        self._ep_total = torch.zeros((), dtype=torch.long, device=self.device)

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        self._obs = obs["policy"]
        return self._obs

    def step(self, actions: torch.Tensor):
        if self.clip is not None:
            actions = actions.clamp(-self.clip, self.clip)
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        next_obs = obs["policy"]
        done = terminated | truncated
        self._ret += rew
        self._len += 1
        d = done.to(rew.dtype)
        self._done_ret_sum += (self._ret * d).sum()
        self._done_len_sum += (self._len * d).sum()
        self._done_count += d.sum()
        self._record_finished(done)
        self._ret *= 1.0 - d
        self._len *= 1.0 - d
        self._obs = next_obs
        return next_obs, rew, terminated, truncated, extras

    def _record_finished(self, done: torch.Tensor):
        """Append returns/lengths of envs that just finished into the ring, without boolean indexing.

        Finished envs get consecutive slots via a cumsum over the done mask; unfinished envs are routed
        to slot `window` of a padded scratch row and dropped.
        """
        d = done.long()
        slot = (self._ep_total + torch.cumsum(d, 0) - 1) % self.window
        slot = torch.where(done, slot, torch.full_like(slot, self.window))
        ret = torch.cat([self._ep_ret, self._ep_ret.new_zeros(1)])
        ln = torch.cat([self._ep_len, self._ep_len.new_zeros(1)])
        ret.scatter_(0, slot, self._ret)
        ln.scatter_(0, slot, self._len)
        self._ep_ret.copy_(ret[: self.window])
        self._ep_len.copy_(ln[: self.window])
        self._ep_total += d.sum()

    def pop_episode_stats(self) -> dict:
        n = self._done_count.item()
        total = int(self._ep_total.item())
        out = {"episodes": int(n), "episodes_total": total}
        if n > 0:
            out["reward_mean_interval"] = self._done_ret_sum.item() / n
        if total > 0:
            k = min(total, self.window)
            # slots [0, k) are filled once total >= k; before the ring wraps that is exactly the first k
            out["reward_mean"] = self._ep_ret[:k].mean().item()
            out["ep_len_mean"] = self._ep_len[:k].mean().item()
        self._done_ret_sum.zero_()
        self._done_len_sum.zero_()
        self._done_count.zero_()
        return out

    def close(self):
        self.env.close()
