"""CPU-runnable tests for everything that does not need Isaac Sim (run locally and on the instance)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpubench.algos.pets.buffer import GpuReplayBuffer  # noqa: E402
from gpubench.algos.pets.ensemble_naive import NaiveEnsemble  # noqa: E402
from gpubench.algos.pets.mppi_naive import NaiveMPPI  # noqa: E402
from gpubench.algos.pets.run import train_model  # noqa: E402
from gpubench.common.config import load_config  # noqa: E402
from gpubench.common.timing import PhaseTimer  # noqa: E402
from gpubench.envs.costs import make_cost  # noqa: E402
from gpubench.launcher.jobs import expand  # noqa: E402


# ---------------------------------------------------------------------------------------------- config
def test_config_merge_and_overrides():
    cfg = load_config(["algo=pets_naive", "task=cartpole", "seed=3", "algo.mppi.horizon=7", "run.exp=x"])
    assert cfg["seed"] == 3 and cfg["run"]["exp"] == "x"
    assert cfg["algo"]["mppi"]["horizon"] == 7  # CLI beats task override
    assert cfg["algo"]["num_envs"] == 32 and cfg["algo"]["total_env_steps"] == 40000  # task algo_overrides.pets
    assert cfg["algo"]["mppi"]["noise_std_final"] == 0.15  # nested task override merged, defaults kept
    assert cfg["algo"]["mppi"]["samples"] == 400
    cfg = load_config(["algo=ppo", "task=ant"])
    assert cfg["task"]["gym_id"] == "Isaac-Ant-Direct-v0" and "algo_overrides" not in cfg["task"]


@pytest.mark.parametrize("exp", sorted((ROOT / "configs" / "exp").glob("*.yaml")))
def test_exp_files_expand(exp):
    spec = expand(exp)
    assert spec["jobs"]
    for j in spec["jobs"]:
        assert j["cmd"].startswith(("python ", "bash "))
        if "scripts/train.py" in j["cmd"] or "profile.sh" in j["cmd"]:
            import shlex

            args = [a for a in shlex.split(j["cmd"]) if "=" in a]
            if "tuned=true" in args:  # tuned configs appear only after `tune.py report`
                kv = dict(a.split("=", 1) for a in args)
                if not (ROOT / "configs" / "tuned" / f"{kv['algo']}_{kv['task']}.yaml").exists():
                    args.remove("tuned=true")
            load_config(args)  # every expanded job resolves to a valid config


def test_step1_ppo_matrix():
    spec = expand(ROOT / "configs" / "exp" / "step1_ppo.yaml")
    assert len(spec["jobs"]) == 15


# ---------------------------------------------------------------------------------------------- costs
def _isaac_cartpole_reward(cfg, pole_pos, pole_vel, cart_pos, cart_vel, reset_terminated):
    """Verbatim logic of isaaclab_tasks/direct/cartpole/cartpole_env.py::compute_rewards."""
    rew_alive = cfg.rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = cfg.rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = cfg.rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = cfg.rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = cfg.rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    return rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel


CARTPOLE_CFG = SimpleNamespace(rew_scale_alive=1.0, rew_scale_terminated=-2.0, rew_scale_pole_pos=-1.0,
                               rew_scale_cart_vel=-0.01, rew_scale_pole_vel=-0.005, max_cart_pos=3.0)


def test_cartpole_cost_matches_env_formula():
    torch.manual_seed(0)
    nxt = torch.randn(4096, 4) * torch.tensor([1.2, 2.0, 2.5, 2.0])
    fn = make_cost("cartpole", CARTPOLE_CFG)
    r, term = fn(None, torch.zeros(4096, 1), nxt)
    ref_term = (nxt[:, 2].abs() > 3.0) | (nxt[:, 0].abs() > math.pi / 2)
    ref = _isaac_cartpole_reward(CARTPOLE_CFG, nxt[:, 0], nxt[:, 1], nxt[:, 2], nxt[:, 3], ref_term)
    assert torch.equal(term, ref_term)
    assert torch.allclose(r, ref, atol=1e-6)
    r2, _ = fn(None, torch.zeros(3, 7, 1), nxt[:21].view(3, 7, 4))  # arbitrary leading dims
    assert r2.shape == (3, 7)


def test_ant_cost_shapes_and_terms():
    cfg = SimpleNamespace(heading_weight=0.5, up_weight=0.1, actions_cost_scale=0.005, energy_cost_scale=0.05,
                          alive_reward_scale=0.5, death_cost=-2.0, termination_height=0.31, decimation=2)
    fn = make_cost("ant", cfg)
    obs = torch.zeros(5, 36)
    obs[:, 0] = 0.55
    obs[:, 10] = 1.0
    obs[:, 11] = 1.0
    moving = obs.clone()
    moving[:, 1] = 1.0  # 1 m/s forward, target straight ahead (angle_to_target=0)
    r_still, t = fn(obs, torch.zeros(5, 8), obs)
    r_move, _ = fn(obs, torch.zeros(5, 8), moving)
    assert not t.any()
    assert torch.allclose(r_still, torch.full((5,), 0.5 + 0.1 + 0.5))
    assert torch.allclose(r_move - r_still, torch.full((5,), 2.0))  # decimation * v
    fallen = obs.clone()
    fallen[:, 0] = 0.2
    r_f, t_f = fn(obs, torch.zeros(5, 8), fallen)
    assert t_f.all() and torch.allclose(r_f, torch.full((5,), -2.0))


# ---------------------------------------------------------------------------------------------- buffer
def test_buffer_ring_and_recent():
    b = GpuReplayBuffer(10, 2, 1, "cpu")
    for i in range(4):
        x = torch.full((3, 2), float(i))
        b.add(x, torch.zeros(3, 1), x + 1, torch.ones(3, dtype=torch.bool))
    assert b.size == 10 and b.ptr == 2
    recent = b.obs[b.recent_idx(3)][:, 0]
    assert recent.tolist() == [3.0, 3.0, 3.0]


# ---------------------------------------------------------------------------------------------- MPPI
def test_mppi_member_layout_roundtrip():
    model = NaiveEnsemble(3, 1, ensemble_size=5, hidden=8, layers=1)
    mp = NaiveMPPI(model, None, num_envs=4, act_dim=1, device="cpu", samples=10, particles=2)
    x = torch.arange(4 * 20).float().view(4, 20, 1)
    m = mp._to_members(x)
    assert m.shape == (5, 16, 1)
    assert torch.equal(mp._from_members(m), x)
    # particle j of each env -> member j % E
    assert torch.equal(m[0, :4, 0], torch.tensor([0.0, 5.0, 10.0, 15.0]))


class _PerfectModel(torch.nn.Module):
    """Ensemble interface wrapping known 1-D dynamics s' = s + a (for planner tests)."""

    E = 5

    def predict(self, obs, act, sample=True):
        return obs + act


def test_mppi_drives_point_to_goal():
    torch.manual_seed(0)
    cost = lambda s, a, s2: (-(s2 - 3.0).square().sum(-1), torch.zeros(s2.shape[:-1], dtype=torch.bool))  # noqa: E731
    mp = NaiveMPPI(_PerfectModel(), cost, num_envs=8, act_dim=1, device="cpu", horizon=8, samples=200, iterations=3,
                   noise_std=0.5)
    s = torch.zeros(8, 1)
    for _ in range(12):
        s = s + mp.plan(s)
    assert (s - 3.0).abs().max() < 0.3


def test_ensemble_learns_and_pets_beats_random():
    """End-to-end naive PETS on a toy 2-D double integrator: train ensemble from random data, plan."""
    torch.manual_seed(0)
    dt = 0.1

    def true_step(s, a):
        pos, vel = s[..., :1], s[..., 1:]
        vel = vel + dt * a
        return torch.cat([pos + dt * vel, vel], -1)

    cost = lambda s, a, s2: (-(s2[..., 0] - 1.0).square() - 0.01 * a.square().sum(-1), torch.zeros(s2.shape[:-1], dtype=torch.bool))  # noqa: E731
    buf = GpuReplayBuffer(20000, 2, 1, "cpu")
    s = torch.randn(64, 2) * 0.5
    for _ in range(100):
        a = torch.rand(64, 1) * 2 - 1
        s2 = true_step(s, a)
        buf.add(s, a, s2, torch.ones(64, dtype=torch.bool))
        s = s2 if s2.abs().max() < 3 else torch.randn(64, 2) * 0.5
    model = NaiveEnsemble(2, 1, ensemble_size=5, hidden=64, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    _, mse, _ = train_model(model, opt, buf, steps=600, batch=256)
    assert mse.item() < 0.05, mse.item()

    def rollout(policy, n=40):
        s = torch.zeros(16, 2)
        total = torch.zeros(16)
        for _ in range(n):
            a = policy(s)
            s = true_step(s, a)
            total += cost(None, a, s)[0]
        return total.mean().item()

    mp = NaiveMPPI(model, cost, num_envs=16, act_dim=1, device="cpu", horizon=15, samples=200, iterations=2)
    r_pets = rollout(mp.plan)
    r_rand = rollout(lambda s: torch.rand(16, 1) * 2 - 1)
    assert r_pets > r_rand + 5.0, (r_pets, r_rand)


# ---------------------------------------------------------------------------------------------- env stats
class _FakeEnv:
    """Gym-like vectorised env: env i terminates every (i+2) steps with reward 1 per step."""

    def __init__(self, n):
        import gymnasium  # noqa: F401  (spaces only)
        from gymnasium import spaces

        self.unwrapped = self
        self.num_envs, self.device = n, "cpu"
        self.single_observation_space = spaces.Dict({"policy": spaces.Box(-1, 1, (3,))})
        self.single_action_space = spaces.Box(-1, 1, (1,))
        self.t = torch.zeros(n)
        self.period = torch.arange(n) + 2.0

    def reset(self):
        return {"policy": torch.zeros(self.num_envs, 3)}, {}

    def step(self, a):
        self.t += 1
        term = self.t >= self.period
        self.t[term] = 0
        return {"policy": torch.zeros(self.num_envs, 3)}, torch.ones(self.num_envs), term, torch.zeros_like(term), {}


def test_gpuvecenv_episode_window():
    pytest.importorskip("gymnasium")
    from gpubench.envs.isaac import GpuVecEnv

    assert GpuVecEnv(_FakeEnv(4)).window == 4  # default min(100, num_envs)
    env = GpuVecEnv(_FakeEnv(4), window=5)
    env.reset()
    finished = []  # reference: python bookkeeping of (return) per finished episode, in order
    t = [0] * 4
    for _ in range(12):
        env.step(torch.zeros(4, 1))
        for i in range(4):
            t[i] += 1
            if t[i] >= i + 2:
                finished.append(float(t[i]))
                t[i] = 0
    stats = env.pop_episode_stats()
    assert stats["episodes_total"] == len(finished)
    assert stats["reward_mean"] == pytest.approx(sum(finished[-5:]) / 5)
    # ring holds exactly the last `window` returns (as a multiset)
    assert sorted(env._ep_ret.tolist()) == sorted(finished[-5:])
    assert stats["episodes"] == len(finished) and env.pop_episode_stats()["episodes"] == 0


# ---------------------------------------------------------------------------------------------- timing
def test_phase_timer_cpu():
    t = PhaseTimer("cpu")
    for _ in range(3):
        with t.phase("a"):
            pass
    f = t.wrap("b", lambda x: x + 1)
    assert f(1) == 2
    out = t.flush()
    assert out["calls"] == {"a": 3, "b": 1} and out["gpu_ms"] == {}
    assert t.flush()["calls"] == {}
