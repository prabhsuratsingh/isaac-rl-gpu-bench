"""Step-2 tests (CPU): shared reporter schema, PETS planner knobs, buffer fields, SAC, MBPO rollouts, and
end-to-end runs of PETS / MBPO / random on a fake Cartpole-like vectorised env."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpubench.algos.mbpo.run import branched_rollout, rollout_length, sac_update  # noqa: E402
from gpubench.algos.mbpo.sac import SAC  # noqa: E402
from gpubench.algos.pets.buffer import GpuReplayBuffer  # noqa: E402
from gpubench.algos.pets.ensemble_naive import NaiveEnsemble  # noqa: E402
from gpubench.algos.pets.mppi_naive import NaiveMPPI  # noqa: E402
from gpubench.common.config import load_config  # noqa: E402
from gpubench.common.logger import RunLogger  # noqa: E402
from gpubench.common.reporter import COMMON_KEYS, Reporter  # noqa: E402
from gpubench.common.timing import PhaseTimer  # noqa: E402
from gpubench.envs.costs import make_cost  # noqa: E402

CARTPOLE_CFG = SimpleNamespace(rew_scale_alive=1.0, rew_scale_terminated=-2.0, rew_scale_pole_pos=-1.0,
                               rew_scale_cart_vel=-0.01, rew_scale_pole_vel=-0.005, max_cart_pos=3.0)


# ---------------------------------------------------------------------------------------------- fakes
class FakeCartpole:
    """Vectorised, auto-resetting toy with Cartpole's observation layout and termination rule."""

    def __init__(self, n=8, dt=0.05):
        from gymnasium import spaces

        self.unwrapped = self
        self.num_envs, self.device, self.dt = n, "cpu", dt
        self.single_observation_space = spaces.Dict({"policy": spaces.Box(-10, 10, (4,))})
        self.single_action_space = spaces.Box(-1, 1, (1,))
        self.max_len = 100
        self.reset()

    def reset(self):
        self.s = (torch.rand(self.num_envs, 4) - 0.5) * 0.2
        self.t = torch.zeros(self.num_envs)
        return {"policy": self.s.clone()}, {}

    def step(self, a):
        pole, pole_v, cart, cart_v = self.s.unbind(-1)
        a = a[:, 0]
        pole_v = pole_v + self.dt * (9.8 * torch.sin(pole) - 2.0 * a)
        pole = pole + self.dt * pole_v
        cart_v = cart_v + self.dt * a
        cart = cart + self.dt * cart_v
        self.s = torch.stack([pole, pole_v, cart, cart_v], -1)
        self.t += 1
        term = (cart.abs() > 3.0) | (pole.abs() > math.pi / 2)
        trunc = (self.t >= self.max_len) & ~term
        rew, _ = make_cost("cartpole", CARTPOLE_CFG)(None, None, self.s)
        done = term | trunc
        fresh = (torch.rand(self.num_envs, 4) - 0.5) * 0.2
        self.s = torch.where(done[:, None], fresh, self.s)
        self.t = torch.where(done, torch.zeros_like(self.t), self.t)
        return {"policy": self.s.clone()}, rew, term, trunc, {}

    def close(self):
        pass


def _cfg(algo: str, tmp_path, **ov):
    cfg = load_config({"algo": algo, "task": "cartpole", "run.exp": "test", "run.results_root": str(tmp_path), **ov})
    return cfg


def _rows(logger) -> list[dict]:
    return [json.loads(l) for l in open(logger.dir / "metrics.jsonl")]


# ---------------------------------------------------------------------------------------------- reporter
def test_reporter_schema_cap_and_tau(tmp_path):
    cfg = _cfg("random", tmp_path)
    logger = RunLogger(cfg, tmp_path)
    rep = Reporter(logger, PhaseTimer("cpu"), "x", 4, stop_tau=10.0)
    assert not rep.emit(10, 1, {"reward_mean": 11.0}, extras={"loss": torch.tensor(2.0)})
    assert rep.emit(20, 2, {"reward_mean": 12.0})  # second consecutive row >= tau -> stop
    row = _rows(logger)[0]
    assert set(COMMON_KEYS) <= set(row) and row["x/loss"] == 2.0 and rep.summary()["stopped_at_tau"]
    rep2 = Reporter(logger, PhaseTimer("cpu"), "x", 4, max_wall_s=1e-9)
    assert rep2.emit(30, 3, {}) and rep2.summary()["stopped_by_wall_cap"]


# ---------------------------------------------------------------------------------------------- PETS knobs
class _PerfectModel(torch.nn.Module):
    E = 5

    def predict(self, obs, act, sample=True):
        return obs + act


def test_action_repeat_layout_and_goal():
    cost = lambda s, a, s2: (-(s2 - 3.0).square().sum(-1), torch.zeros(s2.shape[:-1], dtype=torch.bool))  # noqa: E731
    mp = NaiveMPPI(_PerfectModel(), cost, num_envs=4, act_dim=1, device="cpu", horizon=6, samples=100,
                   iterations=3, noise_std=0.5, action_repeat=3, noise_decay=0.5)
    coarse = torch.arange(2.0).view(1, 1, 2, 1).expand(4, 1, 2, 1)
    assert mp._to_fine(coarse)[0, 0, :, 0].tolist() == [0, 0, 0, 1, 1, 1]
    mp._phase = 2  # two env steps of the first segment already executed
    assert mp._to_fine(coarse)[0, 0, :, 0].tolist() == [0, 1, 1, 1, 1, 1]
    mp._phase = 0
    torch.manual_seed(0)
    s = torch.zeros(4, 1)
    for i in range(12):
        s = s + mp.plan(s)
        assert mp._phase == (i + 1) % 3
    assert (s - 3.0).abs().max() < 0.4
    with pytest.raises(ValueError):
        NaiveMPPI(_PerfectModel(), cost, 4, 1, "cpu", horizon=7, samples=100, action_repeat=3)


def test_buffer_term_and_rew():
    b = GpuReplayBuffer(8, 2, 1, "cpu")
    b.add(torch.zeros(3, 2), torch.zeros(3, 1), torch.zeros(3, 2), torch.tensor([True, False, False]),
          torch.tensor([False, True, False]), torch.tensor([1.0, 2.0, 3.0]))
    got = b.get_batch(torch.arange(3))
    assert got["term"].tolist() == [0, 1, 0] and got["rew"].tolist() == [1, 2, 3] and got["valid"].tolist() == [1, 0, 0]
    b.add(torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(1, 2), torch.tensor([True]))  # PETS call signature
    assert b.term[3] == 0 and b.rew[3] == 0


# ---------------------------------------------------------------------------------------------- SAC / MBPO
def test_sac_solves_one_step_bandit():
    """r = -(a - 0.5)^2, every transition terminal: the deterministic action must move to 0.5."""
    torch.manual_seed(0)
    sac = SAC(2, 1, hidden=64, layers=2, actor_lr=3e-3, critic_lr=3e-3, alpha_lr=3e-3, init_alpha=0.01)
    for _ in range(600):
        obs = torch.randn(128, 2)
        act = torch.rand(128, 1) * 2 - 1
        rew = -(act[:, 0] - 0.5) ** 2
        out = sac.update(obs, act, rew, obs, torch.ones(128), torch.ones(128))
    a = sac.act(torch.randn(256, 2), deterministic=True)
    assert abs(a.mean().item() - 0.5) < 0.15, a.mean().item()
    assert all(torch.is_tensor(v) for v in out.values())


def test_branched_rollout_masks_after_termination():
    model = NaiveEnsemble(4, 1, ensemble_size=5, hidden=8, layers=1)
    sac = SAC(4, 1, hidden=8, layers=1)
    real = GpuReplayBuffer(64, 4, 1, "cpu")
    real.add(torch.zeros(16, 4), torch.zeros(16, 1), torch.zeros(16, 4), torch.ones(16, dtype=torch.bool))
    model_buf = GpuReplayBuffer(64, 4, 1, "cpu")
    always_done = lambda s, a, s2: (torch.ones(s.shape[0]), torch.ones(s.shape[0], dtype=torch.bool))  # noqa: E731
    n = branched_rollout(model, sac, always_done, real, model_buf, n=8, k=3)
    assert n == 24 and model_buf.size == 24
    assert model_buf.valid[:8].tolist() == [1.0] * 8 and model_buf.valid[8:24].sum() == 0  # dead after step 1
    out = sac_update(sac, real, model_buf, batch=32, real_ratio=0.25)
    assert set(out) >= {"q_loss", "pi_loss", "alpha"}
    assert rollout_length([1, 10, 100, 200], 50) == 1 and rollout_length([1, 10, 100, 200], 150) == 6
    assert rollout_length(3, 10**9) == 3


# ---------------------------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("algo,ov", [
    ("pets_naive", {"algo.total_env_steps": 640, "algo.seed_steps": 160, "algo.train_every": 160, "algo.log_every": 160,
                    "algo.model.first_train_steps": 20, "algo.model.train_steps": 10, "algo.mppi.samples": 50,
                    "algo.mppi.horizon": 6, "algo.mppi.action_repeat": 2}),
    ("mbpo_naive", {"algo.total_env_steps": 640, "algo.seed_steps": 160, "algo.train_every": 160, "algo.log_every": 160,
                    "algo.rollout_every": 160, "algo.rollout_batch": 64, "algo.rollout_len": 2, "algo.utd": 0.5,
                    "algo.model.first_train_steps": 20, "algo.model.train_steps": 10, "algo.sac.batch_size": 32}),
    ("random", {"algo.min_episodes": 10, "algo.min_env_steps": 200, "algo.log_every_iters": 5}),
])
def test_runner_end_to_end_schema(algo, ov, tmp_path):
    pytest.importorskip("gymnasium")
    import importlib

    from gpubench.envs.isaac import GpuVecEnv

    cfg = _cfg(algo, tmp_path, **ov)
    cfg["algo"]["num_envs"] = 8
    logger = RunLogger(cfg, tmp_path)
    venv = GpuVecEnv(FakeCartpole(8), window=cfg["algo"].get("reward_window"))
    out = importlib.import_module(f"gpubench.algos.{cfg['algo']['name']}.run").run(cfg, logger, venv, CARTPOLE_CFG, PhaseTimer("cpu"))
    rows = _rows(logger)
    assert rows and out["env_steps"] > 0
    for r in rows:
        missing = set(COMMON_KEYS) - set(r)
        assert not missing, missing
    assert rows[-1]["reward_mean"] is not None
