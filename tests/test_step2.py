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


# ---------------------------------------------------------------------------------------------- step 2b
def _tune_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("tune", ROOT / "scripts" / "tune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_round2_generation_is_wall_clock_only_and_same_configs():
    import yaml

    for m in ("pets", "mbpo"):
        for t in ("cartpole", "ant"):
            r1 = yaml.safe_load((ROOT / "configs" / "exp" / f"tune_{m}_{t}.yaml").read_text())
            r2 = yaml.safe_load((ROOT / "configs" / "exp" / f"tune_r2_{m}_{t}.yaml").read_text())
            assert [j["overrides"] for j in r1["jobs"]] == [j["overrides"] for j in r2["jobs"]]
            assert all("algo.total_env_steps=100000000" in j["cmd"] and "algo.max_wall_s=" in j["cmd"] for j in r2["jobs"])
            assert all(f"run.exp=tune_r2_{m}_{t}" in j["cmd"] for j in r2["jobs"])


def _fake_trial(root: Path, exp: str, name: str, curve, wall_per_row=10.0, cost=0.01):
    d = root / exp / name
    d.mkdir(parents=True)
    with open(d / "metrics.jsonl", "w") as f:
        for i, r in enumerate(curve):
            f.write(json.dumps({"wall_s": (i + 1) * wall_per_row, "reward_mean": r, "env_steps": (i + 1) * 100}) + "\n")
    (d / "summary.json").write_text(json.dumps({"wall_s": len(curve) * wall_per_row, "cost_usd": cost}))


def test_tune_report_prefers_newest_round_and_counts_all_spend(tmp_path):
    tune = _tune_module()
    gpu = tmp_path / "gpu"
    for i in range(8):  # round 1: never solves; round 2: trial 5 solves fastest
        _fake_trial(gpu, "tune_pets_cartpole", f"trial_{i:02d}", [10, 20, 30])
        _fake_trial(gpu, "tune_r2_pets_cartpole", f"trial_{i:02d}", [100, 270, 280] if i == 5 else [10, 20, 30, 40])
    budget = {"methods": ["pets"], "tasks": ["cartpole"], "max_wall_s": {"cartpole": 300}, "trials": 8, "seed": 100}
    tune.report(gpu, budget, None, tag="r2", tuned_dir=tmp_path / "tuned")
    import yaml

    best = yaml.safe_load((tmp_path / "tuned" / "pets_naive_cartpole.yaml").read_text())
    assert best["trial"] == "trial_05" and best["round"] == "tune_r2_pets_cartpole" and best["solved"]
    table = json.loads((gpu / "analysis" / "tuning_budget.json").read_text())[0]
    assert table["round"] == "tune_r2_pets_cartpole"
    assert abs(table["discarded_gpu_hours"] - 8 * 30 / 3600) < 1e-9  # round-1 spend still reported


def test_analyze_exclude_filter(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("analyze", ROOT / "scripts" / "analyze.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    for name in ("pets_naive_cartpole_s0", "ppo_cartpole_s0"):
        d = tmp_path / "step2_baselines" / name
        d.mkdir(parents=True)
        algo = name.split("_")[0]
        (d / "metrics.jsonl").write_text(json.dumps({"algo": algo, "impl": "x", "task": "cartpole", "seed": 0,
                                                     "wall_s": 1, "env_steps": 1, "reward_mean": 1}) + "\n")
    runs = an.load_runs(tmp_path, ["step2_baselines"], ["step2_baselines/pets_*"])
    assert [r["algo"] for r in runs] == ["ppo"]


def test_exp_never_destroys_a_stopped_instance(monkeypatch):
    from types import SimpleNamespace as NS

    from gpubench.launcher import cli

    destroyed = []
    monkeypatch.setattr(cli.I, "show", lambda i: {"id": i, "actual_status": "exited", "intended_status": "stopped"})
    monkeypatch.setattr(cli.I, "destroy", lambda i: destroyed.append(i))
    monkeypatch.setattr(cli, "expand", lambda f: {"exp": "x", "jobs": [1]})

    def boom(gpu, offer, allow_dirty, created):  # simulate: created, then the run fails while the instance is stopped
        created.append(123)
        raise RuntimeError("ssh failed")

    monkeypatch.setattr(cli, "_up_with_retry", boom)
    args = NS(exp=["e.yaml"], max_hours=1, instance=None, gpu="rtx4090", offer=None, allow_dirty=False, keep=False)
    assert cli.cmd_exp(args) == 1
    assert destroyed == []  # stopped -> kept for manual recovery
    monkeypatch.setattr(cli.I, "show", lambda i: None)  # gone / never existed -> clean up
    assert cli.cmd_exp(args) == 1 and destroyed == [123]


# ---------------------------------------------------------------------------------------------- step 2c
def test_batched_ensemble_matches_naive():
    from gpubench.algos.pets.ensemble_batched import BatchedEnsemble

    torch.manual_seed(0)
    naive = NaiveEnsemble(6, 2, ensemble_size=5, hidden=32, layers=3, action_obs_slice=slice(4, 6))
    naive.in_mean.normal_()
    naive.in_std.uniform_(0.5, 2.0)
    naive.out_mean.normal_()
    naive.out_std.uniform_(0.5, 2.0)
    batched = BatchedEnsemble(6, 2, ensemble_size=5, hidden=32, layers=3, action_obs_slice=slice(4, 6)).load_from_naive(naive)
    obs, act, nxt = torch.randn(5, 64, 6), torch.randn(5, 64, 2), torch.randn(5, 64, 6)
    w = (torch.rand(5, 64) > 0.2).float()
    assert torch.allclose(naive.predict(obs, act, sample=False), batched.predict(obs, act, sample=False), atol=1e-5)
    ln, mn = naive.loss(obs, act, nxt, w)
    lb, mb = batched.loss(obs, act, nxt, w)
    assert torch.allclose(ln, lb, atol=1e-5) and torch.allclose(mn, mb, atol=1e-5)
    o2, a2, n2, w2 = obs[0], act[0], nxt[0], w[0]
    assert torch.allclose(naive.eval_mse(o2, a2, n2, w2), batched.eval_mse(o2, a2, n2, w2), atol=1e-5)
    # gradients flow to every member's stacked weights
    lb.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in batched.W)


def test_early_stopping_halts_on_plateau():
    from gpubench.algos.pets.ensemble_batched import BatchedEnsemble
    from gpubench.algos.pets.run import train_model

    torch.manual_seed(0)
    buf = GpuReplayBuffer(4000, 3, 1, "cpu")
    x = torch.randn(4000, 3)
    buf.add(x, torch.zeros(4000, 1), x + 0.01 * torch.randn(4000, 3), torch.ones(4000, dtype=torch.bool))  # pure noise
    model = BatchedEnsemble(3, 1, ensemble_size=5, hidden=16, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _, _, done = train_model(model, opt, buf, steps=5000, batch=64,
                             early_stop={"holdout_frac": 0.1, "check_every": 20, "patience": 3})
    assert done < 5000  # nothing learnable -> holdout stops improving -> stopped early
    _, _, done2 = train_model(model, opt, buf, steps=50, batch=64)
    assert done2 == 50


def test_elite_refit_shrinks_sigma_and_reaches_goal():
    cost = lambda s, a, s2: (-(s2 - 3.0).square().sum(-1), torch.zeros(s2.shape[:-1], dtype=torch.bool))  # noqa: E731
    torch.manual_seed(0)
    mp = NaiveMPPI(_PerfectModel(), cost, num_envs=4, act_dim=1, device="cpu", horizon=6, samples=100,
                   iterations=3, noise_std=0.5, elite_frac=0.1)
    s = torch.zeros(4, 1)
    for _ in range(10):
        s = s + mp.plan(s)
    assert (s - 3.0).abs().max() < 0.4 and mp.n_elite == 10


def test_pets_batched_end_to_end(tmp_path):
    pytest.importorskip("gymnasium")
    import importlib

    from gpubench.envs.isaac import GpuVecEnv

    cfg = _cfg("pets_batched", tmp_path, **{"algo.total_env_steps": 480, "algo.seed_steps": 160, "algo.train_every": 160,
                                            "algo.log_every": 160, "algo.model.first_train_steps": 30,
                                            "algo.model.batch_size": 64, "algo.model.check_every": 10,
                                            "algo.mppi.samples": 50, "algo.mppi.horizon": 6, "algo.mppi.elite_frac": 0.2})
    cfg["algo"]["num_envs"] = 8
    logger = RunLogger(cfg, tmp_path)
    venv = GpuVecEnv(FakeCartpole(8))
    importlib.import_module("gpubench.algos.pets.run").run(cfg, logger, venv, CARTPOLE_CFG, PhaseTimer("cpu"))
    rows = _rows(logger)
    assert rows and rows[-1]["impl"] == "batched" and all(set(COMMON_KEYS) <= set(r) for r in rows)
    assert any(r.get("pets/model_train_steps") for r in rows)
