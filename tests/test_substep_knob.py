"""Sim-cost knob (task.sim_substep_mult = k) must not change the reward (protocol v0.9.2). CPU only: a fake
LocomotionEnv reproduces Isaac Lab's step order (_get_dones updates potentials, then _get_rewards)."""

from types import SimpleNamespace

import torch

from gpubench.envs.costs import make_ant
from gpubench.envs.isaac import _undo_dt_scaled_progress

ANT = dict(heading_weight=0.5, up_weight=0.1, actions_cost_scale=0.005, energy_cost_scale=0.05,
           alive_reward_scale=0.5, death_cost=-2.0, termination_height=0.31)


class FakeLocomotion:
    """Progress = (prev_dist - dist) / sim_dt per control step; other terms constant; death_cost if terminated."""

    def __init__(self, sim_dt, dist0, dist1, terminated):
        self.sim_dt = sim_dt
        self.prev_potentials = -dist0 / sim_dt
        self.potentials = -dist1 / sim_dt
        self.reset_terminated = terminated

    def _get_rewards(self):
        r = (self.potentials - self.prev_potentials) + 0.5
        return torch.where(self.reset_terminated, torch.full_like(r, -2.0), r)


def _env(u):
    return SimpleNamespace(unwrapped=u)


def test_progress_reward_is_invariant_to_k():
    d0, d1 = torch.tensor([10.0, 10.0, 5.0]), torch.tensor([9.9, 9.95, 5.2])
    term = torch.tensor([False, True, False])
    ref = FakeLocomotion(1 / 120, d0, d1, term)._get_rewards()
    for k in (2, 4, 16, 32):
        u = FakeLocomotion(1 / 120 / k, d0, d1, term)
        assert not torch.allclose(u._get_rewards(), ref)  # the bug: progress scales with k
        _undo_dt_scaled_progress(_env(u), k)
        torch.testing.assert_close(u._get_rewards(), ref)


def test_k1_and_non_locomotion_envs_are_untouched():
    u = FakeLocomotion(1 / 120, torch.ones(2), torch.zeros(2), torch.zeros(2, dtype=torch.bool))
    f = u._get_rewards
    _undo_dt_scaled_progress(_env(u), 1)
    assert u._get_rewards == f
    cart = SimpleNamespace(_get_rewards=lambda: torch.ones(2))  # no potentials attribute
    g = cart._get_rewards
    _undo_dt_scaled_progress(_env(cart), 8)
    assert cart._get_rewards is g


def test_ant_cost_uses_original_decimation_under_the_knob():
    base = make_ant(SimpleNamespace(**ANT, decimation=2))
    for k in (4, 16):
        knob = make_ant(SimpleNamespace(**ANT, decimation=2 * k, gpubench_substep_mult=k))
        assert knob.params == base.params
        obs, act, nxt = torch.randn(64, 36), torch.rand(64, 8) * 2 - 1, torch.randn(64, 36)
        nxt[:, 0] = 0.6  # alive
        torch.testing.assert_close(knob(obs, act, nxt)[0], base(obs, act, nxt)[0])
