"""Task reward / termination as pure torch functions of (obs, act, next_obs).

These are what a model-based planner optimises on *predicted* states, so they must be computable from
the observation alone. They are ported from the Isaac Lab 2.3 Direct envs:
  - cartpole: isaaclab_tasks/direct/cartpole/cartpole_env.py::compute_rewards / _get_dones  (EXACT)
  - ant:      isaaclab_tasks/direct/locomotion/locomotion_env.py::compute_rewards            (APPROX.)

Ant approximation: the env's progress term is (potential_t - potential_{t-1}) with
potential = -dist_xy(torso, target) / sim.dt. The torso position is not in the observation, so we use
the equivalent velocity form  progress ~= decimation * v_towards_target, with v_towards_target taken from
the body-frame velocity rotated by the (yaw-frame) angle to the target. It ignores roll/pitch and the
intra-step velocity change; `scripts/check_costs.py` measures the error against true env rewards.

Everything is batched over arbitrary leading dims and does no host syncs (safe inside the planner).
Cfg values are read from the live env cfg object so they stay in sync with Isaac Lab.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

CostFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def make_cartpole(cfg) -> CostFn:
    # obs = [pole_pos, pole_vel, cart_pos, cart_vel]; action in [-1, 1] (scaled x100 N inside the env)
    alive, term_s = float(cfg.rew_scale_alive), float(cfg.rew_scale_terminated)
    pole_pos_s, cart_vel_s, pole_vel_s = float(cfg.rew_scale_pole_pos), float(cfg.rew_scale_cart_vel), float(cfg.rew_scale_pole_vel)
    max_cart = float(cfg.max_cart_pos)

    def fn(obs, act, next_obs):
        pole_pos, pole_vel, cart_pos, cart_vel = next_obs.unbind(-1)
        terminated = (cart_pos.abs() > max_cart) | (pole_pos.abs() > math.pi / 2)
        t = terminated.to(next_obs.dtype)
        r = alive * (1.0 - t) + term_s * t + pole_pos_s * pole_pos.square() + cart_vel_s * cart_vel.abs() + pole_vel_s * pole_vel.abs()
        return r, terminated

    return fn


# Ant (Isaac-Ant-Direct-v0) observation layout, 36 dims:
#   0 torso z | 1:4 vel_loc | 4:7 angvel_loc*s | 7 yaw | 8 roll | 9 angle_to_target | 10 up_proj
#   11 heading_proj | 12:20 dof_pos_scaled | 20:28 dof_vel*dof_vel_scale | 28:36 previous actions
ANT_ACTION_SLICE = slice(28, 36)


def make_ant(cfg) -> CostFn:
    heading_w, up_w = float(cfg.heading_weight), float(cfg.up_weight)
    act_cost_s, energy_s = float(cfg.actions_cost_scale), float(cfg.energy_cost_scale)
    alive_s, death_cost = float(cfg.alive_reward_scale), float(cfg.death_cost)
    term_h = float(cfg.termination_height)
    decimation = float(cfg.decimation)
    # motor_effort_ratio is all-ones in LocomotionEnv; dof_vel in obs is already * dof_vel_scale.

    def fn(obs, act, next_obs):
        z = next_obs[..., 0]
        vx, vy = next_obs[..., 1], next_obs[..., 2]
        ang = next_obs[..., 9]
        up_proj, heading_proj = next_obs[..., 10], next_obs[..., 11]
        dof_pos_scaled, dof_vel_scaled = next_obs[..., 12:20], next_obs[..., 20:28]

        progress = decimation * (vx * torch.cos(ang) + vy * torch.sin(ang))
        heading_r = torch.where(heading_proj > 0.8, torch.full_like(heading_proj, heading_w), heading_w * heading_proj / 0.8)
        up_r = torch.where(up_proj > 0.93, torch.full_like(up_proj, up_w), torch.zeros_like(up_proj))
        actions_cost = act.square().sum(-1)
        electricity = (act * dof_vel_scaled).abs().sum(-1)
        at_limit = (dof_pos_scaled > 0.98).to(next_obs.dtype).sum(-1)
        r = progress + alive_s + up_r + heading_r - act_cost_s * actions_cost - energy_s * electricity - at_limit
        terminated = z < term_h
        r = torch.where(terminated, torch.full_like(r, death_cost), r)
        return r, terminated

    return fn


def make_anymal(cfg) -> CostFn:
    raise NotImplementedError(
        "Anymal-C cost is not ported yet (step 1 runs PPO only on Anymal). It needs the velocity-command "
        "tracking terms plus contact/air-time terms, which are not all observable."
    )


FACTORIES = {"cartpole": make_cartpole, "ant": make_ant, "anymal": make_anymal}


def make_cost(kind: str, env_cfg) -> CostFn:
    return FACTORIES[kind](env_cfg)
