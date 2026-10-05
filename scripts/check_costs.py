"""Validate envs/costs.py against the simulator's own rewards on real rollouts (runs on the instance).

    python scripts/check_costs.py --task cartpole     # expect max |err| < 1e-4  (exact port)
    python scripts/check_costs.py --task ant          # approximate port: reports error + correlation

Only non-done transitions are compared (for done envs Isaac Lab returns the post-reset observation).
Writes results/checks/costs_<task>.json. Exit code 1 if an exact task exceeds tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXACT_TOL = {"cartpole": 1e-4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    from gpubench.envs.isaac import launch_app

    app = launch_app(device=args.device, headless=True)
    import torch
    import yaml

    from gpubench.envs.costs import make_cost
    from gpubench.envs.isaac import make_env

    task = yaml.safe_load((ROOT / "configs" / "task" / f"{args.task}.yaml").read_text())
    env, env_cfg = make_env(task, args.device, seed=0, num_envs=args.num_envs)
    cost = make_cost(task["cost"], env_cfg)
    u = env.unwrapped
    obs = env.reset()[0]["policy"]
    errs, r_env_all, r_hat_all, term_agree = [], [], [], []
    for _ in range(args.steps):
        act = (torch.rand(args.num_envs, u.single_action_space.shape[-1], device=u.device) * 2 - 1) * 0.5
        nobs, rew, term, trunc, _ = env.step(act)
        nobs = nobs["policy"]
        r_hat, t_hat = cost(obs, act, nobs)
        keep = ~(term | trunc)
        errs.append((r_hat - rew)[keep].abs())
        r_env_all.append(rew[keep])
        r_hat_all.append(r_hat[keep])
        term_agree.append((t_hat[keep] == term[keep]).float())
        obs = nobs
    e, re, rh = torch.cat(errs), torch.cat(r_env_all), torch.cat(r_hat_all)
    corr = torch.corrcoef(torch.stack([re, rh]))[0, 1].item()
    res = {"task": args.task, "n": int(e.numel()), "max_abs_err": e.max().item(), "mean_abs_err": e.mean().item(),
           "reward_std": re.std().item(), "corr": corr, "termination_agreement": torch.cat(term_agree).mean().item()}
    tol = EXACT_TOL.get(args.task)
    res["pass"] = (res["max_abs_err"] < tol) if tol is not None else (corr > 0.9)
    print(json.dumps(res, indent=2))
    out = ROOT / "results" / "checks"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"costs_{args.task}.json").write_text(json.dumps(res, indent=2))
    env.close()
    app.close()
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
