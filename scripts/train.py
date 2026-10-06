"""Single entrypoint for every run.

    python scripts/train.py algo=ppo task=cartpole seed=0
    python scripts/train.py algo=pets_naive task=cartpole seed=0 algo.sync_debug=warn
    python scripts/train.py algo=ppo task=ant seed=0 profile.start=20 profile.length=30   # (under nsys)

Writes results/<run.exp>/<run_id>/{run_meta.json, config.json, metrics.jsonl, summary.json}.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .` too

from gpubench.common.config import dump, load_config  # noqa: E402


def main(argv: list[str]) -> int:
    cfg = load_config(argv)
    print(dump(cfg), flush=True)

    # Isaac Sim must be up before any isaaclab.* import (other than isaaclab.app).
    from gpubench.envs.isaac import launch_app

    app = launch_app(device=cfg["device"], headless=True)

    import torch

    from gpubench.common.logger import RunLogger
    from gpubench.common.seeding import seed_everything
    from gpubench.common.timing import PhaseTimer
    from gpubench.envs.isaac import GpuVecEnv, make_env

    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("tf32", True))  # Isaac Lab's rsl_rl default
    torch.backends.cudnn.allow_tf32 = bool(cfg.get("tf32", True))
    seed_everything(cfg["seed"], deterministic=cfg.get("deterministic", False))

    logger = RunLogger(cfg, cfg["run"]["results_root"])
    print(f"[train] run dir: {logger.dir}", flush=True)
    timer = PhaseTimer(cfg["device"])
    status, summary = "ok", {}
    env = None
    try:
        t_env = time.perf_counter()
        num_envs = cfg["algo"].get("num_envs") or cfg["task"].get("num_envs")
        env, env_cfg = make_env(cfg["task"], cfg["device"], cfg["seed"], num_envs=num_envs)
        summary["env_create_s"] = round(time.perf_counter() - t_env, 2)
        summary["control_dt"] = env_cfg.sim.dt * env_cfg.decimation
        summary["physics_dt"] = env_cfg.sim.dt
        summary["decimation"] = env_cfg.decimation

        algo = cfg["algo"]["name"]
        if algo == "ppo":
            from gpubench.algos.ppo.run import run
            summary.update(run(cfg, logger, env, env_cfg, timer))
        elif algo == "pets":
            from gpubench.algos.pets.run import run
            venv = GpuVecEnv(env, clip_actions=cfg["algo"].get("clip_actions", 1.0), window=cfg["algo"].get("reward_window"))
            summary["reward_window"] = venv.window
            summary.update(run(cfg, logger, venv, env_cfg, timer))
        else:
            raise ValueError(f"unknown algo {algo!r}")
    except Exception:
        status = "failed"
        summary["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        summary["status"] = status
        logger.close(summary)
        print(f"[train] {status}: {logger.dir}", flush=True)
    from gpubench.envs.isaac import shutdown

    shutdown(app, 0 if status == "ok" else 1, env)  # does not return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
