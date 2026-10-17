"""Record a video of a trained policy in Isaac Lab (runs on the instance; needs RT cores, e.g. an RTX 4090).

    python scripts/record.py results/videos/ppo_ant_s2                 # run dir trained with run.save_policy=true
    python scripts/record.py --random --task cartpole --steps 60       # rendering smoke test, no policy

Recording is a separate process from training on purpose: with cameras enabled Isaac Lab renders on every
step, which would distort the benchmark's timings. The policy is loaded from the run dir:
    ppo   rsl_rl/model_<last>.pt (written when learn() completes) + agent_cfg.json
    pets  policy.pt (ensemble state) -> the run's own planner (same impl), final exploration noise
    mbpo  policy.pt (SAC state) -> deterministic actor
Writes <out>/<run_id>.mp4 (default results/recordings/) and <run_id>.json (episode returns seen during the recording).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Camera presets on env_cfg.viewer: Cartpole from the front over a few envs, legged robots followed.
VIEWS = {
    "cartpole": dict(origin_type="env", env_index=0, eye=(0.0, 9.0, 3.0), lookat=(0.0, 0.0, 1.5)),
    "ant": dict(origin_type="asset_root", asset_name="robot", env_index=0, eye=(4.0, 4.0, 2.5), lookat=(0.0, 0.0, 0.5)),
    "anymal": dict(origin_type="asset_root", asset_name="robot", env_index=0, eye=(3.0, 3.0, 2.0), lookat=(0.0, 0.0, 0.4)),
}


def _camera(task_name: str):
    def hook(env_cfg):
        v = env_cfg.viewer
        v.resolution = (1280, 720)
        for k, val in VIEWS.get(task_name, {}).items():
            setattr(v, k, val)
    return hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", type=Path)
    p.add_argument("--random", action="store_true", help="random actions (rendering smoke test)")
    p.add_argument("--task", help="with --random: task name (configs/task/<task>.yaml)")
    p.add_argument("--steps", type=int, default=600, help="control steps to record")
    p.add_argument("--num-envs", type=int, help="default: 16 (PPO/random/MBPO), the training count for PETS")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "recordings")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    if a.random:
        import yaml

        task = yaml.safe_load((ROOT / "configs" / "task" / f"{a.task}.yaml").read_text())
        task.setdefault("name", a.task)
        cfg = {"task": task, "algo": {"name": "random"}, "seed": 0, "run": {"run_id": f"random_{a.task}"}}
        run_dir = None
    else:
        run_dir = a.run_dir.resolve()
        cfg = json.loads((run_dir / "config.json").read_text())
    algo, task = cfg["algo"]["name"], cfg["task"]
    run_id = run_dir.name if run_dir else cfg["run"]["run_id"]

    from gpubench.envs.isaac import launch_app

    app = launch_app(device=a.device, headless=True, enable_cameras=True)

    import gymnasium as gym
    import torch

    from gpubench.common.seeding import seed_everything
    from gpubench.envs.isaac import GpuVecEnv, make_env, shutdown

    seed_everything(int(cfg.get("seed", 0)))
    n = a.num_envs or (int(cfg["algo"]["num_envs"]) if algo == "pets" else 16)
    env, env_cfg = make_env(task, a.device, int(cfg.get("seed", 0)), num_envs=n, render_mode="rgb_array",
                            cfg_hook=_camera(task["name"]))
    tmp = a.out / f"_tmp_{run_id}"
    env = gym.wrappers.RecordVideo(env, video_folder=str(tmp), step_trigger=lambda s: s == 0, video_length=a.steps,
                                   disable_logger=True)

    code, info = 0, {"run_id": run_id, "algo": algo, "task": task["name"], "seed": cfg.get("seed"), "num_envs": n,
                     "steps": a.steps, "note": "policy retrained for the video with the benchmark config (videos.yaml)"}
    try:
        returns = []
        if algo == "ppo":
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg = json.loads((run_dir / "agent_cfg.json").read_text())
            venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
            ckpts = sorted((run_dir / "rsl_rl").glob("model_*.pt"), key=lambda f: int(re.findall(r"\d+", f.stem)[-1]))
            if not ckpts:
                raise FileNotFoundError(f"no rsl_rl/model_*.pt in {run_dir} (did learn() complete?)")
            runner = OnPolicyRunner(venv, agent_cfg, log_dir=None, device=a.device)
            runner.load(str(ckpts[-1]))
            policy = runner.get_inference_policy(device=a.device)
            info["checkpoint"] = ckpts[-1].name
            obs = venv.get_observations()
            ret = torch.zeros(venv.num_envs, device=a.device)
            for _ in range(a.steps):
                with torch.inference_mode():
                    obs, rew, dones, _ = venv.step(policy(obs))
                ret += rew
                done = dones.bool()
                returns += ret[done].tolist()
                ret[done] = 0
        else:
            venv = GpuVecEnv(env, clip_actions=cfg["algo"].get("clip_actions", 1.0))
            act_fn = _model_based_policy(algo, cfg, run_dir, venv, env_cfg, a.device)
            obs = venv.reset()
            ret = torch.zeros(venv.num_envs, device=a.device)
            for _ in range(a.steps):
                act, after = act_fn(obs)
                obs, rew, term, trunc, _ = venv.step(act)
                done = term | trunc
                after(done)
                ret += rew
                returns += ret[done].tolist()
                ret[done] = 0
        info["episodes"] = len(returns)
        info["return_mean"] = sum(returns) / len(returns) if returns else None
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        env.close()  # flushes the video
    vids = sorted(tmp.glob("*.mp4")) if tmp.exists() else []
    if vids and code == 0:
        a.out.mkdir(parents=True, exist_ok=True)
        shutil.move(str(vids[0]), a.out / f"{run_id}.mp4")
        info["video"] = f"{run_id}.mp4"
        info["video_bytes"] = (a.out / f"{run_id}.mp4").stat().st_size
    elif code == 0:
        print("no video was written (is rendering working? enable_cameras + RT cores needed)", flush=True)
        code = 1
    shutil.rmtree(tmp, ignore_errors=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{run_id}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    shutdown(app, code, None)


def _model_based_policy(algo, cfg, run_dir, venv, env_cfg, device):
    """-> act(obs) returning (actions, after_step(done)) for random / PETS / MBPO."""
    import torch

    B, do, da = venv.num_envs, venv.obs_dim, venv.act_dim
    if algo == "random":
        return lambda obs: (torch.rand(B, da, device=device) * 2 - 1, lambda done: None)
    state = torch.load(run_dir / "policy.pt", map_location=device)
    a = cfg["algo"]
    if algo == "pets":
        from gpubench.algos.pets.run import build_model, build_planner
        from gpubench.envs.costs import make_cost

        impl = a.get("impl", "naive")
        model, _ = build_model(a["model"], do, da, cfg["task"]["cost"], device, impl)
        model.load_state_dict(state["model"])
        mp = a["mppi"]
        planner = build_planner(impl, model, make_cost(cfg["task"]["cost"], env_cfg), B, da, device, mp)
        planner.sigma = mp.get("noise_std_final") or mp["noise_std"]  # end-of-training exploration noise
        return lambda obs: (planner.plan(obs), planner.reset)
    if algo == "mbpo":
        from gpubench.algos.mbpo.sac import SAC

        s = a["sac"]
        sac = SAC(do, da, hidden=s["hidden"], layers=s["layers"]).to(device)
        sac.load_state_dict(state["sac"])
        return lambda obs: (sac.act(obs, deterministic=True), lambda done: None)
    raise ValueError(f"cannot record algo {algo!r}")


if __name__ == "__main__":
    sys.exit(main())
