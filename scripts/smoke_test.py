"""Instance acceptance test: Isaac Sim boots headless on this GPU and every task steps on the device.

    python scripts/smoke_test.py --out /workspace/smoke.json            # all tasks, one subprocess each
    python scripts/smoke_test.py --task cartpole --num-envs 256          # single task (used internally)

Each task runs in its own process (one Kit app per process is the robust pattern) and reports
env-steps/s at a small and a large batch, plus boot time. Exit code != 0 if any task fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TASKS = ["cartpole", "ant", "anymal"]


def single(task_name: str, num_envs: int, steps: int, device: str) -> dict:
    t0 = time.perf_counter()
    from gpubench.envs.isaac import launch_app

    app = launch_app(device=device, headless=True)
    boot_s = time.perf_counter() - t0
    import torch
    import yaml

    from gpubench.envs.isaac import make_env

    task = yaml.safe_load((ROOT / "configs" / "task" / f"{task_name}.yaml").read_text())
    env, _ = make_env(task, device, seed=0, num_envs=num_envs)
    obs, _ = env.reset()
    u = env.unwrapped
    assert obs["policy"].is_cuda, "observations are not on the GPU"
    act_dim = u.single_action_space.shape[-1]
    for _ in range(10):  # warmup
        env.step(torch.zeros(num_envs, act_dim, device=u.device))
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        env.step(torch.rand(num_envs, act_dim, device=u.device) * 2 - 1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    res = {
        "task": task_name, "ok": True, "num_envs": num_envs, "steps": steps, "boot_s": round(boot_s, 1),
        "env_steps_per_s": round(num_envs * steps / dt, 1), "obs_dim": int(obs["policy"].shape[-1]), "act_dim": int(act_dim),
        "gpu": torch.cuda.get_device_name(0),
    }
    return res, app, env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out")
    p.add_argument("--tasks", default=",".join(TASKS))
    args = p.parse_args()

    if args.task:  # child mode: print one JSON line prefixed for the parent to find
        from gpubench.envs.isaac import shutdown

        res, app, env = single(args.task, args.num_envs, args.steps, args.device)
        print("SMOKE_RESULT " + json.dumps(res), flush=True)  # before shutdown: app.close() ends the process
        shutdown(app, 0, env)

    from gpubench.common import sysinfo

    report = {"sys": sysinfo.collect(), "tasks": []}
    ok = True
    for task in args.tasks.split(","):
        for n in (64, args.num_envs):
            t0 = time.perf_counter()
            proc = subprocess.run([sys.executable, __file__, "--task", task, "--num-envs", str(n), "--steps", str(args.steps),
                                   "--device", args.device], capture_output=True, text=True, timeout=1800)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("SMOKE_RESULT ")]
            if proc.returncode == 0 and lines:
                res = json.loads(lines[-1][len("SMOKE_RESULT "):])
            else:
                ok = False
                res = {"task": task, "ok": False, "num_envs": n, "returncode": proc.returncode,
                       "stderr_tail": proc.stderr[-4000:], "stdout_tail": proc.stdout[-2000:]}
            res["wall_s"] = round(time.perf_counter() - t0, 1)
            report["tasks"].append(res)
            print(json.dumps(res), flush=True)
    report["ok"] = ok
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
