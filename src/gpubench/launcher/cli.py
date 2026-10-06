"""gpubench-vast: run the benchmark on any Vast.ai GPU from your laptop.

Typical flow (see README):
    gpubench-vast doctor                             # checks API key, ssh key, git, token
    gpubench-vast offers rtx4090                     # what's available / price
    gpubench-vast exp rtx4090 configs/exp/step1_ppo.yaml configs/exp/step1_pets.yaml
        -> up -> bootstrap -> run all jobs -> pull results -> destroy   (destroy also on failure/Ctrl-C)

Interactive / iterative work:
    gpubench-vast up rtx4090          gpubench-vast ssh           gpubench-vast sync   (push, then checkout on box)
    gpubench-vast run configs/exp/x.yaml     gpubench-vast status    gpubench-vast pull    gpubench-vast down
Instance ids default to the most recent instance; a GPU alias (e.g. `rtx4090`) also works.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import instance as I
from .jobs import expand
from .remote import REMOTE_REPO, Remote

RESULTS = I.ROOT / "results"
SECRET_KEYS = ("GH_TOKEN", "VAST_API_KEY", "GPUBENCH_GH_TOKEN")


def _redact(d):
    if isinstance(d, dict):
        return {k: ("***" if k in SECRET_KEYS else _redact(v)) for k, v in d.items()}
    return d


def _remote(ident) -> tuple[int, dict, Remote]:
    inst_id = I.resolve_id(ident)
    inst = I.show(inst_id)
    if not inst:
        raise SystemExit(f"instance {inst_id} not found on Vast (destroyed?). `gpubench-vast ls`")
    return inst_id, inst, Remote.from_instance(inst)


def _gpu_alias(inst_id: int, inst: dict) -> str:
    return I.load_state().get(str(inst_id), {}).get("gpu") or (inst.get("extra_env") or {}).get("GPUBENCH_GPU") or "unknown"


# ----------------------------------------------------------------------------------------------------
def _check_ssh_key() -> str:
    """The key ssh will actually offer must be one registered on the Vast account."""
    registered = {k.get("public_key", "").split()[1] for k in (I.vast_client().show_ssh_keys() or [])
                  if len(k.get("public_key", "").split()) > 1}
    own = os.environ.get("GPUBENCH_SSH_KEY")
    if own:
        pub = Path(own + ".pub")
        if not Path(own).exists() or not pub.exists():
            raise RuntimeError(f"GPUBENCH_SSH_KEY={own}: private key or {pub.name} not found")
        if pub.read_text().split()[1] not in registered:
            raise RuntimeError(f"{pub} is not registered on Vast (cloud.vast.ai/manage-keys)")
        return f"{Path(own).name} (GPUBENCH_SSH_KEY) is registered"
    # Without GPUBENCH_SSH_KEY, OpenSSH only offers its default key names.
    defaults = ["id_rsa", "id_ecdsa", "id_ed25519"]
    home = Path.home() / ".ssh"
    for name in defaults:
        pub = home / f"{name}.pub"
        if pub.exists() and pub.read_text().split()[1] in registered:
            return f"~/.ssh/{name} is registered"
    matches = [p.stem for p in home.glob("*.pub") if len(p.read_text().split()) > 1 and p.read_text().split()[1] in registered]
    hint = f"; set GPUBENCH_SSH_KEY to {home / matches[0]} in .env" if matches else "; add a key at cloud.vast.ai/manage-keys"
    raise RuntimeError("no default key (" + ", ".join(defaults) + ") is registered on Vast" + hint)


def cmd_doctor(a):
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            msg = fn()
            print(f"  [ok]   {name}: {msg}")
        except BaseException as e:  # SystemExit included
            ok = False
            print(f"  [FAIL] {name}: {e}")

    def require(cond, fail_msg, ok_msg):
        if not cond:
            raise RuntimeError(fail_msg)
        return ok_msg

    I.load_dotenv()
    check("vast api key", lambda: f"{len(I.vast_client().show_instances())} running instance(s)")
    check("vast ssh keys", lambda: (lambda k: require(k, "none registered: add your public key at cloud.vast.ai/manage-keys",
                                                        f"{len(k)} registered"))(I.vast_client().show_ssh_keys()))
    check("ssh key matches Vast", _check_ssh_key)
    check("ssh client", lambda: subprocess.run(["ssh", "-V"], capture_output=True, text=True).stderr.strip())
    check("github token", lambda: require(os.environ.get("GPUBENCH_GH_TOKEN"),
                                          "set GPUBENCH_GH_TOKEN in .env (fine-grained PAT, Contents: read-only)", "GPUBENCH_GH_TOKEN set"))
    check("git", lambda: (lambda g: f"{g['repo']} @ {g['sha'][:10]} (pushed)")(I.git_state()))
    check("lock file", lambda: "requirements/remote.lock.txt present" if (I.ROOT / "requirements/remote.lock.txt").exists()
          else "absent -> first instance installs loose pins; run `gpubench-vast freeze` afterwards")
    print("doctor:", "all good" if ok else "fix the failures above")
    return 0 if ok else 1


def cmd_offers(a):
    offers = I.search(a.gpu, limit=a.n)
    print(f"{'offer':>10} {'gpu':<10} {'$/hr':>6} {'rel':>5} {'cuda':>5} {'down':>6} {'cores':>5}  {'cpu':<38} location")
    for o in offers:
        cpu = (o.get("cpu_name") or "?").replace(" Processor", "").replace("-Core", "c")[:38]
        print(f"{o['id']:>10} {o.get('gpu_name', ''):<10} {o.get('dph_total', 0):>6.3f} {o.get('reliability2', o.get('reliability', 0)):>5.3f} "
              f"{o.get('cuda_max_good', ''):>5} {o.get('inet_down', 0):>6.0f} {o.get('cpu_cores_effective', 0):>5.0f}  {cpu:<38} "
              f"{o.get('geolocation', '')}")
    return 0


def _up(gpu: str, offer: int | None, allow_dirty: bool, wait: bool = True,
        created: list | None = None) -> tuple[int, dict, Remote | None]:
    """Create + wait. The new id is appended to `created` immediately, so callers can always clean up.

    An instance that never reaches `running` is useless and is destroyed here. One that runs but fails
    bootstrap is left to the caller (kept by `up` for debugging, destroyed by `exp`/`smoke`).
    """
    git = I.git_state(allow_dirty=allow_dirty)
    res = I.create(gpu, offer, git)
    inst_id = res["id"]
    if created is not None:
        created.append(inst_id)
    price = res["offer"].get("dph_total")
    print(f"created instance {inst_id} on offer {res['offer']['id']} ({f'${price:.3f}' if price else 'price on start'}/hr) @ {git['sha'][:10]}")
    if not wait:
        return inst_id, {}, None
    try:
        inst = I.wait_running(inst_id)
    except BaseException:
        print(f"instance {inst_id} never started; destroying it (try another offer: `gpubench-vast offers {gpu}`)")
        I.destroy(inst_id)
        if created is not None:
            created.remove(inst_id)
        raise
    rem = I.wait_ready(inst)
    _stamp_price(rem, inst)
    return inst_id, inst, rem


def _up_with_retry(gpu: str, offer: int | None, allow_dirty: bool, created: list, attempts: int = 3):
    """`_up`, moving on to the next-cheapest offer when a host never starts its container."""
    if offer is not None:
        return _up(gpu, offer, allow_dirty, created=created)
    tried: set = set()
    last_err = None
    for _ in range(attempts):
        cands = [o for o in I.search(gpu, limit=10) if o["id"] not in tried]
        if not cands:
            break
        tried.add(cands[0]["id"])
        n_before = len(created)
        try:
            return _up(gpu, cands[0]["id"], allow_dirty, created=created)
        except (RuntimeError, TimeoutError) as e:
            if len(created) > n_before:  # started but failed later (bootstrap): not a host-start problem
                raise
            last_err = e
            print(f"offer {cands[0]['id']} failed to start ({e}); trying the next one")
    raise RuntimeError(f"no {gpu} offer started after {len(tried)} attempt(s): {last_err}")


def _stamp_price(rem: Remote, inst: dict):
    """Write the instance's real $/hr into the job environment (cost_usd in every run summary)."""
    dph = inst.get("dph_total")
    if dph:
        rem.run(f"sed -i '/^export GPUBENCH_DPH=/d' /etc/gpubench.env && echo 'export GPUBENCH_DPH=\"{dph:.4f}\"' >> /etc/gpubench.env",
                cwd=None)


def cmd_rebootstrap(a):
    """Re-run infra/bootstrap.sh on a live instance (after `sync`), detached; then wait for it."""
    inst_id, inst, rem = _remote(a.id)
    rem.run("rm -f /workspace/.ready /workspace/.bootstrap_failed", cwd=None)
    rem.detach("bootstrap", "bash infra/bootstrap.sh", "/workspace/bootstrap.log")
    print("bootstrap restarted; waiting (Ctrl-C is safe, it keeps running on the instance)")
    I.wait_ready(inst)
    _stamp_price(rem, inst)
    print(f"ready: gpubench-vast ssh {inst_id}")
    return 0


def cmd_wait(a):
    """Wait for an instance's bootstrap to finish (e.g. after Ctrl-C on `up`)."""
    inst_id, inst, rem = _remote(a.id)
    I.wait_ready(inst)
    _stamp_price(rem, inst)
    print(f"ready: gpubench-vast ssh {inst_id}")
    return 0


def cmd_up(a):
    inst_id, _, _ = _up(a.gpu, a.offer, a.allow_dirty, wait=not a.no_wait)
    print(f"ready: gpubench-vast ssh {inst_id}")
    return 0


def cmd_ls(a):
    rows = I.vast_client().show_instances() or []
    local = I.load_state()
    if not rows:
        print("no instances running on your Vast account")
    for r in rows:
        age = (time.time() - r.get("start_date", time.time())) / 3600
        tag = "gpubench" if str(r["id"]) in local else "other"
        print(f"{r['id']:>10}  {r.get('gpu_name', ''):<12} {r.get('actual_status', ''):<9} ${r.get('dph_total', 0):.3f}/hr  "
              f"{age:5.2f} h  ~${age * (r.get('dph_total') or 0):.2f}  label={r.get('label')}  [{tag}]")
    stale = set(local) - {str(r["id"]) for r in rows}
    for s in stale:
        I.forget(int(s))
    return 0


def cmd_show(a):
    inst_id = I.resolve_id(a.id)
    print(json.dumps(_redact(I.show(inst_id)), indent=2, default=str))
    return 0


def cmd_ssh(a):
    _, _, rem = _remote(a.id)
    return rem.interactive(a.cmd)


def cmd_sync(a):
    """Checkout a new commit on a live instance without re-bootstrapping."""
    inst_id, inst, rem = _remote(a.id)
    git = I.git_state(allow_dirty=a.allow_dirty)
    _stamp_price(rem, inst)
    I.load_dotenv()
    tok = os.environ["GPUBENCH_GH_TOKEN"]
    url = f"https://x-access-token:{tok}@github.com/{git['repo']}.git"
    rem.run(f"git fetch -q {shlex.quote(url)} '+refs/heads/*:refs/remotes/origin/*' && git checkout -q {git['sha']} "
            f"&& sed -i 's/^export GIT_SHA=.*/export GIT_SHA=\"{git['sha']}\"/' /etc/gpubench.env "
            f"&& python -m pip install -q -e . --no-deps")
    I.remember(inst_id, git_sha=git["sha"])
    print(f"instance {inst_id} now at {git['sha'][:10]}")
    return 0


def _start_exp(rem: Remote, exp_file: str, sha: str | None) -> str:
    spec = expand(exp_file)
    spec["git_sha"] = sha
    remote_file = f"/workspace/jobs/{spec['exp']}.json"
    rem.put_text(remote_file, json.dumps(spec, indent=2))
    rem.detach(f"exp_{spec['exp']}", f"python -m gpubench.launcher.jobrunner {remote_file}",
               f"/workspace/logs/{spec['exp']}.runner.log")
    print(f"started {spec['exp']}: {len(spec['jobs'])} job(s)  (tmux session exp_{spec['exp']} on the instance)")
    return spec["exp"]


def cmd_run(a):
    inst_id, inst, rem = _remote(a.id)
    sha = I.load_state().get(str(inst_id), {}).get("git_sha")
    for i, f in enumerate(a.exp):
        exp = _start_exp(rem, f, sha)
        if i < len(a.exp) - 1:  # jobs must not overlap on one GPU: wait before starting the next file
            _wait_exp(rem, exp)
    return 0


def _exp_status(rem: Remote, exp: str) -> dict | None:
    p = rem.run(f"cat /workspace/jobs/{exp}.status.json", check=False, cwd=None, timeout=60)
    return json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None


def _print_status(st: dict):
    jobs = st.get("jobs", {})
    counts = {}
    for j in jobs.values():
        counts[j["state"]] = counts.get(j["state"], 0) + 1
    running = [n for n, j in jobs.items() if j["state"] == "running"]
    print(f"  {st['exp']}: {st.get('state')}  {counts}  of {st.get('n_jobs')}  running={running}")
    for n, j in jobs.items():
        if j["state"] == "failed":
            print(f"    FAILED {n} (rc={j.get('returncode')}) -> gpubench-vast logs --job {st['exp']}/{n}")


def _wait_exp(rem: Remote, exp: str, poll_s: int = 60, deadline: float | None = None) -> dict:
    last = None
    while True:
        try:
            st = _exp_status(rem, exp)
        except Exception as e:
            print(f"  (status poll failed: {e}; retrying)")
            st = None
        if st:
            key = json.dumps({k: v["state"] for k, v in st["jobs"].items()}) + st["state"]
            if key != last:
                _print_status(st)
                last = key
            if st["state"] in ("done", "failed"):
                return st
        if deadline and time.time() > deadline:
            raise TimeoutError("max-hours reached")
        time.sleep(poll_s)


def cmd_status(a):
    _, _, rem = _remote(a.id)
    exps = [a.exp] if a.exp else [Path(p).stem.replace(".status", "") for p in
                                  rem.text("ls /workspace/jobs/*.status.json 2>/dev/null", check=False, cwd=None).split()]
    for e in exps:
        st = _exp_status(rem, e)
        _print_status(st) if st else print(f"  {e}: no status")
    print(rem.text("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader",
                   check=False, cwd=None).strip())
    return 0


def cmd_logs(a):
    inst_id, _, rem = _remote(a.id)
    if a.bootstrap:
        path = "/workspace/bootstrap.log"
    elif a.job:
        path = f"/workspace/logs/{a.job}.log"
    else:
        print(I.vast_client().logs(instance_id=inst_id, tail=str(a.tail)))
        return 0
    print(rem.text(f"tail -n {a.tail} {shlex.quote(path)}", check=False, cwd=None))
    return 0


def _pull(inst_id: int, inst: dict, rem: Remote) -> Path:
    alias = _gpu_alias(inst_id, inst)
    dest = RESULTS / alias
    got = rem.pull_dir(f"{REMOTE_REPO}/results", dest)
    rem.pull_dir("/workspace/logs", dest / "_logs" / str(inst_id))
    for f in ("smoke.json", "bootstrap.log"):
        rem.pull_file(f"/workspace/{f}", dest / "_instance" / str(inst_id) / f)
    (dest / "_instance" / str(inst_id)).mkdir(parents=True, exist_ok=True)
    (dest / "_instance" / str(inst_id) / "vast_instance.json").write_text(json.dumps(_redact(inst), indent=2, default=str))
    print(f"pulled -> {dest}" + ("" if got else " (no results/ yet)"))
    return dest


def cmd_pull(a):
    inst_id, inst, rem = _remote(a.id)
    _pull(inst_id, inst, rem)
    return 0


def cmd_freeze(a):
    """Write requirements/remote.lock.txt = packages installed on top of the pinned base image."""
    _, _, rem = _remote(a.id)
    base = {l.strip().lower() for l in rem.text("cat /workspace/base_freeze.txt", cwd=None).splitlines() if "==" in l}
    now = [l.strip() for l in rem.text("python -m pip list --format=freeze --exclude-editable", cwd=None).splitlines() if "==" in l]
    skip = {"pip", "setuptools", "wheel", "gpubench"}
    lock = sorted((l for l in now if l.lower() not in base and l.split("==")[0].lower() not in skip), key=str.lower)
    header = ["# AUTO-GENERATED by `gpubench-vast freeze` from a working instance. Do not edit by hand.",
              f"# base image packages are excluded (pinned by the image); generated {time.strftime('%Y-%m-%d')}",
              "--extra-index-url https://pypi.nvidia.com",
              "--extra-index-url https://download.pytorch.org/whl/cu128"]
    out = I.ROOT / "requirements" / "remote.lock.txt"
    out.write_text("\n".join(header + lock) + "\n")
    print(f"wrote {out} ({len(lock)} packages). Commit it: every later instance installs exactly this set.")
    return 0


def cmd_down(a):
    inst_id = I.resolve_id(a.id)
    if not a.yes:
        if input(f"destroy instance {inst_id}? results not pulled are lost [y/N] ").strip().lower() != "y":
            return 1
    I.destroy(inst_id)
    return 0


def cmd_exp(a):
    """One-shot: up -> run each exp file sequentially -> pull -> destroy (always, unless --keep)."""
    for f in a.exp:
        expand(f)  # validate before paying for a GPU
    inst_id = None
    created: list = []
    code = 1
    deadline = time.time() + a.max_hours * 3600
    try:
        if a.instance:
            inst_id, inst, rem = _remote(a.instance)
        else:
            inst_id, inst, rem = _up_with_retry(a.gpu, a.offer, a.allow_dirty, created)
        sha = I.load_state().get(str(inst_id), {}).get("git_sha")
        failed = False
        for f in a.exp:
            exp = _start_exp(rem, f, sha)
            st = _wait_exp(rem, exp, deadline=deadline)
            failed |= st["state"] == "failed"
            _pull(inst_id, inst, rem)  # pull after every exp so nothing is lost if a later one hangs
        code = 1 if failed else 0
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        inst_id = inst_id if inst_id is not None else (created[-1] if created else None)
        if inst_id is not None:
            try:
                inst = I.show(inst_id)
                if inst and inst.get("actual_status") == "running":
                    _pull(inst_id, inst, Remote.from_instance(inst))
            except Exception as e:
                print(f"final pull failed: {e}")
            if a.keep or a.instance:
                print(f"keeping instance {inst_id} (remember `gpubench-vast down {inst_id}`)")
            else:
                I.destroy(inst_id)
    return code


def cmd_smoke(a):
    """Boot + bootstrap + smoke test on a GPU class, pull the report, destroy. (~15-25 min)"""
    inst_id = None
    created: list = []
    try:
        inst_id, inst, rem = _up_with_retry(a.gpu, a.offer, a.allow_dirty, created)
        dest = _pull(inst_id, inst, rem)
        print((dest / "_instance" / str(inst_id) / "smoke.json").read_text())
        return 0
    except Exception as e:
        print(f"SMOKE FAILED on {a.gpu}: {e}")
        inst_id = inst_id if inst_id is not None else (created[-1] if created else None)
        if inst_id is not None:
            try:
                inst = I.show(inst_id)
                _pull(inst_id, inst, Remote.from_instance(inst))
            except Exception:
                pass
        return 1
    finally:
        inst_id = inst_id if inst_id is not None else (created[-1] if created else None)
        if inst_id is not None and not a.keep:
            I.destroy(inst_id)


# ----------------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="gpubench-vast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    add("doctor", cmd_doctor, "check local setup")
    sp = add("offers", cmd_offers, "list cheapest matching offers")
    sp.add_argument("gpu")
    sp.add_argument("-n", type=int, default=10)
    sp.add_argument("--cpu", help="regex on host CPU name, e.g. 'EPYC 9' (comparisons should pin the host CPU class)")
    for name, fn, h in [("up", cmd_up, "create instance + bootstrap"), ("smoke", cmd_smoke, "up + smoke test + pull + destroy")]:
        sp = add(name, fn, h)
        sp.add_argument("gpu")
        sp.add_argument("--offer", type=int)
        sp.add_argument("--cpu", help="regex on host CPU name (see `offers --cpu`)")
        sp.add_argument("--allow-dirty", action="store_true")
        if name == "up":
            sp.add_argument("--no-wait", action="store_true")
        else:
            sp.add_argument("--keep", action="store_true")
    add("ls", cmd_ls, "list your Vast instances (and their running cost)")
    add("rebootstrap", cmd_rebootstrap, "re-run infra/bootstrap.sh on the instance (after sync)").add_argument("id", nargs="?")
    add("wait", cmd_wait, "wait until the instance's bootstrap is done").add_argument("id", nargs="?")
    for name, fn, h in [("show", cmd_show, "instance details"), ("pull", cmd_pull, "download results/ + logs"),
                        ("freeze", cmd_freeze, "write requirements/remote.lock.txt from instance")]:
        add(name, fn, h).add_argument("id", nargs="?")
    sp = add("ssh", cmd_ssh, "interactive shell (or run a command)")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--cmd")
    sp = add("sync", cmd_sync, "checkout current pushed HEAD on the instance")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--allow-dirty", action="store_true")
    sp = add("run", cmd_run, "start experiment file(s) on an instance (detached)")
    sp.add_argument("exp", nargs="+")
    sp.add_argument("--id")
    sp = add("status", cmd_status, "job status + GPU utilisation")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--exp")
    sp = add("logs", cmd_logs, "container logs, bootstrap log, or a job log")
    sp.add_argument("id", nargs="?")
    sp.add_argument("--bootstrap", action="store_true")
    sp.add_argument("--job", help="<exp>/<job name>")
    sp.add_argument("--tail", type=int, default=100)
    sp = add("down", cmd_down, "destroy an instance")
    sp.add_argument("id", nargs="?")
    sp.add_argument("-y", "--yes", action="store_true")
    sp = add("exp", cmd_exp, "up -> run -> pull -> destroy")
    sp.add_argument("gpu")
    sp.add_argument("exp", nargs="+")
    sp.add_argument("--offer", type=int)
    sp.add_argument("--cpu", help="regex on host CPU name (see `offers --cpu`)")
    sp.add_argument("--instance", help="reuse a running instance instead of creating one (kept afterwards)")
    sp.add_argument("--keep", action="store_true", help="do not destroy at the end")
    sp.add_argument("--max-hours", type=float, default=12.0, help="watchdog: pull + destroy after this long")
    sp.add_argument("--allow-dirty", action="store_true")

    a = p.parse_args(argv)
    if getattr(a, "cpu", None):
        os.environ["GPUBENCH_CPU_REGEX"] = a.cpu
    os.chdir(I.ROOT)  # state dir / results are relative to the repo root
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
