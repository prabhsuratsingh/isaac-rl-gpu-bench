"""Vast.ai instance lifecycle (search -> create -> wait -> destroy) and local instance bookkeeping."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from .remote import STATE_DIR, Remote

ROOT = Path(__file__).resolve().parents[3]
GPUS_FILE = ROOT / "infra" / "gpus.yaml"
STATE_FILE = STATE_DIR / "instances.json"

ONSTART = r"""#!/bin/bash
# gpubench onstart: clone the repo at the pinned commit, then bootstrap in the background.
touch /root/.no_auto_tmux
mkdir -p /workspace && cd /workspace
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ca-certificates >/dev/null)
if [ ! -d repo/.git ]; then
  git clone -q "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" repo || { echo "clone failed" > /workspace/.bootstrap_failed; exit 1; }
  git -C repo remote set-url origin "https://github.com/${GH_REPO}.git"
fi
cd repo && git checkout -q "$GIT_SHA" || { echo "checkout $GIT_SHA failed" > /workspace/.bootstrap_failed; exit 1; }
nohup bash infra/bootstrap.sh > /workspace/bootstrap.log 2>&1 &
"""


# ----------------------------------------------------------------------------------------------------
# local environment / secrets
# ----------------------------------------------------------------------------------------------------
def load_dotenv(path: Path = ROOT / ".env"):
    """Minimal .env support (KEY=VALUE lines) so secrets never need to be typed on the command line."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def vast_client():
    from vastai import VastAI

    load_dotenv()
    key = os.environ.get("VAST_API_KEY")
    return VastAI(api_key=key, raw=True, quiet=True) if key else VastAI(raw=True, quiet=True)


def gpu_catalog() -> dict:
    return yaml.safe_load(GPUS_FILE.read_text())


def gpu_spec(alias: str) -> dict:
    cat = gpu_catalog()
    if alias not in cat["gpus"]:
        raise SystemExit(f"unknown GPU alias {alias!r}; known: {sorted(cat['gpus'])} (edit infra/gpus.yaml)")
    d = cat["defaults"]
    g = cat["gpus"][alias]
    return {
        "alias": alias,
        "image": g.get("image", d["image"]),
        "image_digest": g.get("image_digest", d.get("image_digest") if "image" not in g else None),
        "disk_gb": g.get("disk_gb", d["disk_gb"]),
        "query": f"{d['base_query']} {g['query']}",
        "env": {**d.get("env", {}), **g.get("env", {})},
    }


def dockerhub_digest(image: str) -> str | None:
    """Current registry digest of a Docker Hub `repo:tag` (None if it cannot be determined)."""
    import json as _json
    import urllib.request

    repo, _, tag = image.partition(":")
    if "/" not in repo or "." in repo.split("/")[0]:  # only Docker Hub `org/repo:tag` is supported here
        return None
    try:
        tok_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        token = _json.load(urllib.request.urlopen(tok_url, timeout=20))["token"]
        req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag or 'latest'}", method="HEAD", headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json, "
                      "application/vnd.docker.distribution.manifest.v2+json"})
        return urllib.request.urlopen(req, timeout=20).headers.get("docker-content-digest")
    except Exception:
        return None


def check_image(spec: dict, allow_drift: bool = False):
    want = spec.get("image_digest")
    if not want:
        return
    got = dockerhub_digest(spec["image"])
    if got is None:
        print(f"warning: could not verify digest of {spec['image']} (registry unreachable); launching by tag")
    elif got != want and not allow_drift:
        raise SystemExit(f"{spec['image']} now resolves to {got}, not the pinned {want}.\n"
                         "The base image changed upstream: results would not be comparable across GPUs. Pin a new tag/digest "
                         "in infra/gpus.yaml deliberately (and re-run the lock), or pass --allow-image-drift.")


# ----------------------------------------------------------------------------------------------------
# git: every run is tied to a pushed commit
# ----------------------------------------------------------------------------------------------------
def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_state(allow_dirty: bool = False) -> dict:
    sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty and not allow_dirty:
        raise SystemExit("working tree has uncommitted changes; commit + push first (or --allow-dirty for debugging)")
    _git("fetch", "-q", "origin")
    if not _git("branch", "-r", "--contains", sha):
        raise SystemExit(f"commit {sha[:10]} is not on any remote branch; `git push` first (the instance clones it)")
    url = _git("remote", "get-url", "origin")
    repo = url.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    return {"sha": sha, "repo": repo, "dirty": bool(dirty)}


# ----------------------------------------------------------------------------------------------------
# local state
# ----------------------------------------------------------------------------------------------------
def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state: dict):
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def remember(inst_id: int, **info):
    s = load_state()
    s.setdefault(str(inst_id), {}).update(info)
    save_state(s)


def forget(inst_id: int):
    s = load_state()
    s.pop(str(inst_id), None)
    save_state(s)


def resolve_id(ident: str | int | None) -> int:
    s = load_state()
    if ident in (None, "last"):
        if not s:
            raise SystemExit("no known instances (.gpubench/instances.json is empty); pass an instance id")
        return int(max(s.items(), key=lambda kv: kv[1].get("created", 0))[0])
    if isinstance(ident, str) and not ident.isdigit():  # gpu alias -> most recent instance of that alias
        cands = [(k, v) for k, v in s.items() if v.get("gpu") == ident]
        if not cands:
            raise SystemExit(f"no known instance for {ident!r}")
        return int(max(cands, key=lambda kv: kv[1].get("created", 0))[0])
    return int(ident)


# ----------------------------------------------------------------------------------------------------
# Vast operations
# ----------------------------------------------------------------------------------------------------
def search(alias: str, limit: int = 10) -> list[dict]:
    spec = gpu_spec(alias)
    offers = vast_client().search_offers(query=spec["query"], order="dph", limit=limit * 3, storage=spec["disk_gb"])
    offers = sorted(offers or [], key=lambda o: o.get("dph_total", 1e9))
    return offers[:limit]


def create(alias: str, offer_id: int | None, git: dict, label: str | None = None) -> dict:
    load_dotenv()
    token = os.environ.get("GPUBENCH_GH_TOKEN")
    if not token:
        raise SystemExit("GPUBENCH_GH_TOKEN not set (read-only fine-grained PAT for the repo; put it in .env)")
    spec = gpu_spec(alias)
    check_image(spec, allow_drift=os.environ.get("GPUBENCH_ALLOW_IMAGE_DRIFT") == "1")
    if offer_id is None:
        offers = search(alias, limit=1)
        if not offers:
            raise SystemExit(f"no offers match infra/gpus.yaml:{alias}: {spec['query']}")
        offer = offers[0]
    else:
        offer = {"id": offer_id, "dph_total": None}
    env = {
        **spec["env"],
        "GH_TOKEN": token,
        "GH_REPO": git["repo"],
        "GIT_SHA": git["sha"],
        "GPUBENCH_GPU": alias,
        "GPUBENCH_OFFER_ID": str(offer["id"]),
        "GPUBENCH_IMAGE": spec["image"] + (f"@{spec['image_digest']}" if spec.get("image_digest") else ""),
    }
    if offer.get("dph_total"):
        env["GPUBENCH_DPH"] = f"{offer['dph_total']:.4f}"
    res = vast_client().create_instance(
        id=int(offer["id"]), image=spec["image"], disk=spec["disk_gb"], env=env, onstart_cmd=ONSTART,
        label=label or f"gpubench-{alias}", ssh=True, direct=True,
    )
    if not res or not res.get("success", True) or "new_contract" not in res:
        raise SystemExit(f"create_instance failed: {res}")
    inst_id = int(res["new_contract"])
    remember(inst_id, gpu=alias, offer_id=offer["id"], dph=offer.get("dph_total"), git_sha=git["sha"],
             image=spec["image"], created=time.time(), gpu_name=offer.get("gpu_name"))
    return {"id": inst_id, "offer": offer}


def show(inst_id: int) -> dict | None:
    r = vast_client().show_instance(id=inst_id)
    if isinstance(r, dict) and "instances" in r:  # some API versions wrap the object
        r = r["instances"]
    return r or None


def destroy(inst_id: int):
    vast_client().destroy_instance(id=inst_id)
    s = load_state().get(str(inst_id), {})
    if s.get("created") and s.get("dph"):
        hrs = (time.time() - s["created"]) / 3600
        print(f"destroyed {inst_id}: ~{hrs:.2f} h, ~${hrs * s['dph']:.2f} (excl. storage/bandwidth)")
    forget(inst_id)


def wait_running(inst_id: int, timeout_s: int = 1800, poll_s: int = 15) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        inst = show(inst_id) or {}
        status = inst.get("actual_status")
        msg = (inst.get("status_msg") or "").strip()[:120]
        if (status, msg) != last:
            print(f"  [{int(time.time() - t0):4d}s] status={status} {msg}")
            last = (status, msg)
        if status == "running" and (inst.get("ssh_host") or inst.get("public_ipaddr")):
            if inst.get("dph_total"):
                remember(inst_id, dph=inst["dph_total"])
            return inst
        if status in ("exited", "offline") or "error" in msg.lower():
            raise RuntimeError(f"instance {inst_id} failed to start: status={status} msg={msg}")
        time.sleep(poll_s)
    raise TimeoutError(f"instance {inst_id} not running after {timeout_s}s")


def wait_ready(inst: dict, timeout_s: int = 5400, poll_s: int = 30) -> Remote:
    """Wait for bootstrap.sh to finish (pip install of Isaac Sim + smoke test) -> /workspace/.ready."""
    rem = Remote.from_instance(inst)
    t0 = time.time()
    last_line = ""
    while time.time() - t0 < timeout_s:
        try:
            p = rem.run("if [ -f /workspace/.ready ]; then echo READY; elif [ -f /workspace/.bootstrap_failed ]; then "
                        "echo FAILED; cat /workspace/.bootstrap_failed; fi; tail -n 1 /workspace/bootstrap.log 2>/dev/null",
                        check=False, timeout=60, cwd=None)
            out = p.stdout.decode(errors="replace").strip()
            if p.returncode == 255:  # ssh itself failed (sshd not up yet, or wrong key)
                err = p.stderr.decode(errors="replace").strip().splitlines()
                out = f"(ssh failed: {err[-1] if err else 'no output'} -- if this persists, check GPUBENCH_SSH_KEY / `gpubench-vast doctor`)"
        except Exception as e:
            out = f"(ssh not ready: {type(e).__name__})"
        if out.startswith("READY"):
            print(f"  [{int(time.time() - t0):4d}s] bootstrap complete")
            return rem
        if out.startswith("FAILED"):
            raise RuntimeError(f"bootstrap failed: {out}\n-> `gpubench-vast logs {inst.get('id')} --bootstrap`")
        line = out.splitlines()[-1] if out else ""
        if line and line != last_line:
            print(f"  [{int(time.time() - t0):4d}s] {line[:150]}")
            last_line = line
        time.sleep(poll_s)
    raise TimeoutError(f"bootstrap not finished after {timeout_s}s")
