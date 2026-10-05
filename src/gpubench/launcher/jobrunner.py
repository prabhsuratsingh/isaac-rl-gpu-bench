"""Runs on the GPU instance: execute a job list sequentially, detached from ssh.

    setsid nohup python -m gpubench.launcher.jobrunner /workspace/jobs/<exp>.json &

State (read by `gpubench-vast status`):
    /workspace/jobs/<exp>.status.json   {"state": running|done|failed, "jobs": {name: {...}}}
    /workspace/logs/<exp>/<job>.log     stdout+stderr of each job
Resumable: jobs already marked ok are skipped when the runner is restarted (e.g. after a host reboot).
Jobs run one at a time on purpose: concurrent runs on one GPU would corrupt every throughput number.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WS = Path(os.environ.get("GPUBENCH_WS", "/workspace"))
REPO = WS / "repo"


def _write(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def main(job_file: str) -> int:
    job_file = Path(job_file)
    spec = json.loads(job_file.read_text())
    exp = spec["exp"]
    status_path = job_file.with_suffix(".status.json")
    log_dir = WS / "logs" / exp
    log_dir.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {"exp": exp, "jobs": {}}
    status.update({"state": "running", "pid": os.getpid(), "started": status.get("started") or time.time(),
                   "git_sha": spec.get("git_sha"), "n_jobs": len(spec["jobs"])})
    _write(status_path, status)

    failed_hard = False
    for job in spec["jobs"]:
        name = job["name"]
        prev = status["jobs"].get(name, {})
        if prev.get("state") == "ok":
            continue
        rec = {"state": "running", "cmd": job["cmd"], "start": time.time()}
        status["jobs"][name] = rec
        _write(status_path, status)
        with open(log_dir / f"{name}.log", "w") as log:
            log.write(f"$ {job['cmd']}\n")
            log.flush()
            proc = subprocess.Popen(["bash", "-c", f"source /etc/gpubench.env 2>/dev/null; {job['cmd']}"], cwd=REPO,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=job["timeout_s"])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)  # Kit spawns children; kill the whole group
                proc.wait()
                rc = "timeout"
        rec.update({"end": time.time(), "returncode": rc, "state": "ok" if rc == 0 else "failed"})
        rec["duration_s"] = round(rec["end"] - rec["start"], 1)
        _write(status_path, status)
        if rc != 0 and not job.get("allow_fail"):
            failed_hard = True
            if spec.get("stop_on_fail"):
                break

    status["state"] = "failed" if failed_hard else "done"
    status["finished"] = time.time()
    _write(status_path, status)
    return 1 if failed_hard else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
