"""SSH transport to a Vast instance using the system OpenSSH client (ships with Windows 10+, macOS, Linux).

We deliberately avoid `vastai copy` (needs rsync) and the SDK's `execute` (restricted command set).
Files move as tar streams over ssh stdout/stdin, which works identically on every OS.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

STATE_DIR = Path(".gpubench")
REMOTE_REPO = "/workspace/repo"


class Remote:
    def __init__(self, host: str, port: int, user: str = "root", key: str | None = None):
        self.host, self.port, self.user = host, int(port), user
        self.key = key or os.environ.get("GPUBENCH_SSH_KEY")

    @classmethod
    def from_instance(cls, inst: dict) -> "Remote":
        """Prefer the direct port mapping (faster transfers); fall back to Vast's ssh proxy."""
        ports = inst.get("ports") or {}
        direct = ports.get("22/tcp")
        if direct and inst.get("public_ipaddr"):
            return cls(inst["public_ipaddr"].strip(), int(direct[0]["HostPort"]))
        if inst.get("ssh_host") and inst.get("ssh_port"):
            return cls(inst["ssh_host"], int(inst["ssh_port"]))
        raise RuntimeError(f"instance {inst.get('id')} exposes no ssh endpoint yet")

    def base_cmd(self, tty: bool = False) -> list[str]:
        STATE_DIR.mkdir(exist_ok=True)
        cmd = ["ssh", "-p", str(self.port),
               "-o", "StrictHostKeyChecking=no",
               "-o", f"UserKnownHostsFile={(STATE_DIR / 'known_hosts').resolve()}",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
               "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR"]
        if self.key:
            cmd += ["-i", self.key]
        cmd += ["-t"] if tty else ["-T"]
        return cmd + [f"{self.user}@{self.host}"]

    @staticmethod
    def wrap(cmd: str, cwd: str | None = REMOTE_REPO) -> str:
        inner = "source /etc/gpubench.env 2>/dev/null; " + (f"cd {cwd} 2>/dev/null; " if cwd else "") + cmd
        return "bash -c " + shlex.quote(inner)

    def run(self, cmd: str, *, input: bytes | str | None = None, check: bool = True, capture: bool = True,
            timeout: float | None = 600, cwd: str | None = REMOTE_REPO) -> subprocess.CompletedProcess:
        if isinstance(input, str):
            input = input.encode()
        p = subprocess.run(self.base_cmd() + [self.wrap(cmd, cwd)], input=input, capture_output=capture, timeout=timeout)
        if check and p.returncode != 0:
            err = (p.stderr or b"").decode(errors="replace")[-2000:] if capture else ""
            raise RuntimeError(f"remote command failed ({p.returncode}): {cmd}\n{err}")
        return p

    def text(self, cmd: str, **kw) -> str:
        return self.run(cmd, **kw).stdout.decode(errors="replace")

    def ok(self, cmd: str, timeout: float = 60) -> bool:
        try:
            return self.run(cmd, check=False, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def detach(self, name: str, cmd: str, log: str, cwd: str = REMOTE_REPO):
        """Start `cmd` in a detached tmux session that survives the ssh connection closing.

        (`setsid nohup ... &` over a non-tty ssh session races with session teardown and can be killed
        before it even opens its log file; the tmux server is a proper daemon.)
        """
        inner = f"source /etc/gpubench.env 2>/dev/null; cd {cwd}; {cmd} > {shlex.quote(log)} 2>&1"
        self.run(f"mkdir -p {shlex.quote(os.path.dirname(log))} && tmux kill-session -t {name} 2>/dev/null; "
                 f"tmux new-session -d -s {name} {shlex.quote('bash -c ' + shlex.quote(inner))} && sleep 2 && "
                 f"(tmux has-session -t {name} 2>/dev/null || test -s {shlex.quote(log)})", cwd=None)

    def put_text(self, remote_path: str, content: str):
        d = os.path.dirname(remote_path)
        self.run(f"mkdir -p {shlex.quote(d)} && cat > {shlex.quote(remote_path)}", input=content, cwd=None)

    def pull_dir(self, remote_dir: str, local_dir: Path, timeout: float = 3600, exclude: tuple[str, ...] = ("*.pt",)) -> bool:
        """tar the remote dir over ssh and extract locally. Returns False if the remote dir doesn't exist.

        `exclude`: tar patterns left behind. Model checkpoints (*.pt) are skipped by default; in step 2 they
        were 1.6 GB of a 1.7 GB results dir and are not used by any analysis.
        """
        if not self.ok(f"test -d {shlex.quote(remote_dir)}"):
            return False
        local_dir.mkdir(parents=True, exist_ok=True)
        excl = " ".join(f"--exclude={shlex.quote(p)}" for p in exclude)
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
            tmp_path = tmp.name
            p = subprocess.run(self.base_cmd() + [self.wrap(f"tar -C {shlex.quote(remote_dir)} {excl} -czf - .", cwd=None)],
                               stdout=tmp, stderr=subprocess.PIPE, timeout=timeout)
        try:
            if p.returncode != 0:
                raise RuntimeError(f"tar over ssh failed: {p.stderr.decode(errors='replace')[-1000:]}")
            with tarfile.open(tmp_path, "r:gz") as tf:
                tf.extractall(local_dir, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(local_dir)
        finally:
            os.unlink(tmp_path)
        return True

    def pull_file(self, remote_path: str, local_path: Path) -> bool:
        p = self.run(f"cat {shlex.quote(remote_path)}", check=False, cwd=None)
        if p.returncode != 0:
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(p.stdout)
        return True

    def interactive(self, cmd: str | None = None) -> int:
        args = self.base_cmd(tty=True)
        if cmd:
            args.append(self.wrap(cmd))
        return subprocess.call(args)
