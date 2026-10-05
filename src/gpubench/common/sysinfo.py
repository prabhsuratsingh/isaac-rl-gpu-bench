"""Everything needed to reproduce / compare a run across GPUs, captured once per run."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata

PACKAGES = ["torch", "isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl-rl-lib", "numpy", "gymnasium", "triton"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def _pkg_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _git_sha() -> str | None:
    return os.environ.get("GIT_SHA") or _run(["git", "rev-parse", "HEAD"])


def _nvidia_smi() -> dict:
    q = "name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,pci.bus_id"
    line = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if not line:
        return {}
    first = line.splitlines()[0]
    keys = ["name", "driver_version", "memory_total_mb", "sm_clock_max_mhz", "mem_clock_max_mhz", "power_limit_w", "pci_bus_id"]
    return dict(zip(keys, [v.strip() for v in first.split(",")]))


def collect() -> dict:
    info: dict = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "packages": _pkg_versions(),
        "gpu": _nvidia_smi(),
        "cuda_driver_max": None,
        "vast": {
            "container_id": os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL"),
            "gpu_alias": os.environ.get("GPUBENCH_GPU"),
            "offer_id": os.environ.get("GPUBENCH_OFFER_ID"),
            "dph": float(os.environ["GPUBENCH_DPH"]) if os.environ.get("GPUBENCH_DPH") else None,
            "image": os.environ.get("GPUBENCH_IMAGE"),
        },
    }
    smi = _run(["nvidia-smi"])
    if smi and "CUDA Version:" in smi:
        info["cuda_driver_max"] = smi.split("CUDA Version:")[1].split()[0]
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"].update(
                {
                    "torch_name": p.name,
                    "capability": f"{p.major}.{p.minor}",
                    "sm_count": p.multi_processor_count,
                    "memory_total_mb_torch": round(p.total_memory / 2**20),
                }
            )
            info["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    except ImportError:
        pass
    return info
