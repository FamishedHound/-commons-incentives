"""Machine probes used by the scheduler: other heavy jobs, other programs on the GPU,
free memory and free GPU memory."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

import psutil

#: Command-line fragments of the other heavy jobs on the machine (campaign
#: drivers and training processes); while any runs, the scheduler keeps to its
#: reduced worker count.  Lighter processes (tests, evaluations) are accounted
#: for through the CPU load instead.
DEFAULT_OTHER_JOBS = (
    "chapter4_qolu.run_",
    "chapter4_qolu.experiment",
    "train_with_geomaca",
    "parallel_train",
)


def logical_cpus() -> int:
    return psutil.cpu_count(logical=True) or 1


def other_jobs_running(patterns: Sequence[str] = DEFAULT_OTHER_JOBS, exclude: Sequence[str] = ()) -> list[int]:
    """PIDs of Python processes whose command line contains one of ``patterns``.

    Only Python interpreters count: shells that merely launched such a job, and
    may outlive it, are ignored.
    """
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (process.info["name"] or "").lower():
                continue
            cmdline = " ".join(process.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(p in cmdline for p in patterns) and not any(e in cmdline for e in exclude):
            found.append(process.info["pid"])
    return found


def parse_compute_apps(text: str) -> list[int]:
    """PIDs of the Python processes in ``nvidia-smi --query-compute-apps=pid,process_name`` output."""
    pids = []
    for line in text.splitlines():
        pid, _, name = line.partition(",")
        if "python" not in name.lower():
            continue
        try:
            pids.append(int(pid.strip()))
        except ValueError:
            continue
    return pids


def gpu_python_processes() -> list[int]:
    """PIDs of the Python processes that hold a context on the GPU, or ``[]`` without ``nvidia-smi``."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_compute_apps(out)


def other_gpu_jobs(exclude: Sequence[str] = ()) -> list[int]:
    """Python processes on the GPU whose command line contains none of ``exclude``."""
    found = []
    for pid in gpu_python_processes():
        try:
            cmdline = " ".join(psutil.Process(pid).cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not any(e in cmdline for e in exclude):
            found.append(pid)
    return found


def describe_process(pid: int, width: int = 160) -> str:
    """``"<pid> <command line>"`` (shortened to ``width`` characters) for the logs."""
    try:
        cmdline = " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        cmdline = "(gone)"
    return f"{pid} {cmdline[:width]}"


def available_ram_gb() -> float:
    return psutil.virtual_memory().available / 2**30


def gpu_memory_gb() -> tuple[float, float] | None:
    """``(used, total)`` GPU memory in GB from ``nvidia-smi``, or ``None``."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
    except (subprocess.SubprocessError, IndexError, OSError):
        return None
    used, total = (float(x) / 1024 for x in out.split(","))
    return used, total


def pid_alive(pid: int, marker: str | None = None) -> bool:
    """True if ``pid`` runs and (when given) its command line contains ``marker``."""
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        if marker is None:
            return True
        return marker in " ".join(process.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
