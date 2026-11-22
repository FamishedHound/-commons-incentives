"""Machine probes and the worker entry point."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from chapter3_pollinators.campaign import system
from chapter3_pollinators.campaign.plan import parse_task_id
from chapter3_pollinators.campaign.worker import (
    _heartbeat_writer,
    completion_marker,
    is_done,
    main as worker_main,
    status_path,
)


def test_other_jobs_are_found_by_command_line() -> None:
    marker = "chapter3_probe_marker_7781"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", marker])
    try:
        deadline = time.time() + 10
        found = []
        while time.time() < deadline and not found:
            found = system.other_jobs_running((marker,))
            time.sleep(0.2)
        assert child.pid in found
        assert child.pid not in system.other_jobs_running((marker,), exclude=(marker,))
        assert system.pid_alive(child.pid)
        assert system.pid_alive(child.pid, marker=marker)
        assert not system.pid_alive(child.pid, marker="not-in-the-command-line")
    finally:
        child.kill()
        child.wait()
    assert not system.pid_alive(child.pid)


def test_memory_probes_return_plausible_values() -> None:
    assert system.available_ram_gb() > 0
    gpu = system.gpu_memory_gb()
    assert gpu is None or (0 <= gpu[0] <= gpu[1])


def test_worker_skips_a_finished_task(tmp_path, capsys) -> None:
    task = parse_task_id("branch:HT1:HTC:ltos:s0")
    marker = completion_marker(tmp_path, task)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert is_done(tmp_path, task)
    assert worker_main(["--root", str(tmp_path), "--task", task.task_id]) == 0
    assert "already done" in capsys.readouterr().out


def test_worker_reports_a_failing_task(tmp_path) -> None:
    # no warm-up checkpoint exists, so the branch cannot start
    assert worker_main(["--root", str(tmp_path), "--task", "branch:HT1:HTC:oraa_mb:s0"]) == 1
    assert status_path(tmp_path, "branch:HT1:HTC:oraa_mb:s0").is_file()


def test_heartbeat_files_are_written_atomically(tmp_path) -> None:
    beat = _heartbeat_writer(tmp_path, "warmup:HT1:s3", started=123.0)
    beat({"step": 50, "counter": 0})
    payload = json.loads(status_path(tmp_path, "warmup:HT1:s3").read_text())
    assert payload["step"] == 50 and payload["task_id"] == "warmup:HT1:s3" and payload["started"] == 123.0


def test_compute_app_listing_is_parsed() -> None:
    lines = (
        "123, C:/Python313/python.exe, [N/A]",
        "456, C:/Windows/explorer.exe, [N/A]",
        "789, [Insufficient Permissions], [N/A]",
        " 1011 , /usr/bin/python3, 1200 MiB",
        "garbage line",
    )
    text = "\n".join(lines)
    assert system.parse_compute_apps(text) == [123, 1011]


def test_other_gpu_jobs_are_filtered_by_command_line(monkeypatch) -> None:
    marker = "chapter3_gpu_probe_marker_5531"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", marker])
    try:
        monkeypatch.setattr(system, "gpu_python_processes", lambda: [child.pid, 2**31 - 7])
        deadline = time.time() + 10
        while time.time() < deadline and marker not in " ".join(system.psutil.Process(child.pid).cmdline()):
            time.sleep(0.2)
        assert system.other_gpu_jobs() == [child.pid]
        assert system.other_gpu_jobs(exclude=(marker,)) == []
    finally:
        child.kill()
        child.wait()


def test_heartbeat_survives_a_file_that_stays_busy(tmp_path, monkeypatch, capsys) -> None:
    from chapter3_pollinators.experiment import atomic

    def always_busy(src, dst, retry_seconds=atomic.RETRY_SECONDS):
        raise PermissionError(13, "in use")

    beat = _heartbeat_writer(tmp_path, "warmup:HT1:s4", started=1.0)
    monkeypatch.setattr(atomic, "replace_file", always_busy)
    beat({"step": 50, "counter": 0})
    assert "heartbeat" in capsys.readouterr().err
