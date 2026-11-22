"""The campaign scheduler, driven with a stand-in worker that only writes completion records."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from chapter3_pollinators.campaign import scheduler as scheduler_module
from chapter3_pollinators.campaign.plan import build_plan
from chapter3_pollinators.campaign.scheduler import Scheduler, SchedulerSettings
from chapter3_pollinators.campaign.worker import completion_marker


@pytest.fixture()
def quiet_machine(monkeypatch):
    monkeypatch.setattr(scheduler_module, "WORKER_MODULE", "chapter3_pollinators.tests.fake_worker")
    monkeypatch.setattr(scheduler_module, "other_jobs_running", lambda *a, **k: [])
    monkeypatch.setattr(scheduler_module, "other_gpu_jobs", lambda *a, **k: [])
    monkeypatch.setattr(scheduler_module, "available_ram_gb", lambda: 64.0)
    monkeypatch.setattr(scheduler_module, "gpu_memory_gb", lambda: (1.0, 32.0))
    # the machine running the tests may be busy: the spare-CPU rule is tested on its own
    monkeypatch.setattr(scheduler_module, "logical_cpus", lambda: 32)
    monkeypatch.setattr(scheduler_module.Scheduler, "other_cpu_load", lambda self: 0.0)


def _small_plan():
    plan = build_plan(seeds=[0], group3_seeds=[0])
    keep = {"HT1", "HM1"}
    return [t for t in plan if t.community in keep]


def _settings(root, **kwargs):
    defaults = dict(root=root, busy_workers=2, free_workers=4, poll_seconds=0.05, max_retries=1,
                    free_after_seconds=0.0, cpu_margin=0)
    defaults.update(kwargs)
    return SchedulerSettings(**defaults)


def _launch_order(root) -> list[str]:
    """Task ids in the order the stand-in workers started."""
    folder = root / "order"
    if not folder.is_dir():
        return []
    entries = [f.read_text().split() for f in folder.glob("*.txt")]
    return [task for _, task in sorted(entries, key=lambda e: int(e[0]))]


def _run(scheduler, max_ticks=4000):
    peak = 0
    for _ in range(max_ticks):
        alive = scheduler.tick()
        peak = max(peak, len(scheduler.running))
        if not alive:
            return peak
        time.sleep(0.05)
    raise AssertionError("scheduler did not finish")


def test_runs_every_task_after_its_warmup(tmp_path, quiet_machine) -> None:
    plan = _small_plan()
    scheduler = Scheduler(_settings(tmp_path), plan)
    peak = _run(scheduler)
    assert peak <= 4
    for task in plan:
        assert completion_marker(tmp_path, task).is_file(), task.task_id
    order = _launch_order(tmp_path)
    for task in plan:
        if task.dependency:
            assert order.index(task.dependency) < order.index(task.task_id)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["done"] == progress["total"] == len(plan)
    assert (tmp_path / "PROGRESS.md").is_file()


def test_resume_skips_finished_tasks(tmp_path, quiet_machine) -> None:
    plan = _small_plan()
    finished = [t for t in plan if t.community == "HT1"]
    for task in finished:
        marker = completion_marker(tmp_path, task)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}")
    _run(Scheduler(_settings(tmp_path), plan))
    launched = set(_launch_order(tmp_path))
    assert not launched & {t.task_id for t in finished}
    assert launched == {t.task_id for t in plan if t.community == "HM1"}


def test_failures_are_retried_then_block_their_dependents(tmp_path, quiet_machine) -> None:
    plan = _small_plan()
    (tmp_path / "fail.txt").write_text("warmup:HM1:s0")
    scheduler = Scheduler(_settings(tmp_path), plan)
    _run(scheduler)
    assert scheduler.failed == {"warmup:HM1:s0"}
    assert scheduler.failures["warmup:HM1:s0"] == 2
    for task in plan:
        assert completion_marker(tmp_path, task).is_file() == (task.community == "HT1")


def test_busy_machine_uses_the_reduced_worker_count(tmp_path, quiet_machine, monkeypatch) -> None:
    monkeypatch.setattr(scheduler_module, "other_jobs_running", lambda *a, **k: [1234])
    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    assert scheduler.capacity() == (2, "busy")
    assert _run(scheduler) <= 2
    (tmp_path / "control.json").write_text(json.dumps({"busy_workers": 1}))
    assert scheduler.capacity() == (1, "busy")
    (tmp_path / "control.json").write_text(json.dumps({"paused": True}))
    assert scheduler.capacity() == (0, "paused")


def test_low_memory_holds_new_launches(tmp_path, quiet_machine, monkeypatch) -> None:
    monkeypatch.setattr(scheduler_module, "available_ram_gb", lambda: 3.0)
    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    scheduler.tick()
    assert not scheduler.running
    assert "holding" in json.loads((tmp_path / "progress.json").read_text())["note"]


def test_stop_file_prevents_new_launches(tmp_path, quiet_machine) -> None:
    (tmp_path / "STOP").write_text("stop")
    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    assert scheduler.tick() is False
    assert not _launch_order(tmp_path)


def test_live_workers_are_adopted_not_restarted(tmp_path, quiet_machine) -> None:
    plan = _small_plan()
    task_id = "warmup:HT1:s0"
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "--task", task_id]
    )
    try:
        state = {"running": {task_id: {"pid": sleeper.pid, "started": time.time()}}}
        (tmp_path / "scheduler_state.json").write_text(json.dumps(state))
        scheduler = Scheduler(_settings(tmp_path), plan)
        assert set(scheduler.running) == {task_id}
        scheduler.tick()
        assert task_id not in _launch_order(tmp_path)
    finally:
        sleeper.kill()


def test_free_mode_waits_until_the_machine_has_been_quiet(tmp_path, quiet_machine, monkeypatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(scheduler_module.time, "time", lambda: clock["now"])
    others = {"pids": [42]}
    monkeypatch.setattr(scheduler_module, "other_jobs_running", lambda *a, **k: list(others["pids"]))
    scheduler = Scheduler(_settings(tmp_path, free_after_seconds=900), _small_plan())
    assert scheduler.capacity() == (2, "busy")
    others["pids"] = []
    clock["now"] += 300
    assert scheduler.capacity() == (2, "busy")
    clock["now"] += 601
    assert scheduler.capacity() == (4, "free")
    others["pids"] = [7]
    assert scheduler.capacity() == (2, "busy")


def test_free_capacity_leaves_room_for_other_load(tmp_path, quiet_machine, monkeypatch) -> None:
    monkeypatch.setattr(scheduler_module, "logical_cpus", lambda: 32)
    load = {"other": 16.0}
    scheduler = Scheduler(_settings(tmp_path, free_workers=24, busy_workers=6, cpu_margin=4), _small_plan())
    monkeypatch.setattr(scheduler, "other_cpu_load", lambda: load["other"])
    assert scheduler.capacity() == (12, "free")
    load["other"] = 0.5
    assert scheduler.capacity() == (24, "free")
    load["other"] = 30.0
    assert scheduler.capacity() == (6, "free")


def test_only_heavy_jobs_force_the_reduced_count(tmp_path, quiet_machine, monkeypatch) -> None:
    calls = []

    def fake_other_jobs(patterns, exclude=()):
        calls.append(tuple(patterns))
        return []

    monkeypatch.setattr(scheduler_module, "other_jobs_running", fake_other_jobs)
    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    monkeypatch.setattr(scheduler, "other_cpu_load", lambda: 0.0)
    scheduler.capacity()
    assert calls and all("train_with_geomaca" in p and "chapter4_qolu.run_" in p for p in calls)
    assert all("chapter4_qolu" not in p for p in calls)


def test_update_heavy_tasks_have_their_own_cap(tmp_path, quiet_machine) -> None:
    plan = _small_plan()
    finished = [t for t in plan if t.kind == "warmup"]
    for task in finished:
        marker = completion_marker(tmp_path, task)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}")
    scheduler = Scheduler(_settings(tmp_path, free_workers=12, gpu_heavy_workers=2), plan)
    scheduler.tick()
    kinds = [scheduler.by_id[t].run_type for t in scheduler.running]
    heavy = [k for k in kinds if not k.startswith("oraa")]
    assert len(heavy) <= 2
    assert len(kinds) == min(12, 2 + sum(t.run_type.startswith("oraa") for t in plan))
    _run(scheduler)
    assert all(completion_marker(tmp_path, t).is_file() for t in plan)


def test_control_file_adds_other_job_patterns(tmp_path, quiet_machine, monkeypatch) -> None:
    calls = []

    def fake_other_jobs(patterns, exclude=()):
        calls.append(tuple(patterns))
        return [99] if "some_other_study.train" in patterns else []

    monkeypatch.setattr(scheduler_module, "other_jobs_running", fake_other_jobs)
    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    monkeypatch.setattr(scheduler, "other_cpu_load", lambda: 0.0)
    assert scheduler.capacity() == (4, "free")
    (tmp_path / "control.json").write_text(json.dumps({"other_job_patterns": ["some_other_study.train"]}))
    assert scheduler.capacity() == (2, "busy")
    assert "train_with_geomaca" in calls[-1] and "some_other_study.train" in calls[-1]


def test_workers_run_below_normal_priority(tmp_path, quiet_machine) -> None:
    import psutil

    plan = [t for t in _small_plan() if t.kind == "warmup"][:1]
    _run(Scheduler(_settings(tmp_path), plan))
    recorded = (tmp_path / "priority" / (plan[0].task_id.replace(":", "_") + ".txt")).read_text().split()
    if sys.platform == "win32":
        assert int(recorded[0]) == int(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        assert int(recorded[0]) >= 10


def test_gpu_reserve_and_recent_launches_hold_new_launches(tmp_path, quiet_machine, monkeypatch) -> None:
    gpu = {"used": 20.0}
    monkeypatch.setattr(scheduler_module, "gpu_memory_gb", lambda: (gpu["used"], 32.0))
    scheduler = Scheduler(_settings(tmp_path, gpu_reserve_gb=6.0, gpu_per_worker_gb=1.5), _small_plan())
    assert scheduler.can_launch_more()[0]
    gpu["used"] = 25.0
    assert not scheduler.can_launch_more()[0]
    (tmp_path / "control.json").write_text(json.dumps({"gpu_reserve_gb": 2.0}))
    assert scheduler.can_launch_more()[0]
    now = time.time()
    for i in range(3):
        scheduler.running[f"t{i}"] = scheduler_module.Running(f"t{i}", 0, now)
    assert not scheduler.can_launch_more()[0]
    for i in range(3):
        scheduler.running[f"t{i}"] = scheduler_module.Running(f"t{i}", 0, now - 3600)
    assert scheduler.can_launch_more()[0]


def test_python_processes_on_the_gpu_count_as_other_jobs(tmp_path, quiet_machine, monkeypatch) -> None:
    gpu_jobs = {"pids": []}
    monkeypatch.setattr(scheduler_module, "other_gpu_jobs", lambda *a, **k: list(gpu_jobs["pids"]))
    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    monkeypatch.setattr(scheduler, "other_cpu_load", lambda: 0.0)
    assert scheduler.capacity() == (4, "free")
    gpu_jobs["pids"] = [321]
    assert scheduler.capacity() == (2, "busy")
    (tmp_path / "control.json").write_text(json.dumps({"gpu_jobs_are_heavy": False}))
    assert scheduler.capacity() == (4, "free")


def test_progress_files_that_stay_busy_do_not_stop_the_scheduler(tmp_path, quiet_machine, monkeypatch) -> None:
    from chapter3_pollinators.experiment import atomic

    def always_busy(src, dst, retry_seconds=atomic.RETRY_SECONDS):
        raise PermissionError(13, "in use")

    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    monkeypatch.setattr(atomic, "replace_file", always_busy)
    assert scheduler.tick() is True
    assert "could not write" in (tmp_path / "scheduler.log").read_text()


def test_an_error_in_one_round_does_not_end_the_run(tmp_path, quiet_machine, monkeypatch) -> None:
    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    rounds = iter([RuntimeError("transient"), False])

    def tick():
        outcome = next(rounds)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(scheduler, "tick", tick)
    scheduler.run()
    log = (tmp_path / "scheduler.log").read_text()
    assert "error in scheduling round" in log and "transient" in log and "scheduler finished" in log


def test_the_jobs_that_start_busy_mode_are_logged_once(tmp_path, quiet_machine, monkeypatch) -> None:
    import os

    monkeypatch.setattr(scheduler_module, "other_jobs_running", lambda *a, **k: [os.getpid()])
    scheduler = Scheduler(_settings(tmp_path), _small_plan())
    assert scheduler.capacity() == (2, "busy")
    assert scheduler.capacity() == (2, "busy")
    lines = [line for line in (tmp_path / "scheduler.log").read_text().splitlines() if "other jobs" in line]
    assert len(lines) == 1 and str(os.getpid()) in lines[0] and "python" in lines[0].lower()


def test_a_new_scheduler_is_free_at_once_when_no_other_job_runs(tmp_path, quiet_machine) -> None:
    scheduler = Scheduler(_settings(tmp_path, free_after_seconds=900), _small_plan())
    assert scheduler.capacity() == (4, "free")


def test_bound_branches_are_oraa_tasks() -> None:
    from chapter3_pollinators.campaign.plan import parse_task_id
    from chapter3_pollinators.campaign.scheduler import is_update_heavy

    assert not is_update_heavy(parse_task_id("branch:HT1:HTC:oraa_mb_b3:s0"))
    assert not is_update_heavy(parse_task_id("branch:HT1:HTC:oraa_omni_b3:s0"))
    assert is_update_heavy(parse_task_id("branch:HT1:HTC:reward_shape:s0"))


def _group_three_and_four(tmp_path):
    """HM1 (group 3) still needs its warm-up; HT1's group-4 branches are ready."""
    plan = [t for t in build_plan(seeds=[0], group3_seeds=[0]) if t.community in {"HT1", "HM1"}]
    for task in plan:
        if task.priority <= 2 or (task.kind == "warmup" and task.community == "HT1"):
            marker = completion_marker(tmp_path, task)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}")
    return plan


def test_lower_priority_waits_while_higher_priority_work_of_its_class_waits(tmp_path, quiet_machine) -> None:
    plan = _group_three_and_four(tmp_path)
    scheduler = Scheduler(_settings(tmp_path, free_workers=12, busy_workers=2), plan)
    scheduler.tick()
    assert set(scheduler.running) == {"warmup:HM1:s0"}
    _run(scheduler)
    launched = [line.split("launched ")[1].split(" (pid")[0]
                for line in (tmp_path / "scheduler.log").read_text().splitlines() if " launched " in line]
    group3_oraa = [launched.index(t) for t in launched if ":oraa_mb:" in t and ":HM1:" in t]
    group4 = [launched.index(t) for t in launched if t.endswith("_b3:s0")]
    assert len(group3_oraa) == 2 and len(group4) == 2
    assert min(group4) > max(group3_oraa)
    assert all(completion_marker(tmp_path, t).is_file() for t in plan)


def test_lower_priority_may_fill_free_places_when_allowed(tmp_path, quiet_machine) -> None:
    plan = _group_three_and_four(tmp_path)
    (tmp_path / "control.json").write_text(json.dumps({"hold_lower_priority": False}))
    scheduler = Scheduler(_settings(tmp_path, free_workers=12, busy_workers=2), plan)
    scheduler.tick()
    assert {"warmup:HM1:s0", "branch:HT1:HTC:oraa_mb_b3:s0", "branch:HT1:HTC:oraa_omni_b3:s0"} <= set(
        scheduler.running)
