"""The campaign scheduler: launches workers, tracks them, and writes the progress files.

The scheduler is resumable.  A task counts as done when its record is on disk,
so restarting the scheduler skips finished tasks; workers that are still
running from an earlier scheduler are adopted rather than started twice.

Worker count: ``busy_workers`` while another heavy job is running on the
machine (a process named by ``other_job_patterns``, or another program's
Python process on the GPU) and until none has been seen
for ``free_after_seconds``; after that as many as the spare CPUs allow (the
logical CPUs not used by other processes, less ``cpu_margin``), between
``busy_workers`` and ``free_workers``,
and never more than memory allows: a new worker needs ``ram_per_worker_gb`` of
free RAM beyond ``reserve_ram_gb`` and ``gpu_per_worker_gb`` of free GPU memory
beyond ``gpu_reserve_gb``, the GPU memory kept free for other work (workers
started in the last ``LAUNCH_SETTLE_SECONDS`` count as already holding theirs).
At most ``gpu_heavy_workers`` update-heavy tasks (warm-ups, runs without a
regulator, reward-shape and LToS branches) run at once: their many small GPU
kernels share one GPU, so more of them at once does not raise throughput;
ORAA branches, bound by the CPU, fill the remaining places.
Tasks start in priority order.  A task of a lower priority group waits while a
task of a higher one and of the same class (update-heavy or ORAA) is still
waiting to start, including tasks still waiting for their warm-up, so a later
group never takes the places an earlier group is about to need
(``hold_lower_priority``; off, later groups fill any free place).
Workers run at below-normal CPU priority, so other work on the machine keeps
precedence.
``control.json`` in the campaign folder can override ``busy_workers``,
``free_workers``, ``gpu_heavy_workers``, ``cpu_margin``, ``free_after_seconds``,
``gpu_reserve_gb``, ``gpu_jobs_are_heavy``, ``hold_lower_priority`` and
``paused``, and add ``other_job_patterns`` to the default ones, while the
scheduler runs.  Creating a file
named ``STOP`` in the folder stops new launches; the scheduler exits once the
running workers have finished.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from chapter3_pollinators.campaign.plan import (
    StepCosts,
    Task,
    dispatch_key,
    estimate_seconds,
    parse_task_id,
    ready_tasks,
)
import psutil

from chapter3_pollinators.campaign.system import (
    DEFAULT_OTHER_JOBS,
    available_ram_gb,
    describe_process,
    gpu_memory_gb,
    logical_cpus,
    other_gpu_jobs,
    other_jobs_running,
    pid_alive,
)
from chapter3_pollinators.campaign.worker import is_done, safe_name, status_path
from chapter3_pollinators.experiment import atomic

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "chapter3_pollinators.campaign.worker"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
#: Niceness of the workers where processes have a nice value (not Windows).
WORKER_NICENESS = 10
#: A worker started this recently may not yet show in the memory readings.
LAUNCH_SETTLE_SECONDS = 120.0


def worker_process_options() -> dict:
    """``subprocess.Popen`` options that start a worker without a window, below normal priority."""
    if sys.platform == "win32":
        return {"creationflags": CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(WORKER_NICENESS)}

#: Seconds per step used for the ETA until finished tasks give better numbers.
DEFAULT_COSTS = StepCosts(
    warmup=0.64, no_regulator=0.33, reward_shape=0.34, oraa_mb=6.5, oraa_omni=6.2, ltos=0.4
)


@dataclass
class SchedulerSettings:
    root: Path
    busy_workers: int = 6
    free_workers: int = 24
    poll_seconds: float = 20.0
    max_retries: int = 2
    reserve_ram_gb: float = 6.0
    ram_per_worker_gb: float = 2.0
    gpu_per_worker_gb: float = 1.3
    #: GPU memory left free for other work on the machine.
    gpu_reserve_gb: float = 6.0
    #: Whether other programs' Python processes on the GPU count as heavy jobs.
    gpu_jobs_are_heavy: bool = True
    #: Free mode starts only after no other job has been seen for this long.
    free_after_seconds: float = 900.0
    #: In free mode, logical CPUs left for everything else on the machine.
    cpu_margin: int = 4
    #: Most update-heavy tasks (warm-ups, runs without a regulator, reward-shape
    #: and LToS branches) at once: their many small GPU kernels share one GPU,
    #: while ORAA branches are bound by the CPU.
    gpu_heavy_workers: int = 6
    other_job_patterns: tuple[str, ...] = DEFAULT_OTHER_JOBS
    python: str = sys.executable


def is_update_heavy(task: Task) -> bool:
    """Tasks whose cost is the landowners' per-sample GPU updates rather than ORAA's search."""
    return not task.run_type.startswith("oraa")


def _held_back(task: Task, waiting: Counter) -> bool:
    """True while a task of a higher priority and of ``task``'s class is waiting to start."""
    heavy = is_update_heavy(task)
    return any(n > 0 and h == heavy and p < task.priority for (h, p), n in waiting.items())


@dataclass
class Running:
    task_id: str
    pid: int
    started: float
    process: subprocess.Popen | None = field(default=None, repr=False)


def load_plan(root: Path) -> list[Task]:
    payload = json.loads((Path(root) / "plan.json").read_text(encoding="utf-8"))
    return [parse_task_id(item["task_id"], item["priority"]) for item in payload["tasks"]]


def write_plan(root: Path, tasks: list[Task], meta: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "tasks": [{"task_id": t.task_id, "priority": t.priority} for t in tasks],
    }
    (root / "plan.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


class Scheduler:
    def __init__(self, settings: SchedulerSettings, plan: list[Task]) -> None:
        self.settings = settings
        self.root = Path(settings.root)
        self.plan = plan
        self.by_id = {t.task_id: t for t in plan}
        self.running: dict[str, Running] = {}
        self.failures: Counter = Counter()
        self.failed: set[str] = set()
        self.durations: dict[str, list[float]] = {}
        self.log_path = self.root / "scheduler.log"
        self._beats: dict[str, tuple[float, float]] = {}
        self._last_busy = float("-inf")  # no other job seen yet
        self._busy_pids: set[int] = set()
        self._cpu_handles: dict[int, psutil.Process] = {}
        psutil.cpu_percent(None)
        self._rates: dict[str, float] = {}
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self._load_state()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.root / "scheduler_state.json"

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.failures.update(state.get("failures", {}))
        self.failed.update(state.get("failed", []))
        self.durations = {k: list(v) for k, v in state.get("durations", {}).items()}
        for task_id, entry in state.get("running", {}).items():
            task = self.by_id.get(task_id)
            if task is None or is_done(self.root, task):
                continue
            if pid_alive(entry["pid"], marker=task_id):
                self.running[task_id] = Running(task_id, entry["pid"], entry["started"])
                self.log(f"adopted running worker {task_id} (pid {entry['pid']})")

    def _save_state(self) -> None:
        state = {
            "running": {k: {"pid": r.pid, "started": r.started} for k, r in self.running.items()},
            "failures": dict(self.failures),
            "failed": sorted(self.failed),
            "durations": self.durations,
            "updated": time.time(),
        }
        self._write_file(self.state_path, json.dumps(state, indent=1))

    def _write_file(self, path: Path, text: str) -> None:
        """Write ``text`` to ``path`` atomically; a target that stays busy is logged and skipped."""
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            atomic.replace_file(tmp, path)
        except OSError as exc:
            self.log(f"could not write {path.name}: {exc}")

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp}  {message}\n")

    # ------------------------------------------------------------------
    # capacity
    # ------------------------------------------------------------------
    def control(self) -> dict:
        path = self.root / "control.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def capacity(self) -> tuple[int, str]:
        control = self.control()
        if control.get("paused"):
            return 0, "paused"
        patterns = tuple(self.settings.other_job_patterns) + tuple(control.get("other_job_patterns", ()))
        others = other_jobs_running(patterns, exclude=("chapter3_pollinators",))
        if not others and control.get("gpu_jobs_are_heavy", self.settings.gpu_jobs_are_heavy):
            others = other_gpu_jobs(exclude=("chapter3_pollinators",))
        now = time.time()
        if others:
            self._last_busy = now
            if set(others) != self._busy_pids:
                self.log("other jobs: " + "; ".join(describe_process(pid) for pid in others[:3]))
        self._busy_pids = set(others)
        hold = float(control.get("free_after_seconds", self.settings.free_after_seconds))
        busy = int(control.get("busy_workers", self.settings.busy_workers))
        if others or now - self._last_busy < hold:
            return busy, "busy"
        free = int(control.get("free_workers", self.settings.free_workers))
        margin = int(control.get("cpu_margin", self.settings.cpu_margin))
        room = int(logical_cpus() - margin - self.other_cpu_load())
        return max(busy, min(free, room)), "free"

    def _worker_processes(self) -> list[psutil.Process]:
        found = []
        for entry in self.running.values():
            try:
                launcher = self._cpu_handles.setdefault(entry.pid, psutil.Process(entry.pid))
                found.append(launcher)
                for child in launcher.children(recursive=True):
                    found.append(self._cpu_handles.setdefault(child.pid, child))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return found

    def other_cpu_load(self) -> float:
        """Logical CPUs in use by everything except this campaign's workers."""
        total = psutil.cpu_percent(None) / 100.0 * logical_cpus()
        mine = 0.0
        for process in self._worker_processes():
            try:
                mine += process.cpu_percent(None) / 100.0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return max(total - mine, 0.0)

    def can_launch_more(self) -> tuple[bool, str]:
        now = time.time()
        settling = sum(1 for e in self.running.values() if now - e.started < LAUNCH_SETTLE_SECONDS)
        ram = available_ram_gb() - settling * self.settings.ram_per_worker_gb
        if ram < self.settings.reserve_ram_gb + self.settings.ram_per_worker_gb:
            return False, f"free RAM {ram:.1f} GB"
        gpu = gpu_memory_gb()
        if gpu is not None:
            reserve = float(self.control().get("gpu_reserve_gb", self.settings.gpu_reserve_gb))
            free = gpu[1] - gpu[0] - settling * self.settings.gpu_per_worker_gb
            if free < reserve + self.settings.gpu_per_worker_gb:
                return False, f"free GPU memory {free:.1f} GB (reserve {reserve:.1f} GB)"
        return True, ""

    # ------------------------------------------------------------------
    # workers
    # ------------------------------------------------------------------
    def launch(self, task: Task) -> None:
        log_file = (self.root / "logs" / f"{safe_name(task.task_id)}.log").open("a", encoding="utf-8")
        env = dict(os.environ)
        env.update(
            PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        process = subprocess.Popen(
            [self.settings.python, "-m", WORKER_MODULE, "--root", str(self.root), "--task", task.task_id],
            cwd=str(PACKAGE_PARENT), stdout=log_file, stderr=subprocess.STDOUT, env=env,
            **worker_process_options(),
        )
        log_file.close()
        self.running[task.task_id] = Running(task.task_id, process.pid, time.time(), process)
        self.log(f"launched {task.task_id} (pid {process.pid})")

    def _finished(self, entry: Running) -> tuple[bool, int | None]:
        if entry.process is not None:
            code = entry.process.poll()
            return code is not None, code
        alive = pid_alive(entry.pid, marker=entry.task_id)
        return not alive, None

    def reap(self) -> None:
        for task_id in list(self.running):
            entry = self.running[task_id]
            finished, code = self._finished(entry)
            if not finished:
                continue
            del self.running[task_id]
            task = self.by_id[task_id]
            elapsed = time.time() - entry.started
            if is_done(self.root, task):
                self.durations.setdefault(task.run_type, []).append(elapsed)
                self.log(f"done {task_id} in {elapsed / 60:.1f} min")
            else:
                self.failures[task_id] += 1
                self.log(f"FAILED {task_id} (exit {code}), attempt {self.failures[task_id]}")
                if self.failures[task_id] > self.settings.max_retries:
                    self.failed.add(task_id)
                    self.log(f"giving up on {task_id}")

    # ------------------------------------------------------------------
    # progress
    # ------------------------------------------------------------------
    def costs(self) -> StepCosts:
        values = asdict(DEFAULT_COSTS)
        measured = self.root / "step_costs.json"
        if measured.is_file():
            try:
                values.update(json.loads(measured.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        return StepCosts(**values)

    def estimated(self, task: Task, costs: StepCosts) -> float:
        history = self.durations.get(task.run_type)
        if history:
            return sum(history[-20:]) / len(history[-20:])
        return estimate_seconds(task, costs)

    def write_progress(self, done: set[str], capacity: int, mode: str, note: str) -> None:
        costs = self.costs()
        remaining = [t for t in self.plan if t.task_id not in done and t.task_id not in self.failed]
        by_group = Counter()
        totals = Counter()
        for task in self.plan:
            totals[task.priority] += 1
            if task.task_id in done:
                by_group[task.priority] += 1
        running = []
        busy_seconds = 0.0
        equivalents = 0.0
        for task_id, entry in self.running.items():
            task = self.by_id[task_id]
            beat = {}
            path = status_path(self.root, task_id)
            if path.is_file():
                try:
                    beat = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    beat = {}
            elapsed = time.time() - entry.started
            expected = self.estimated(task, costs)
            busy_seconds += max(expected - elapsed, 0.0)
            rate = None
            step, updated = beat.get("step"), beat.get("updated")
            if step is not None and updated is not None:
                previous = self._beats.get(task_id)
                # the first heartbeat of a worker reports step 0 whatever step the run resumes at
                if previous is not None and previous[0] > 0 and updated > previous[1] and step > previous[0]:
                    rate = (step - previous[0]) / (updated - previous[1])
                    self._rates[task_id] = rate
                else:
                    rate = self._rates.get(task_id)
                if previous is None or previous[0] == 0 or updated > previous[1] + 60:
                    self._beats[task_id] = (float(step), float(updated))
                if rate is not None:
                    equivalents += rate * costs.per_step(task.run_type)
            running.append(
                {"task": task_id, "pid": entry.pid, "elapsed_min": round(elapsed / 60, 1),
                 "step": step, "counter": beat.get("counter"),
                 "seconds_per_step": None if not rate else round(1.0 / rate, 3)}
            )
        queued = [t for t in remaining if t.task_id not in self.running]
        work = busy_seconds + sum(self.estimated(t, costs) for t in queued)
        eta_hours = work / max(capacity, 1) / 3600
        progress = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "capacity": capacity,
            "note": note,
            "done": len(done),
            "total": len(self.plan),
            "failed": sorted(self.failed),
            "done_by_priority": {str(k): f"{by_group[k]}/{totals[k]}" for k in sorted(totals)},
            "running": running,
            "eta_hours_at_current_capacity": round(eta_hours, 1),
            "single_process_equivalents": round(equivalents, 2),
            "mean_minutes_by_run_type": {
                k: round(sum(v) / len(v) / 60, 1) for k, v in self.durations.items() if v
            },
        }
        self._write_file(self.root / "progress.json", json.dumps(progress, indent=1))
        lines = [
            f"# Campaign progress ({progress['updated']})",
            "",
            f"- done {progress['done']} / {progress['total']}; failed {len(self.failed)}",
            f"- by priority group: {progress['done_by_priority']}",
            f"- workers: {len(self.running)} running, capacity {capacity} ({mode}) {note}",
            f"- ETA at current capacity: {progress['eta_hours_at_current_capacity']} h",
            f"- throughput: {progress['single_process_equivalents']} single-process equivalents",
            f"- mean minutes by run type: {progress['mean_minutes_by_run_type']}",
            "",
            "| task | pid | elapsed (min) | step | counter | s / step |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r['task']} | {r['pid']} | {r['elapsed_min']} | {r['step']} | {r['counter']} "
            f"| {r['seconds_per_step']} |"
            for r in running
        ]
        self._write_file(self.root / "PROGRESS.md", "\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def done_set(self) -> set[str]:
        return {t.task_id for t in self.plan if is_done(self.root, t)}

    def tick(self) -> bool:
        """One scheduling round; ``False`` once nothing is left to do."""
        self.reap()
        done = self.done_set()
        stopping = (self.root / "STOP").exists()
        capacity, mode = self.capacity()
        note = "stopping" if stopping else ""
        costs = self.costs()
        if not stopping:
            blocked = set(self.failed)
            candidates = [
                t for t in ready_tasks(self.plan, done, set(self.running)) if t.task_id not in blocked
            ]
            candidates.sort(key=lambda t: dispatch_key(t, costs))
            control = self.control()
            heavy_cap = int(control.get("gpu_heavy_workers", self.settings.gpu_heavy_workers))
            heavy_running = sum(1 for t in self.running if is_update_heavy(self.by_id[t]))
            waiting = self.waiting_counts(done) if control.get("hold_lower_priority", True) else None
            for task in candidates:
                if len(self.running) >= capacity:
                    break
                if is_update_heavy(task) and heavy_running >= heavy_cap:
                    continue
                if waiting is not None and _held_back(task, waiting):
                    continue
                ok, why = self.can_launch_more()
                if not ok:
                    note = f"holding: {why}"
                    break
                self.launch(task)
                heavy_running += is_update_heavy(task)
                if waiting is not None:
                    waiting[(is_update_heavy(task), task.priority)] -= 1
        self._save_state()
        self.write_progress(done, capacity, mode, note)
        remaining = [
            t for t in self.plan
            if t.task_id not in done and t.task_id not in self.failed
            and (t.dependency is None or t.dependency not in self.failed)
        ]
        if stopping:
            return bool(self.running)
        return bool(remaining) or bool(self.running)

    def waiting_counts(self, done: set[str]) -> Counter:
        """Tasks waiting to start (ready or waiting for their warm-up), per (update-heavy, priority)."""
        waiting: Counter = Counter()
        for task in self.plan:
            if task.task_id in done or task.task_id in self.running or task.task_id in self.failed:
                continue
            if task.dependency is not None and task.dependency in self.failed:
                continue
            waiting[(is_update_heavy(task), task.priority)] += 1
        return waiting

    def run(self) -> None:
        self.log(f"scheduler started (pid {os.getpid()}), {len(self.plan)} tasks")
        while True:
            try:
                alive = self.tick()
            except Exception as exc:  # noqa: BLE001 - one bad round must not end the campaign
                traceback.print_exc()
                self.log(f"error in scheduling round: {exc!r}")
                alive = True
            if not alive:
                break
            time.sleep(self.settings.poll_seconds)
        self.log("scheduler finished")
