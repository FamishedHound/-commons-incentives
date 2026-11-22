"""Campaign command line.

Usage (from the folder that contains the package)::

    # the plan, task counts and an ETA from the measured step costs
    python -m chapter3_pollinators.campaign plan --root chapter3_pollinators/results/campaign_main

    # start the scheduler as a detached background process
    python -m chapter3_pollinators.campaign launch --root chapter3_pollinators/results/campaign_main \\
        --seeds 0 1 2 3 4 5 6 7 8 9 --group3-seeds 0 1 2 3 4

    python -m chapter3_pollinators.campaign status --root chapter3_pollinators/results/campaign_main
    python -m chapter3_pollinators.campaign stop --root chapter3_pollinators/results/campaign_main

``launch`` writes ``plan.json`` (unless one exists), then starts ``run`` detached
from the terminal; ``run`` is the scheduler itself (see :mod:`.scheduler`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from chapter3_pollinators.campaign.plan import build_plan, estimate_seconds
from chapter3_pollinators.campaign.scheduler import (
    PACKAGE_PARENT,
    Scheduler,
    SchedulerSettings,
    load_plan,
    write_plan,
)

DETACHED = (
    getattr(subprocess, "DETACHED_PROCESS", 0)
    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
)
BREAKAWAY = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)


def _settings(args) -> SchedulerSettings:
    return SchedulerSettings(
        root=args.root,
        busy_workers=args.busy_workers,
        free_workers=args.free_workers,
        poll_seconds=args.poll_seconds,
    )


def _ensure_plan(args) -> None:
    if (args.root / "plan.json").is_file() and not args.replan:
        return
    tasks = build_plan(args.seeds, args.group3_seeds)
    write_plan(
        args.root,
        tasks,
        {"seeds": list(args.seeds), "group3_seeds": list(args.group3_seeds),
         "created": time.strftime("%Y-%m-%d %H:%M:%S")},
    )


def cmd_plan(args) -> int:
    from chapter3_pollinators.campaign.scheduler import DEFAULT_COSTS

    tasks = load_plan(args.root) if (args.root / "plan.json").is_file() else build_plan(
        args.seeds, args.group3_seeds
    )
    scheduler_costs = Scheduler(_settings(args), tasks).costs() if args.root.is_dir() else DEFAULT_COSTS
    totals: dict[str, list[float]] = {}
    for task in tasks:
        totals.setdefault(task.run_type, []).append(estimate_seconds(task, scheduler_costs))
    work = sum(sum(v) for v in totals.values())
    for run_type, values in sorted(totals.items()):
        print(f"{run_type:14s} {len(values):4d} tasks  {sum(values) / 3600:8.1f} worker-hours")
    print(f"{'total':14s} {len(tasks):4d} tasks  {work / 3600:8.1f} worker-hours")
    for workers in (args.busy_workers, args.free_workers):
        print(f"  at {workers:2d} workers: {work / workers / 3600:6.1f} h wall-clock (ideal packing)")
    return 0


def cmd_run(args) -> int:
    _ensure_plan(args)
    Scheduler(_settings(args), load_plan(args.root)).run()
    return 0


def cmd_launch(args) -> int:
    args.root.mkdir(parents=True, exist_ok=True)
    _ensure_plan(args)
    stop = args.root / "STOP"
    if stop.exists():
        stop.unlink()
    command = [
        sys.executable, "-m", "chapter3_pollinators.campaign", "run", "--root", str(args.root),
        "--busy-workers", str(args.busy_workers), "--free-workers", str(args.free_workers),
        "--poll-seconds", str(args.poll_seconds),
    ]
    out = (args.root / "scheduler.out").open("a", encoding="utf-8")
    kwargs = dict(cwd=str(PACKAGE_PARENT), stdout=out, stderr=subprocess.STDOUT,
                  stdin=subprocess.DEVNULL, close_fds=True)
    try:
        process = subprocess.Popen(command, creationflags=DETACHED | BREAKAWAY, **kwargs)
    except OSError:
        process = subprocess.Popen(command, creationflags=DETACHED, **kwargs)
    (args.root / "scheduler.pid").write_text(str(process.pid), encoding="utf-8")
    print(f"scheduler started, pid {process.pid}; progress in {args.root / 'PROGRESS.md'}")
    return 0


def cmd_status(args) -> int:
    path = args.root / "progress.json"
    if not path.is_file():
        print("no progress file yet")
        return 1
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=1))
    return 0


def cmd_stop(args) -> int:
    (args.root / "STOP").write_text("stop\n", encoding="utf-8")
    print("STOP written: no new launches; the scheduler exits after the running workers")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "launch", "run", "status", "stop"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--group3-seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--busy-workers", type=int, default=6)
    parser.add_argument("--free-workers", type=int, default=24)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--replan", action="store_true", help="rewrite plan.json")
    args = parser.parse_args(argv)
    commands = {"plan": cmd_plan, "launch": cmd_launch, "run": cmd_run,
                "status": cmd_status, "stop": cmd_stop}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
