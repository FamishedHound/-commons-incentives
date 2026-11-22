"""Run one campaign task in its own process.

Usage::

    python -m chapter3_pollinators.campaign.worker --root results/campaign_main --task warmup:HT1:s0

The worker writes ``status/<task>.json`` every 50 steps (step, counter,
elapsed time) and exits 0 once the task's record is on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from chapter3_pollinators.campaign.plan import Task, parse_task_id
from chapter3_pollinators.experiment import atomic
from chapter3_pollinators.experiment.run import (
    branch_paths,
    no_regulator_paths,
    run_branch,
    run_no_regulator,
    run_warmup,
    warmup_dir,
)


def safe_name(task_id: str) -> str:
    return task_id.replace(":", "_")


def completion_marker(root: Path, task: Task) -> Path:
    """The file whose presence marks ``task`` as finished."""
    if task.kind == "warmup":
        return warmup_dir(root, task.community, task.seed) / "warmup.json"
    if task.kind == "no_regulator":
        return no_regulator_paths(root, task.community, task.seed)[1]
    return branch_paths(root, task.community, task.government, task.condition, task.seed)[1]


def is_done(root: Path, task: Task) -> bool:
    return completion_marker(root, task).is_file()


def status_path(root: Path, task_id: str) -> Path:
    return Path(root) / "status" / f"{safe_name(task_id)}.json"


def _heartbeat_writer(root: Path, task_id: str, started: float):
    path = status_path(root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(beat: dict) -> None:
        payload = {**beat, "task_id": task_id, "pid": os.getpid(), "updated": time.time(),
                   "started": started}
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            atomic.replace_file(tmp, path)
        except OSError as exc:  # a missed heartbeat only delays the progress report
            print(f"heartbeat not written: {exc}", file=sys.stderr, flush=True)

    return write


def execute(root: Path, task: Task) -> None:
    started = time.time()
    beat = _heartbeat_writer(root, task.task_id, started)
    beat({"task": task.task_id, "step": 0, "counter": None, "epsilon": None, "elapsed_s": 0.0})
    if task.kind == "warmup":
        run_warmup(task.community, task.seed, root, heartbeat=beat)
    elif task.kind == "no_regulator":
        run_no_regulator(task.community, task.seed, root, heartbeat=beat)
    else:
        run_branch(task.community, task.government, task.condition, task.seed, root, heartbeat=beat)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    task = parse_task_id(args.task)
    if is_done(args.root, task):
        print(f"{args.task}: already done", flush=True)
        return 0
    try:
        execute(args.root, task)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"{args.task}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
