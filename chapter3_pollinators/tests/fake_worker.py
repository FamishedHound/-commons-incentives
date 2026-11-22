"""A stand-in worker for the scheduler tests: it only writes the task's completion record.

A task whose id appears in ``<root>/fail.txt`` exits with an error instead.
Each worker also writes its CPU priority to ``<root>/priority/<task>.txt``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import psutil

from chapter3_pollinators.campaign.plan import parse_task_id
from chapter3_pollinators.campaign.worker import completion_marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    failing = args.root / "fail.txt"
    if failing.is_file() and args.task in failing.read_text().split():
        return 3
    started = time.time_ns()
    time.sleep(0.05)
    marker = completion_marker(args.root, parse_task_id(args.task))
    marker.parent.mkdir(parents=True, exist_ok=True)
    order = args.root / "order"
    order.mkdir(exist_ok=True)
    (order / (args.task.replace(":", "_") + ".txt")).write_text(f"{started} {args.task}")
    priority = args.root / "priority"
    priority.mkdir(exist_ok=True)
    (priority / (args.task.replace(":", "_") + ".txt")).write_text(str(int(psutil.Process().nice())))
    marker.write_text("{}", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
