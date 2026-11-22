"""The campaign plan: every warm-up, branch and run without a regulator.

One warm-up per (community, seed) serves every branch of that community and
seed, whichever the government, because nothing before the branch point depends
on the targets.  For the same reason one run without a regulator per (community,
seed) serves both governments; the targets only enter its metrics.

Dispatch order: priority group first (HT+HTC, then HT+HMC, then the HM
groups, then the runs of ORAA with the wider incentive bound, which reuse the
HT warm-ups); within a group every warm-up first, in seed order, because each
one unlocks the runs of its seed; then the runs seed by seed, the longest first.
Seeds therefore complete one after another within a group, which gives complete
per-seed comparisons early.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chapter3_pollinators.experiment.schedule import WINDOW_STEPS
from chapter3_pollinators.experiment.settings import GROUPS, base_condition

#: Approximate length of each run type in environment steps (see ``schedule``).
WARMUP_STEPS = 7015
NO_REGULATOR_STEPS = 6199

KINDS = ("warmup", "no_regulator", "branch")


@dataclass(frozen=True)
class Task:
    kind: str
    community: str
    seed: int
    government: str | None = None
    condition: str | None = None
    priority: int = 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Task) and self.task_id == other.task_id

    def __hash__(self) -> int:
        return hash(self.task_id)

    @property
    def task_id(self) -> str:
        if self.kind == "branch":
            return f"branch:{self.community}:{self.government}:{self.condition}:s{self.seed}"
        return f"{self.kind}:{self.community}:s{self.seed}"

    @property
    def dependency(self) -> str | None:
        return None if self.kind == "warmup" else f"warmup:{self.community}:s{self.seed}"

    @property
    def run_type(self) -> str:
        return self.condition if self.kind == "branch" else self.kind


def parse_task_id(task_id: str, priority: int = 0) -> Task:
    parts = task_id.split(":")
    seed = int(parts[-1][1:])
    if parts[0] == "branch":
        return Task("branch", parts[1], seed, parts[2], parts[3], priority)
    if parts[0] in ("warmup", "no_regulator"):
        return Task(parts[0], parts[1], seed, priority=priority)
    raise ValueError(f"not a task id: {task_id!r}")


def build_plan(seeds: Iterable[int], group3_seeds: Iterable[int]) -> list[Task]:
    """All tasks of the campaign; group 3 (the HM groups) uses ``group3_seeds``."""
    seeds, group3_seeds = list(seeds), list(group3_seeds)
    tasks: dict[str, Task] = {}

    def add(task: Task) -> None:
        current = tasks.get(task.task_id)
        if current is None or task.priority < current.priority:
            tasks[task.task_id] = task

    for group in GROUPS:
        group_seeds = group3_seeds if group.priority == 3 else seeds
        for seed in group_seeds:
            for community in group.communities:
                add(Task("warmup", community, seed, priority=group.priority))
                for condition in group.conditions:
                    if condition == "no_regulator":
                        add(Task("no_regulator", community, seed, priority=group.priority))
                    else:
                        add(Task("branch", community, seed, group.government, condition, group.priority))
    return list(tasks.values())


@dataclass(frozen=True)
class StepCosts:
    """Seconds per environment step of each run type."""

    warmup: float
    no_regulator: float
    reward_shape: float
    oraa_mb: float
    oraa_omni: float
    ltos: float

    def per_step(self, run_type: str) -> float:
        """Seconds per step; a bound variant of ORAA costs what its form costs."""
        return getattr(self, base_condition(run_type))


def estimate_seconds(task: Task, costs: StepCosts) -> float:
    if task.kind == "warmup":
        return WARMUP_STEPS * costs.warmup
    if task.kind == "no_regulator":
        return NO_REGULATOR_STEPS * costs.no_regulator
    return WINDOW_STEPS * costs.per_step(task.condition)


def dispatch_key(task: Task, costs: StepCosts) -> tuple:
    kind_rank = 0 if task.kind == "warmup" else 1
    return (
        task.priority,
        kind_rank,
        task.seed,
        -estimate_seconds(task, costs),
        task.community,
        task.government or "",
        task.condition or "",
    )


def ready_tasks(plan: Iterable[Task], done: set[str], running: set[str]) -> list[Task]:
    """Tasks not done or running whose dependency is done."""
    return [
        t
        for t in plan
        if t.task_id not in done
        and t.task_id not in running
        and (t.dependency is None or t.dependency in done)
    ]
