"""Measure the per-step cost of every run type on this machine.

A short run is fast-forwarded to the end of the curriculum (epsilon 0.4,
counter 975) and played for 24 steps with both learners, which fills the replay
memory past the 16 transitions training needs.  Then each run type is timed on
its own copy of that state:

* ``warmup``: the curriculum with the landowners' learner and the model learner;
* ``no_regulator``: no incentive, the landowners' learner only;
* each regulated condition, exactly as in its branch (500 ORAA trials).

The cost of a step does not depend on what the networks have learnt (a fixed
mini-batch per update, a fixed number of ORAA trials), so these timings scale
to full runs: a warm-up is about 7,000 steps, a run without a regulator about
6,200 and a branch 1,200.

Usage::

    python -m chapter3_pollinators.campaign.measure --steps 40 --out results/cost_measurement.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from chapter3_pollinators.experiment.driver import Driver
from chapter3_pollinators.experiment.run import environment_info, new_run
from chapter3_pollinators.experiment.schedule import CONDITION_SCHEDULES, WARMUP
from chapter3_pollinators.experiment.settings import community_alphas, government_targets
from chapter3_pollinators.regulators.oraa import N_TRIALS

RUN_TYPES = ("warmup", "no_regulator", "reward_shape", "oraa_mb", "oraa_omni", "ltos")


def _prepared_run(seed: int, community: str):
    run = new_run(seed, community_alphas(community), with_model=True)
    run.state.epsilon = 0.4
    run.state.counters = 975
    driver = Driver(run.game, run.real, run.model, run.state, WARMUP, seed)
    for _ in range(24):
        driver.step()
    return run


def time_run_type(run_type: str, steps: int, seed: int = 0, community: str = "HT1",
                  oraa_trials: int = N_TRIALS) -> dict:
    run = _prepared_run(seed, community)
    targets = government_targets(community, "HTC")
    if run_type == "warmup":
        run.state.counters = 0
        run.state.epsilon = 0.8
        driver = Driver(run.game, run.real, run.model, run.state, WARMUP, seed)
    else:
        model = run.model if run_type == "oraa_mb" else None
        sharing = None
        if run_type == "ltos":
            from chapter3_pollinators.ltos.scheme import RewardSharing

            sharing = RewardSharing.for_run(run, seed=seed)
        driver = Driver(
            run.game, run.real, model, run.state, CONDITION_SCHEDULES[run_type], seed,
            targets=targets, oraa_trials=oraa_trials, sharing=sharing,
        )
    durations = []
    for _ in range(steps):
        started = time.perf_counter()
        driver.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return {
        "run_type": run_type,
        "steps": steps,
        "mean_s": statistics.fmean(durations),
        "median_s": statistics.median(durations),
        "max_s": max(durations),
        "min_s": min(durations),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--run-types", nargs="+", default=list(RUN_TYPES), choices=RUN_TYPES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    results = {"environment": environment_info(), "timings": []}
    for run_type in args.run_types:
        timing = time_run_type(run_type, args.steps)
        results["timings"].append(timing)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(timing), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
