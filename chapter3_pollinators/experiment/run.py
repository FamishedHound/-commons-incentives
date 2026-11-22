"""Single runs: the warm-up, a regulated branch, and the run without a regulator.

* :func:`run_warmup` plays one community and seed from step 0 through the
  curriculum to counted step 999 with the landowners' learner and the model
  learner (the second, independently initialised P-MADDPG that Model-based ORAA
  consults, trained on the same transitions).  It writes two checkpoints:
  ``precurriculum.pt`` at the start of the first step with epsilon <= 0.8 (the
  last state before any incentive is ever paid) and ``branch.pt`` after counted
  step 999.
* :func:`run_branch` restores ``branch.pt`` and plays counted steps
  1,000-2,199 under one regulated condition.  The model learner is kept (and
  trained) only for Model-based ORAA.  ``oraa_mb_b3`` and ``oraa_omni_b3`` are
  the two forms of ORAA with the incentive searched in [-3, 3] instead of
  [-1, 1]; everything else is as in ``oraa_mb`` and ``oraa_omni``.
* :func:`run_no_regulator` restores ``precurriculum.pt`` and plays on with no
  incentive at all, through counted step 2,199.  Up to the restore point its
  history is that of the warm-up, which paid no incentive either, so it is the
  run the same seed produces without a regulator.

Usage::

    python -m chapter3_pollinators.experiment.run warmup --community HT1 --seed 0 --root results/campaign_main
    python -m chapter3_pollinators.experiment.run branch --community HT1 --government HTC \\
        --condition oraa_mb --seed 0 --root results/campaign_main
    python -m chapter3_pollinators.experiment.run no_regulator --community HT1 --seed 0 --root results/campaign_main
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from chapter3_pollinators import __version__
from chapter3_pollinators.env.game import PollinatorsGame
from chapter3_pollinators.experiment import atomic
from chapter3_pollinators.experiment.checkpoint import load_checkpoint, save_checkpoint
from chapter3_pollinators.experiment.driver import Driver, LoopState
from chapter3_pollinators.experiment.metrics import window_metrics
from chapter3_pollinators.experiment.recorder import Recorder
from chapter3_pollinators.experiment.schedule import (
    BATCH_SIZE,
    BRANCH_COUNTER,
    CAPACITY,
    CONDITION_SCHEDULES,
    CURRICULUM_EPSILON,
    DIM_ACT,
    DIM_OBS,
    EPISODES_BEFORE_TRAIN,
    EPSILON_START,
    LAST_COUNTER,
    N_AGENTS,
    WARMUP,
)
from chapter3_pollinators.experiment.seeding import seed_everything
from chapter3_pollinators.experiment.settings import (
    CONDITIONS,
    GOVERNMENTS,
    community_alphas,
    government_targets,
)
from chapter3_pollinators.learning.pmaddpg import PMADDPG
from chapter3_pollinators.regulators.oraa import N_TRIALS

HEARTBEAT_EVERY = 50
Heartbeat = Callable[[dict], None]
Prepare = Callable[["Run"], None]


@dataclass
class Run:
    """The objects of one run."""

    seed: int
    game: PollinatorsGame
    real: PMADDPG
    model: PMADDPG | None
    state: LoopState
    meta: dict = field(default_factory=dict)


def build_learner(agents) -> PMADDPG:
    return PMADDPG(N_AGENTS, DIM_OBS, DIM_ACT, BATCH_SIZE, CAPACITY, EPISODES_BEFORE_TRAIN, agents)


def new_run(seed: int, alphas, with_model: bool = True) -> Run:
    """A fresh run: seed every generator, build the game, then the learners."""
    seed_everything(seed)
    game = PollinatorsGame()
    game.reset()
    obs, global_state = game.reset()
    agents = game.agent_processor.all_agents
    real = build_learner(agents)
    model = build_learner(agents) if with_model else None
    for i, agent in enumerate(agents):
        agent.alpha = alphas[i]
    state = LoopState(obs=obs, global_state=global_state, epsilon=EPSILON_START, counters=0)
    return Run(seed=seed, game=game, real=real, model=model, state=state)


# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
def warmup_dir(root: Path, community: str, seed: int) -> Path:
    return Path(root) / "warmups" / community / f"seed{seed}"


def branch_paths(root: Path, community: str, government: str, condition: str, seed: int):
    base = Path(root) / "runs" / community / government / f"{condition}_seed{seed}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def no_regulator_paths(root: Path, community: str, seed: int):
    base = Path(root) / "runs" / community / f"no_regulator_seed{seed}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    atomic.replace_file(tmp, path)


def environment_info() -> dict:
    import optuna

    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "optuna": optuna.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(),
    }


def _beat(heartbeat: Heartbeat | None, run: Run, task: str, started: float) -> None:
    if heartbeat is not None and run.state.step % HEARTBEAT_EVERY == 0:
        heartbeat(
            {
                "task": task,
                "step": run.state.step,
                "counter": run.state.counters,
                "epsilon": run.state.epsilon,
                "elapsed_s": time.time() - started,
            }
        )


def _drive(driver: Driver, run: Run, last_counter: int, heartbeat, task: str, started: float):
    while run.state.counters < last_counter:
        driver.step()
        _beat(heartbeat, run, task, started)


# ----------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------
def _has_window(records: dict) -> bool:
    counter = records["counter"]
    return bool(((counter >= BRANCH_COUNTER) & (counter <= LAST_COUNTER)).any())


def run_warmup(
    community: str,
    seed: int,
    root: Path,
    heartbeat: Heartbeat | None = None,
    prepare: Prepare | None = None,
) -> dict:
    """``prepare`` may adjust the fresh run before the first step (used by the tests)."""
    out = warmup_dir(root, community, seed)
    started = time.time()
    run = new_run(seed, community_alphas(community), with_model=True)
    if prepare is not None:
        prepare(run)
    recorder = Recorder()
    driver = Driver(run.game, run.real, run.model, run.state, WARMUP, seed, recorder=recorder)
    meta = {"community": community, "seed": seed, "alphas": list(community_alphas(community))}
    precurriculum_step = None
    task = f"warmup {community} seed{seed}"
    while run.state.counters < BRANCH_COUNTER - 1:
        if precurriculum_step is None and run.state.epsilon <= CURRICULUM_EPSILON:
            precurriculum_step = run.state.step
            save_checkpoint(
                out / "precurriculum.pt",
                run,
                include_model=False,
                meta={**meta, "kind": "precurriculum", "step": run.state.step},
            )
        driver.step()
        _beat(heartbeat, run, task, started)
    save_checkpoint(
        out / "branch.pt", run, include_model=True,
        meta={**meta, "kind": "branch", "step": run.state.step},
    )
    recorder.save(out / "steps.npz")
    record = {
        **meta,
        "kind": "warmup",
        "precurriculum_step": precurriculum_step,
        "branch_step": run.state.step,
        "first_counted_step": int(np.argmax(recorder.arrays()["counter"] > 0)),
        "steps": run.state.step,
        "runtime_s": time.time() - started,
        "environment": environment_info(),
    }
    write_json(out / "warmup.json", record)
    return record


def run_branch(
    community: str,
    government: str,
    condition: str,
    seed: int,
    root: Path,
    heartbeat: Heartbeat | None = None,
    oraa_trials: int = N_TRIALS,
    last_counter: int = LAST_COUNTER,
) -> dict:
    if condition not in CONDITIONS or condition == "no_regulator":
        raise ValueError(f"{condition!r} is not a regulated condition")
    started = time.time()
    run = load_checkpoint(warmup_dir(root, community, seed) / "branch.pt")
    targets = government_targets(community, government)
    schedule = CONDITION_SCHEDULES[condition]
    with_oraa = schedule.oraa_from is not None
    model = run.model if with_oraa and schedule.oraa_networks == "model" else None
    sharing = None
    if condition == "ltos":
        from chapter3_pollinators.ltos.scheme import RewardSharing

        sharing = RewardSharing.for_run(run, seed=seed)
    recorder = Recorder()
    driver = Driver(
        run.game, run.real, model, run.state, schedule, seed,
        targets=targets, recorder=recorder, oraa_trials=oraa_trials, sharing=sharing,
    )
    task = f"{condition} {community}+{government} seed{seed}"
    _drive(driver, run, last_counter, heartbeat, task, started)
    npz_path, json_path = branch_paths(root, community, government, condition, seed)
    recorder.save(npz_path)
    records = recorder.arrays()
    record = {
        "kind": "branch",
        "community": community,
        "government": government,
        "condition": condition,
        "seed": seed,
        "targets": list(targets),
        "alphas": list(community_alphas(community)),
        "oraa_trials": oraa_trials if with_oraa else None,
        "oraa_bound": schedule.oraa_bound if with_oraa else None,
        "steps": len(recorder),
        "runtime_s": time.time() - started,
        "metrics": window_metrics(records, targets, condition) if _has_window(records) else None,
        "metrics_greedy": (
            window_metrics(records, targets, condition, greedy_only=True)
            if _has_window(records) and records["greedy"][records["counter"] >= BRANCH_COUNTER].any()
            else None
        ),
        "environment": environment_info(),
    }
    if sharing is not None:
        record["ltos"] = sharing.summary()
    write_json(json_path, record)
    return record


def run_no_regulator(
    community: str,
    seed: int,
    root: Path,
    heartbeat: Heartbeat | None = None,
    last_counter: int = LAST_COUNTER,
) -> dict:
    started = time.time()
    run = load_checkpoint(warmup_dir(root, community, seed) / "precurriculum.pt")
    start_step = run.state.step
    recorder = Recorder()
    driver = Driver(
        run.game, run.real, None, run.state, CONDITION_SCHEDULES["no_regulator"], seed,
        recorder=recorder,
    )
    task = f"no_regulator {community} seed{seed}"
    _drive(driver, run, last_counter, heartbeat, task, started)
    npz_path, json_path = no_regulator_paths(root, community, seed)
    recorder.save(npz_path)
    records = recorder.arrays()
    metrics = {}
    if _has_window(records):
        for government in GOVERNMENTS:
            targets = government_targets(community, government)
            greedy_ok = records["greedy"][records["counter"] >= BRANCH_COUNTER].any()
            metrics[government] = {
                "targets": list(targets),
                "metrics": window_metrics(records, targets, "no_regulator"),
                "metrics_greedy": (
                    window_metrics(records, targets, "no_regulator", greedy_only=True)
                    if greedy_ok
                    else None
                ),
            }
    record = {
        "kind": "no_regulator",
        "community": community,
        "condition": "no_regulator",
        "seed": seed,
        "alphas": list(community_alphas(community)),
        "start_step": start_step,
        "steps": len(recorder),
        "runtime_s": time.time() - started,
        "by_government": metrics,
        "environment": environment_info(),
    }
    write_json(json_path, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["warmup", "branch", "no_regulator"])
    parser.add_argument("--community", required=True)
    parser.add_argument("--government", choices=list(GOVERNMENTS))
    parser.add_argument("--condition", choices=[c for c in CONDITIONS if c != "no_regulator"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.kind == "warmup":
        run_warmup(args.community, args.seed, args.root)
    elif args.kind == "branch":
        if args.government is None or args.condition is None:
            parser.error("branch needs --government and --condition")
        run_branch(args.community, args.government, args.condition, args.seed, args.root)
    else:
        run_no_regulator(args.community, args.seed, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
