"""A run restored from a checkpoint continues exactly as the uninterrupted run."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch as th

from chapter3_pollinators.experiment.checkpoint import (
    compact_memory,
    load_checkpoint,
    restore_memory,
    save_checkpoint,
)
from chapter3_pollinators.experiment.driver import Driver
from chapter3_pollinators.experiment.recorder import Recorder
from chapter3_pollinators.experiment.run import new_run
from chapter3_pollinators.experiment.schedule import Schedule

pytestmark = pytest.mark.usefixtures("cuda_required")

SCHEDULE = Schedule(curriculum=True)


def _drive(run, steps, recorder):
    driver = Driver(run.game, run.real, run.model, run.state, SCHEDULE, seed=run.seed, recorder=recorder)
    for _ in range(steps):
        driver.step()


def _scramble_generators() -> None:
    random.seed(999)
    np.random.seed(999)
    th.manual_seed(999)
    random.random()


@pytest.mark.parametrize("with_model", [True, False])
def test_resume_is_bit_identical(tmp_path, with_model: bool) -> None:
    run = new_run(seed=3, alphas=(0.4, 1.0, 0.5, 0.6), with_model=True)
    run.state.epsilon = 0.8  # start inside the penalising phase
    _drive(run, 24, Recorder())
    path = tmp_path / "state.pt"
    save_checkpoint(path, run, include_model=with_model, meta={"note": "test"})
    register_order = list(run.game.economy_manager.pollinators_processor.all_pollinators)
    continued = Recorder()
    _drive(run, 22, continued)

    _scramble_generators()
    restored = load_checkpoint(path)
    assert restored.meta["note"] == "test"
    assert (restored.model is not None) == with_model
    assert list(restored.game.economy_manager.pollinators_processor.all_pollinators) == register_order
    assert (
        restored.game.economy_manager.pollinators_processor
        is restored.game.environmental_manager.pollinators_processor
    )
    if not with_model:
        restored.model = None
        run_model = None
    resumed = Recorder()
    if with_model:
        _drive(restored, 22, resumed)
        a, b = continued.arrays(), resumed.arrays()
        for name in a:
            assert np.array_equal(a[name], b[name], equal_nan=True), name
        for p, q in zip(
            run.real.actors + run.real.critics + run.model.actors,
            restored.real.actors + restored.real.critics + restored.model.actors,
        ):
            for key, value in p.state_dict().items():
                assert th.equal(value, q.state_dict()[key]), key
    else:
        assert run_model is None
        assert restored.real.episode_done == run.real.episode_done - 22


def test_memory_round_trip_preserves_values_and_types() -> None:
    run = new_run(seed=1, alphas=(0.6,) * 4, with_model=False)
    _drive(run, 30, None)
    memory = run.real.memory
    rebuilt = restore_memory(compact_memory(memory))
    assert rebuilt.position == memory.position and rebuilt.capacity == memory.capacity
    assert len(rebuilt) == len(memory)
    for original, copy in zip(memory.memory, rebuilt.memory):
        for agent in range(4):
            assert np.array_equal(np.asarray(original.states[agent]), np.asarray(copy.states[agent]))
            assert type(original.states[agent]) is type(copy.states[agent])
            assert np.array_equal(
                np.asarray(original.next_states[agent]), np.asarray(copy.next_states[agent])
            )
            assert [float(a) for a in original.actions[agent]] == [float(a) for a in copy.actions[agent]]
            assert all(a.dtype == th.float32 and a.dim() == 0 for a in copy.actions[agent])
            assert original.rewards[agent] == copy.rewards[agent]
            assert type(original.rewards[agent]) is type(copy.rewards[agent])
        assert np.array_equal(original.global_state, copy.global_state)
        assert np.array_equal(original.next_global_state, copy.next_global_state)
