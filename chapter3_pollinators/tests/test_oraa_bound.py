"""ORAA's incentive bound: the default reproduces the regulator as coded; a wider bound is searched when set."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch as th

from chapter3_pollinators.env.actions import make_random_action
from chapter3_pollinators.env.game import PollinatorsGame
from chapter3_pollinators.experiment.driver import Driver, LoopState
from chapter3_pollinators.experiment.schedule import BRANCH_COUNTER, CONDITION_SCHEDULES
from chapter3_pollinators.experiment.settings import (
    BOUND_CONDITIONS,
    CONDITIONS,
    GROUPS,
    MAIN_CONDITIONS,
    base_condition,
    oraa_bound,
)
from chapter3_pollinators.learning.pmaddpg import PMADDPG
from chapter3_pollinators.regulators.oraa import DEFAULT_BOUND, ORAA, tpe_seed


def _game(seed: int, steps: int = 6):
    random.seed(seed)
    game = PollinatorsGame()
    game.reset()
    obs = game.reset()
    for _ in range(steps):
        obs, _, _, _ = game.step(make_random_action(game.agents), False, [0, 0, 0, 0])
    return game, obs


class _RecordingTrial:
    def __init__(self, x: float) -> None:
        self.x = x
        self.calls = []

    def suggest_float(self, name, low, high):
        self.calls.append((name, low, high))
        return self.x


def test_the_default_bound_is_one() -> None:
    assert DEFAULT_BOUND == 1.0
    assert ORAA([0.2] * 4).bound == 1.0


@pytest.mark.usefixtures("cuda_required")
def test_default_and_explicit_bound_one_are_bit_identical() -> None:
    """With the default bound ORAA is the regulator of the equivalence tests (see
    ``test_regulators_equivalence``): an explicit bound of 1 changes nothing."""
    game, (obs, _) = _game(2)
    th.manual_seed(2)
    learner = PMADDPG(4, 12, 1, 16, 9000, 10, game.agents)
    target = [0.2, 0.1, 0.8, 0.9]
    seeds = [tpe_seed(2, 1000, i) for i in range(4)]
    default = ORAA(target, n_trials=60)
    explicit = ORAA(target, n_trials=60, bound=1.0)
    d_incentive, d_obs = default.optimise_incentives(obs, game.agents, learner.actors, seeds)
    e_incentive, e_obs = explicit.optimise_incentives(obs, game.agents, learner.actors, seeds)
    assert d_incentive == e_incentive
    assert np.array_equal(np.asarray(d_obs), np.asarray(e_obs))
    assert default.last_best_values == explicit.last_best_values
    assert all(abs(x) <= 1.0 for x in d_incentive)


@pytest.mark.usefixtures("cuda_required")
def test_a_wider_bound_is_searched_when_set() -> None:
    game, (obs, _) = _game(3)
    th.manual_seed(3)
    learner = PMADDPG(4, 12, 1, 16, 9000, 10, game.agents)
    target = [1.0, 1.0, 0.0, 0.0]
    oraa = ORAA(target, n_trials=80, bound=3.0)
    oraa.counter_networks = 0
    oraa.agent_networks = learner.actors
    oraa.interpreted_obs = np.array(obs)[0][0][0]
    oraa._prepare_device_inputs(game.agents[0])
    trial = _RecordingTrial(2.5)
    oraa.objective(trial)
    assert trial.calls == [("x", -3.0, 3.0)]
    painted = oraa.new_state[0, 1].cpu().numpy()
    assert set(np.unique(painted)) <= {0.0, 2.5} and (painted == 2.5).sum() == len(game.agents[0].cells)
    seeds = [tpe_seed(3, 1000, i) for i in range(4)]
    incentive, _ = oraa.optimise_incentives(obs, game.agents, learner.actors, seeds)
    assert all(abs(x) <= 3.0 for x in incentive)
    assert max(abs(x) for x in incentive) > 1.0


def test_bound_conditions_are_the_oraa_schedules_with_bound_three() -> None:
    assert MAIN_CONDITIONS == ("no_regulator", "reward_shape", "oraa_mb", "oraa_omni", "ltos")
    assert BOUND_CONDITIONS == ("oraa_mb_b3", "oraa_omni_b3")
    assert CONDITIONS == MAIN_CONDITIONS + BOUND_CONDITIONS
    for condition, base in (("oraa_mb_b3", "oraa_mb"), ("oraa_omni_b3", "oraa_omni")):
        assert base_condition(condition) == base and oraa_bound(condition) == 3.0
        wide, narrow = CONDITION_SCHEDULES[condition], CONDITION_SCHEDULES[base]
        assert wide.oraa_bound == 3.0 and narrow.oraa_bound == 1.0
        assert (wide.oraa_from, wide.oraa_networks, wide.curriculum) == (
            narrow.oraa_from, narrow.oraa_networks, narrow.curriculum)
    assert base_condition("reward_shape") == "reward_shape" and oraa_bound("oraa_mb") == 1.0
    assert oraa_bound("reward_shape") is None


def test_group_four_is_the_bound_experiment_after_the_others() -> None:
    group = GROUPS[-1]
    assert group.priority == 4 and group.government == "HTC"
    assert group.communities == ("HT1", "HT2", "HT3", "HT4")
    assert group.conditions == BOUND_CONDITIONS
    assert GROUPS[0].conditions == MAIN_CONDITIONS
    assert [g.priority for g in GROUPS] == [1, 2, 3, 3, 4]


@pytest.mark.usefixtures("cuda_required")
def test_the_driver_passes_the_bound_to_oraa() -> None:
    game, (obs, global_state) = _game(1, steps=0)
    th.manual_seed(1)
    real = PMADDPG(4, 12, 1, 16, 9000, 10, game.agents)
    model = PMADDPG(4, 12, 1, 16, 9000, 10, game.agents)
    state = LoopState(obs=obs, global_state=global_state, epsilon=0.4, counters=BRANCH_COUNTER - 1)
    for condition, bound in (("oraa_mb", 1.0), ("oraa_mb_b3", 3.0), ("oraa_omni_b3", 3.0)):
        driver = Driver(game, real, model, state, CONDITION_SCHEDULES[condition], seed=0)
        assert driver.oraa.bound == bound
