"""The curriculum rules, the reward-shape incentive and ORAA match the reference regulator."""

from __future__ import annotations

import random
import types

import numpy as np
import optuna
import pytest
import torch as th

from chapter3_pollinators.env.actions import make_random_action
from chapter3_pollinators.env.game import PollinatorsGame
from chapter3_pollinators.learning.pmaddpg import PMADDPG
from chapter3_pollinators.regulators.curriculum import penalising_incentive, rewarding_incentive
from chapter3_pollinators.regulators.oraa import ORAA, tpe_seed
from chapter3_pollinators.regulators.reward_shape import reward_shape_incentive


def _reference_view(agents):
    """The reference regulator reads ``land_cells_owned`` and ``bag_pointer_actual``."""
    views = []
    for agent in agents:
        cells = [
            types.SimpleNamespace(x=c.x, y=c.y, bag_pointer_actual=c.actual_habitat)
            for c in agent.cells
        ]
        views.append(types.SimpleNamespace(land_cells_owned=cells))
    return views


def _game_with_habitat(seed: int, steps: int):
    random.seed(seed)
    game = PollinatorsGame()
    game.reset()
    obs = game.reset()
    for _ in range(steps):
        obs, _, _, _ = game.step(make_random_action(game.agents), False, [0, 0, 0, 0])
    return game, obs


@pytest.mark.parametrize("seed", range(6))
def test_curriculum_rules_match_reference(reference, seed: int) -> None:
    game, _ = _game_with_habitat(seed, steps=3 + seed)
    meta = reference.goverment_agent.goverment_agent(None, None, _reference_view(game.agents))
    assert penalising_incentive(game.agents) == meta.distribute_incetive()
    assert rewarding_incentive(game.agents) == meta.distribute_incentive_2()


def test_curriculum_thresholds_are_inclusive() -> None:
    game, _ = _game_with_habitat(0, steps=0)
    for agent, level in zip(game.agents, (80, 50, 40, 100)):
        for cell in agent.cells:
            cell.actual_habitat = level
    assert penalising_incentive(game.agents) == [-1, -0.6, 0, -1]
    assert rewarding_incentive(game.agents) == [0.6, 0.1, -0.25, 0.6]


def test_reward_shape_matches_reference_formula() -> None:
    action = [
        [th.tensor(0.1), th.tensor(0.3), th.tensor(0.9), th.tensor(0.0)],
        [th.tensor(1.0)] * 5,
        [th.tensor(0.35)] * 4,
        [th.tensor(0.2), th.tensor(0.4), th.tensor(0.6)],
    ]
    target = [0.35, 0.35, 0.35, 0.35]
    expected = []
    for i in range(4):
        average_action = sum([x.item() for x in action[i]]) / len(action[i])
        expected.append(1 - (abs(target[i] - average_action)))
    assert reward_shape_incentive(action, target) == expected


def test_tpe_seed_is_deterministic_and_distinct() -> None:
    assert tpe_seed(3, 1000, 2) == tpe_seed(3, 1000, 2)
    seeds = {tpe_seed(s, c, a) for s in range(3) for c in (1000, 1001) for a in range(4)}
    assert len(seeds) == 24
    assert all(0 <= value < 2**32 for value in seeds)


@pytest.mark.usefixtures("cuda_required")
@pytest.mark.parametrize("n_trials", [40, 500])
def test_oraa_matches_reference_with_seeded_samplers(reference, monkeypatch, n_trials: int) -> None:
    game, (obs, _) = _game_with_habitat(2, steps=6)
    th.manual_seed(2)
    learner = PMADDPG(4, 12, 1, 16, 9000, 10, game.agents)
    target = [0.2, 0.1, 0.8, 0.9]
    seeds = [tpe_seed(2, 1000, i) for i in range(4)]

    meta = reference.goverment_agent.goverment_agent(None, None, _reference_view(game.agents))
    meta.target = list(target)
    queue = list(seeds)
    real_create_study = optuna.create_study

    def seeded_create_study(*args, **kwargs):
        kwargs["sampler"] = optuna.samplers.TPESampler(seed=queue.pop(0))
        return real_create_study(*args, **kwargs)

    monkeypatch.setattr(optuna, "create_study", seeded_create_study)
    original_optimize = optuna.study.Study.optimize

    def limited_optimize(self, func, n_trials=None, **kwargs):
        """The reference runs 500 trials; the short case runs ``n_trials``."""
        return original_optimize(self, func, n_trials=trials, **kwargs)

    trials = n_trials
    monkeypatch.setattr(optuna.study.Study, "optimize", limited_optimize)

    state = (random.getstate(), np.random.get_state(), th.get_rng_state())
    r_incentive, r_obs = meta.optimise_incentives(
        obs, _reference_view(game.agents), learner.actors, learner.critics
    )
    monkeypatch.setattr(optuna.study.Study, "optimize", original_optimize)
    monkeypatch.setattr(optuna, "create_study", real_create_study)

    oraa = ORAA(target, n_trials=n_trials)
    p_incentive, p_obs = oraa.optimise_incentives(obs, game.agents, learner.actors, seeds)
    assert p_incentive == r_incentive
    assert np.array_equal(np.asarray(p_obs), np.asarray(r_obs))
    assert isinstance(p_obs, list)
    assert len(oraa.last_best_values) == 4
    assert all(v >= 0 for v in oraa.last_best_values)
    # the regulator draws from none of the global generators
    assert random.getstate() == state[0]
    assert np.array_equal(np.random.get_state()[1], state[1][1])
    assert th.equal(th.get_rng_state(), state[2])
