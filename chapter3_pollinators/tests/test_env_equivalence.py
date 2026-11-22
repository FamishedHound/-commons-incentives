"""The ported Pollinators' Game reproduces the reference implementation bit for bit.

Every test seeds Python's ``random`` (the only generator the game uses), builds
the reference game and the package game from the same seed, drives both with
identical actions and incentives, and requires identical observations, global
states, rewards, economy state and generator state after every step.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch as th

from chapter3_pollinators.env.actions import make_random_action
from chapter3_pollinators.env.game import PollinatorsGame

INCENTIVE_CYCLE = (
    [0, 0, 0, 0],
    [-1, -0.6, 0, 0.6],
    [0.6, 0.1, -0.25, 0.1],
    [0.73, -0.98, 0.02, 1.0],
)


def _reference_game(reference, seed: int):
    random.seed(seed)
    world = reference.gym_pollinator_game.gymDriver()
    world.reset()
    first = world.reset()
    return world, first


def _package_game(seed: int):
    random.seed(seed)
    game = PollinatorsGame()
    game.reset()
    first = game.reset()
    return game, first


def _cell_state(cells) -> list[tuple]:
    return [
        (c.x, c.y, c.declared_habitat, c.actual_habitat, c.was_pollinated, c.last_income)
        for c in cells
    ]


def _reference_cell_state(cells) -> list[tuple]:
    return [
        (c.x, c.y, c.bag_pointer_declared, c.bag_pointer_actual, c.was_pollinated, c.last_income)
        for c in cells
    ]


def _assert_observations_equal(package_obs, reference_obs) -> None:
    (p_local, p_global), (r_local, r_global) = package_obs, reference_obs
    assert len(p_local) == len(r_local) == 4
    for p_agent, r_agent in zip(p_local, r_local):
        assert np.array_equal(np.asarray(p_agent), np.asarray(r_agent))
        assert np.asarray(p_agent).dtype == np.asarray(r_agent).dtype
    assert np.array_equal(p_global, r_global)
    assert p_global.dtype == r_global.dtype


@pytest.mark.parametrize("seed", [0, 1, 7, 123])
def test_construction_consumes_the_generator_identically(reference, seed: int) -> None:
    world, reference_first = _reference_game(reference, seed)
    reference_state = random.getstate()
    game, package_first = _package_game(seed)
    assert random.getstate() == reference_state

    _assert_observations_equal(package_first, reference_first)
    reference_agents = world.agent_processor.all_agents
    package_agents = game.agent_processor.all_agents
    for p_agent, r_agent in zip(package_agents, reference_agents):
        assert p_agent.id == r_agent.id
        assert p_agent.alpha == r_agent.alpha
        assert p_agent.color == r_agent.color
        assert p_agent.color2 == r_agent.color2
        assert p_agent.trust_factor == r_agent.trust_factor
        assert p_agent.money == r_agent.money
        assert [(c.x, c.y) for c in p_agent.cells] == [
            (c.x, c.y) for c in r_agent.land_cells_owned
        ]
    assert _cell_state(game.grid.all_cells.values()) == _reference_cell_state(
        world.grid.all_cells.values()
    )


def _random_actions(module_fn, agents):
    return module_fn(agents)


def _constant_actions(agents, value: float):
    return [[th.tensor(value) for _ in range(n)] for n in (4, 5, 4, 3)]


def _drive_both(reference, seed: int, steps: int, action_mode: str) -> dict[str, int]:
    world, _ = _reference_game(reference, seed)
    reference_trace = []
    for step in range(steps):
        incentive = INCENTIVE_CYCLE[step % len(INCENTIVE_CYCLE)]
        agents = world.agent_processor.all_agents
        if action_mode == "random":
            action = reference.gym_pollinator_game.make_random_action(agents)
        else:
            action = _constant_actions(agents, float(action_mode))
        out = world.step(action, step % 3 == 0, incentive, False)
        reference_trace.append(
            (
                [[float(a) for a in row] for row in action],
                out,
                [a.money for a in agents],
                [a.color for a in agents],
                _reference_cell_state(world.grid.all_cells.values()),
                world.get_global_pollinators_reward(),
                sorted(world.polinattor_processor.all_polinattors),
                random.getstate(),
            )
        )

    game, _ = _package_game(seed)
    for step in range(steps):
        incentive = INCENTIVE_CYCLE[step % len(INCENTIVE_CYCLE)]
        agents = game.agent_processor.all_agents
        if action_mode == "random":
            action = make_random_action(agents)
        else:
            action = _constant_actions(agents, float(action_mode))
        out = game.step(action, step % 3 == 0, incentive)
        (
            r_action,
            r_out,
            r_money,
            r_colours,
            r_cells,
            r_global,
            r_register,
            r_state,
        ) = reference_trace[step]
        assert [[float(a) for a in row] for row in action] == r_action, step
        (p_obs, (p_reward, p_plain), p_done, p_info) = out
        (r_obs, (r_reward, r_plain), r_done, r_info) = r_out
        _assert_observations_equal(p_obs, r_obs)
        assert p_reward == r_reward, step
        assert [type(x) for x in p_reward] == [type(x) for x in r_reward], step
        assert p_plain == r_plain, step
        assert p_done == r_done and p_info is None and r_info is None
        assert [a.money for a in agents] == r_money, step
        assert [a.color for a in agents] == r_colours, step
        assert _cell_state(game.grid.all_cells.values()) == r_cells, step
        assert game.get_global_pollinators_reward() == r_global, step
        assert game.last_global_pollinators_reward == r_global, step
        assert sorted(game.pollinators_processor.all_pollinators) == r_register, step
        assert random.getstate() == r_state, step
    dones = sum(1 for entry in reference_trace if entry[1][2])
    resets = sum(
        1
        for before, after in zip(reference_trace, reference_trace[1:])
        for m0, m1 in zip(before[2], after[2])
        if m1 > m0 and m1 == 10_000
    )
    return {"dones": dones, "economy_resets": resets}


@pytest.mark.parametrize("seed", [0, 3])
def test_random_play_matches_reference(reference, seed: int) -> None:
    _drive_both(reference, seed, steps=250, action_mode="random")


def test_extinction_reset_matches_reference(reference) -> None:
    """All-zero habitat empties the pollinator register and re-seeds a random cell."""
    paths = _drive_both(reference, seed=5, steps=12, action_mode="0.0")
    assert paths["dones"] >= 1


def test_economy_reset_matches_reference(reference) -> None:
    """Full habitat earns nothing, so upkeep exhausts the starting gold."""
    paths = _drive_both(reference, seed=11, steps=140, action_mode="1.0")
    assert paths["economy_resets"] >= 1


def test_habitat_snapping_matches_reference(reference) -> None:
    random.seed(0)
    world = reference.gym_pollinator_game.gymDriver()
    reference_agent = world.agent_processor.all_agents[0]
    game = PollinatorsGame()
    package_agent = game.agent_processor.all_agents[0]
    values = np.concatenate(
        [np.linspace(0.0, 1.02, 2049, dtype=np.float32), np.arange(0, 1.06, 0.05, dtype=np.float32)]
    )
    for value in values:
        tensor = th.tensor(float(value))
        assert package_agent.select_action(tensor) == reference_agent.select_action(tensor)
