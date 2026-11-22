"""The reference implementation can be imported and stepped on this machine."""

from __future__ import annotations

import random


def test_reference_game_constructs_and_steps(reference) -> None:
    random.seed(0)
    world = reference.gym_pollinator_game.gymDriver()
    world.reset()
    obs, global_state = world.reset()
    assert len(obs) == 4
    assert global_state.shape == (2, 4, 4)
    agents = world.agent_processor.all_agents
    assert [len(a.land_cells_owned) for a in agents] == [4, 5, 4, 3]
    actions = reference.gym_pollinator_game.make_random_action(agents)
    (obs2, gs2), (reward, plain), done, _ = world.step(actions, False, [0, 0, 0, 0], False)
    assert len(reward) == 4 and len(plain) == 4
