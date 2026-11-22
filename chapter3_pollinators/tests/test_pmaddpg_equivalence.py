"""The ported P-MADDPG learner reproduces the reference learner bit for bit.

Both learners are built from the same torch seed, fed identical transitions
from the package game, and updated from identical generator states; after every
update the losses, every network parameter, the Adam moments and the Python
generator state must match exactly.
"""

from __future__ import annotations

import random
import types

import numpy as np
import pytest
import torch as th

from chapter3_pollinators.env.actions import make_random_action
from chapter3_pollinators.env.game import PollinatorsGame
from chapter3_pollinators.learning.pmaddpg import PMADDPG

pytestmark = pytest.mark.usefixtures("cuda_required")


def _build_pair(reference, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    game = PollinatorsGame()
    game.reset()
    obs, global_state = game.reset()
    agents = game.agent_processor.all_agents
    th.manual_seed(seed)
    ref = reference.MADDPG.MADDPG(4, 12, 1, 16, 9000, 10, _reference_view(agents), False)
    th.manual_seed(seed)
    port = PMADDPG(4, 12, 1, 16, 9000, 10, agents)
    return game, obs, global_state, agents, ref, port


def _reference_view(agents):
    """The reference learner reads a landowner's cells as ``land_cells_owned``."""
    return [types.SimpleNamespace(land_cells_owned=a.cells) for a in agents]


def _networks(learner):
    return learner.actors + learner.critics + learner.actors_target + learner.critics_target


def _assert_same_parameters(port, ref) -> None:
    for p_net, r_net in zip(_networks(port), _networks(ref)):
        p_state, r_state = p_net.state_dict(), r_net.state_dict()
        assert p_state.keys() == r_state.keys()
        for key in p_state:
            assert th.equal(p_state[key], r_state[key]), key


def _assert_same_optimisers(port, ref) -> None:
    for p_opt, r_opt in zip(
        port.critic_optimizer + port.actor_optimizer, ref.critic_optimizer + ref.actor_optimizer
    ):
        p_state, r_state = p_opt.state_dict()["state"], r_opt.state_dict()["state"]
        assert p_state.keys() == r_state.keys()
        for key in p_state:
            for name in ("exp_avg", "exp_avg_sq", "step"):
                assert th.equal(p_state[key][name], r_state[key][name])


@pytest.mark.parametrize("seed", [0, 4])
def test_networks_are_initialised_identically(reference, seed: int) -> None:
    _, _, _, _, ref, port = _build_pair(reference, seed)
    _assert_same_parameters(port, ref)
    assert [type(n).__name__ for n in port.actors] == ["Actor"] * 4
    assert [n.FC4.out_features for n in port.actors] == [4, 5, 4, 3]
    assert [n.FC1.in_features for n in port.critics] == [36, 37, 36, 35]


def test_greedy_actions_match(reference) -> None:
    game, obs, _, agents, ref, port = _build_pair(reference, 2)
    p_actions = port.select_action(obs, agents)
    r_actions = ref.select_action(obs, _reference_view(agents))
    assert [[float(a) for a in row] for row in p_actions] == [
        [float(a) for a in row] for row in r_actions
    ]
    assert all(a.dtype == th.float32 and a.dim() == 0 for row in p_actions for a in row)


def test_updates_match_reference_through_a_target_copy(reference) -> None:
    """146 steps: 16 without training, then 130 updates, crossing the hard copy at 125."""
    game, obs, global_state, agents, ref, port = _build_pair(reference, 9)
    steps = 146
    copies_seen = 0
    for step in range(steps):
        if step % 4 == 3:
            action = port.select_action(obs, agents)
            ref_action = ref.select_action(obs, _reference_view(agents))
            assert [[float(a) for a in r] for r in action] == [
                [float(a) for a in r] for r in ref_action
            ]
        else:
            action = make_random_action(agents)
        incentive = [0.1 * (step % 5) - 0.2, 0.0, -0.3, 0.25]
        (next_obs, next_global), (reward, _), _, _ = game.step(action, True, incentive)
        ref.memory.push(obs, action, next_obs, reward, global_state, next_global)
        port.memory.push(obs, action, next_obs, reward, global_state, next_global)
        obs, global_state = next_obs, next_global

        before = random.getstate()
        r_closs, r_aloss = ref.update_policy(_reference_view(agents), 0.9)
        after = random.getstate()
        random.setstate(before)
        p_closs, p_aloss = port.update_policy(agents, 0.9)
        assert random.getstate() == after, step
        if r_closs is None:
            assert p_closs is None and p_aloss is None
        else:
            assert [float(x) for x in p_closs] == [float(x) for x in r_closs], step
            assert [float(x) for x in p_aloss] == [float(x) for x in r_aloss], step
        ref.episode_done += 1
        port.episode_done += 1
        assert port.steps_done == ref.steps_done
        if port.steps_done and port.steps_done % 250 == 0:
            copies_seen += 1
        _assert_same_parameters(port, ref)
    _assert_same_optimisers(port, ref)
    assert copies_seen == 1
