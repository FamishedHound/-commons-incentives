"""Applying the landowners' actions, and the uniform exploration action."""

from __future__ import annotations

import random

import torch as th

from chapter3_pollinators.env.pollination import PollinatorsProcessor


class ActionProcessor:
    """Hands each landowner its action vector."""

    def __init__(self, all_agents, pollinator_processor: PollinatorsProcessor) -> None:
        self.all_agents = all_agents
        self.action_space = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.pollinators_processor = pollinator_processor
        self.epsilon = 1

    def all_agents_make_a_move(self, actions) -> list[list[int]]:
        lands = []
        for i, agent in enumerate(self.all_agents):
            lands.append(agent.make_a_decision(actions[i], self.epsilon))
        return lands


def make_random_action(all_agents) -> list[list[th.Tensor]]:
    """One uniform random action per cell, ``U(0, 1.02)``, as a 0-d float32 tensor.

    Values above 1 snap to full habitat, so the eleven habitat levels are hit
    with nearly equal probability.
    """
    action = []
    for agent in all_agents:
        agent_actions = [
            th.tensor(random.choice([random.uniform(0, 102)]) / 100)
            for _ in range(len(agent.cells))
        ]
        action.append(agent_actions)
    return action


def measure_sum_of_distances_between_matrices(a, b) -> float:
    """Sum over agents and cells of ``|round(a, 1) - round(b, 1)|``."""
    distances = []
    for x, y in zip(a, b):
        for z, c in zip(x, y):
            distances.append(abs(round(z.item(), 1) - round(c.item(), 1)))
    return sum(distances)
