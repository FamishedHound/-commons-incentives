"""The Pollinators' Game environment.

Four landowners share a 4 x 4 landscape (:mod:`.properties`).  Each step every
landowner declares a habitat share for each of its cells (a continuous action in
[0, 1] snapped to the nearest 0.1); the habitat is in place at once.  Whether a
cell is pollinated is then drawn from the pollination model
(:mod:`.pollination`), pollinated cells earn crop income ``(100 - h)`` gold
coins (:mod:`.economy`), and landowner ``i`` receives

    r_i = max(0, alpha_i * mean_{c in C_i}(income_c / 100 - 0.4)
                 + (1 - alpha_i) * G / 2 + incentive_i),

    G   = sum_{c pollinated} h_c / 16,

where ``h_c`` is the habitat fraction of cell ``c`` and ``incentive_i`` is the
regulator's scalar incentive for that landowner (negative values are penalties).
The reward without the incentive and before the clip, ``alpha_i * ... +
(1 - alpha_i) * G / 2``, is returned as well.

Observations.  Landowner ``i`` observes a ``2 x 4 x 4`` array: its own cells'
habitat and its incentive written on its own cells (every other cell is zero).
The global state is the same two layers over the whole grid (habitat of every
cell and every landowner's incentive on its cells).  Arrays are built on
``(x, y)`` indices and rotated by ``numpy.rot90``.

Extinction.  If no cell holds any habitat after the declarations, the
pollinators have died out: :meth:`reset` re-seeds a random cell with 10 %
habitat and ``done`` is returned as ``True``.  A run is one continuous sequence
of steps; ``done`` does not end it.

The habitat and economy managers work with the pollinator register created
together with the game, in which every cell that has held habitat stays
registered; :meth:`reset` creates the fresh register used by the extinction
check.  Cells without habitat contribute nothing to the pollination sums, so a
registered cell without habitat has no effect on pollination.
"""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces

from chapter3_pollinators.env.actions import ActionProcessor
from chapter3_pollinators.env.constants import GRID_SIZE, LAND_UPKEEP, NUMBER_OF_AGENTS
from chapter3_pollinators.env.economy import EconomyManager
from chapter3_pollinators.env.habitat import EnvironmentalManager
from chapter3_pollinators.env.land import Grid
from chapter3_pollinators.env.pollination import PollinatorsProcessor
from chapter3_pollinators.env.properties import AgentProcessor


class PollinatorsGame(gymnasium.Env):
    """The Pollinators' Game: ``step(action, greedy, incentive)``."""

    def __init__(self) -> None:
        self.grid = Grid()
        self.pollinators_processor = PollinatorsProcessor(grid=self.grid)
        self.n_agents = NUMBER_OF_AGENTS
        self.agent_processor = AgentProcessor(grid=self.grid)
        self.agent_processor.clear_empty_agents()
        self.action_processor = ActionProcessor(
            all_agents=self.agent_processor.all_agents,
            pollinator_processor=self.pollinators_processor,
        )
        self.observation_space = spaces.Box(0, 1, shape=[1])
        self.action_space = spaces.Box(-1, 1, shape=[4])
        self.environmental_manager = EnvironmentalManager(self.pollinators_processor)
        self.economy_manager = EconomyManager(
            self.agent_processor.all_agents, self.pollinators_processor
        )
        self.counter = 0
        self.index = 0
        self.last_global_pollinators_reward = 0.0

    @property
    def agents(self):
        return self.agent_processor.all_agents

    def reset(self):  # type: ignore[override]
        """Re-seed a random cell and start a fresh register for the extinction check."""
        self.pollinators_processor = PollinatorsProcessor(grid=self.grid)
        self.n_agents = len(self.agent_processor.all_agents)
        return self.create_observation([0, 0, 0, 0])

    def get_global_pollinators_reward(self) -> float:
        """``G = sum_{c pollinated} h_c / 16`` over the whole grid."""
        pollinators_reward = 0
        for land in self.grid.all_cells.values():
            if land.was_pollinated:
                pollinators_reward += (land.actual_habitat / 100) / len(self.grid.all_cells)
        return pollinators_reward

    def _get_reward(self, incentive):
        agents_rewards = []
        global_pollinators_reward = self.get_global_pollinators_reward()
        self.last_global_pollinators_reward = global_pollinators_reward
        reward_without_incentive = []
        for j, agent in enumerate(self.agent_processor.all_agents):
            cumulative_reward = 0
            for land in agent.cells:
                single_reward = land.last_income / 100 - LAND_UPKEEP / 100
                cumulative_reward += single_reward
            money_reward = cumulative_reward / len(agent.cells)
            reward_internal = (
                agent.alpha * (money_reward) + (1 - agent.alpha) * global_pollinators_reward / 2
            )
            final_reward = reward_internal + incentive[j]
            if final_reward < 0:
                final_reward = 0
            agents_rewards.append(final_reward)
            reward_without_incentive.append(reward_internal)
        return agents_rewards, reward_without_incentive

    def create_observation(self, incentive):
        """Per-landowner observations and the global state, with ``incentive`` written in."""
        board_size = GRID_SIZE
        land_per_agent_obs = []
        for j, agent in enumerate(self.agent_processor.all_agents):
            single_agent_obs = []
            local_declared = np.zeros((board_size, board_size))
            local_actual = np.zeros((board_size, board_size))
            incentive_np = np.zeros((board_size, board_size))
            for land in agent.cells:
                local_declared[land.x, land.y] = land.declared_habitat / 100
                local_actual[land.x, land.y] = land.actual_habitat / 100
                incentive_np[land.x, land.y] = incentive[j]
            local_declared = np.rot90(local_declared)
            local_actual = np.rot90(local_actual)
            incentive_np = np.rot90(incentive_np)
            single_agent_obs.append(np.array([local_actual, incentive_np]))
            land_per_agent_obs.append(single_agent_obs)
        _, global_actual, incentive_global = self.get_global_state_without_position(
            board_size, incentive
        )
        global_actual = np.rot90(global_actual)
        incentive_global = np.rot90(incentive_global)
        global_obs = np.array([global_actual, incentive_global])
        return land_per_agent_obs, global_obs

    def get_global_state_without_position(self, board_size: int, incentive):
        declared = np.zeros((board_size, board_size))
        actual = np.zeros((board_size, board_size))
        incentive_global = np.zeros((board_size, board_size))
        for j, agent in enumerate(self.agent_processor.all_agents):
            for land in agent.cells:
                declared[land.x, land.y] = land.declared_habitat / 100
                actual[land.x, land.y] = land.actual_habitat / 100
                incentive_global[land.x, land.y] = incentive[j]
        return declared, actual, incentive_global

    def step(self, action=None, greedy=None, incentive=None):  # type: ignore[override]
        """Apply the declarations, draw pollination, pay income; return obs and rewards.

        Returns ``((local_obs, global_state), (reward, reward_without_incentive),
        done, None)``.  ``greedy`` only labels the step (greedy or exploratory).
        """
        if isinstance(action, np.ndarray):
            action = [action.tolist()]
        self.action_processor.all_agents_make_a_move(action)
        self.pollinators_processor.clear_pollinators()
        self.environmental_manager.process_declared_lands()
        self.economy_manager.deduce_land_fee()

        done = False
        if self.pollinators_processor.check_for_failed_pollinators_during_exploration():
            self.reset()
            done = True

        reward, reward_without_incentive = self._get_reward(incentive)
        observation = self.create_observation(incentive)
        return observation, (reward, reward_without_incentive), done, None
