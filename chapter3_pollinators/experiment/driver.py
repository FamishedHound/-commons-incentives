"""The experiment loop: one environment step of landowners, regulator and learners.

Each call of :meth:`Driver.step` performs, in order:

1. draw the exploration number ``u ~ U(0, 1)`` and advance the counter when
   epsilon is at or below 0.4;
2. ORAA (if active) chooses this step's incentives and writes them into the
   landowners' observations; LToS (if active) fixes this step's sharing
   weights and writes the incoming weights into the observations;
3. the landowners act: uniform random actions if ``epsilon > u``, otherwise
   their actors' greedy actions;
4. the curriculum rules or the reward-shape baseline (if active) set the
   incentives from the current habitat or action;
5. the game steps with the incentives; LToS shares the rewards;
6. the transition is stored, every learner is updated once and epsilon decays.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chapter3_pollinators.env.actions import (
    make_random_action,
    measure_sum_of_distances_between_matrices,
)
from chapter3_pollinators.experiment.recorder import Recorder
from chapter3_pollinators.experiment.schedule import (
    BRANCH_COUNTER,
    CURRICULUM_EPSILON,
    EPSILON_FLOOR,
    PENALISING_LAST_COUNTER,
    PHASE_CODES,
    REWARDING_FIRST_COUNTER,
    Schedule,
    next_epsilon,
)
from chapter3_pollinators.regulators.curriculum import penalising_incentive, rewarding_incentive
from chapter3_pollinators.regulators.oraa import N_TRIALS, ORAA, tpe_seed
from chapter3_pollinators.regulators.reward_shape import reward_shape_incentive

NO_INCENTIVE = (0, 0, 0, 0)


@dataclass
class LoopState:
    """The loop variables carried from one step to the next."""

    obs: Any
    global_state: np.ndarray
    incentive: list = field(default_factory=lambda: [0, 0, 0, 0])
    epsilon: float = 1
    counters: int = 0
    step: int = 0
    greedy: bool = False


def _mean(values) -> float:
    return sum([x.item() for x in values]) / len(values)


def declared_means(agents, action) -> list[float]:
    """Mean declared habitat share per landowner for ``action`` (after the 0.1 snap)."""
    return [
        sum(agent.select_action(a) for a in row) / 100 / len(row)
        for agent, row in zip(agents, action)
    ]


class Driver:
    """Steps one run: a game, the landowners' learner, an optional model and a regulator."""

    def __init__(
        self,
        game,
        real,
        model,
        state: LoopState,
        schedule: Schedule,
        seed: int,
        targets=(0.35, 0.35, 0.35, 0.35),
        recorder: Recorder | None = None,
        oraa_trials: int = N_TRIALS,
        sharing=None,
    ) -> None:
        self.game = game
        self.real = real
        self.model = model
        self.state = state
        self.schedule = schedule
        self.seed = seed
        self.targets = list(targets)
        self.recorder = recorder
        self.oraa = (
            ORAA(self.targets, n_trials=oraa_trials, bound=schedule.oraa_bound)
            if schedule.oraa_from is not None
            else None
        )
        self.sharing = sharing
        if schedule.ltos_from is not None and sharing is None:
            raise ValueError("an LToS schedule needs a sharing scheme")
        if schedule.oraa_from is not None and schedule.oraa_networks == "model" and model is None:
            raise ValueError("Model-based ORAA needs a model learner")

    @property
    def agents(self):
        return self.game.agent_processor.all_agents

    def run_until(self, last_counter: int) -> None:
        """Step until the step with counter ``last_counter`` has been taken."""
        while self.state.counters < last_counter:
            self.step()

    def step(self) -> None:
        s = self.state
        agents = self.agents
        randy_random = random.uniform(0, 1)
        if s.epsilon <= EPSILON_FLOOR:
            s.counters += 1

        phase = "none"
        oraa_best = None
        if self.schedule.oraa_active(s.counters):
            networks = self.model.actors if self.schedule.oraa_networks == "model" else self.real.actors
            seeds = [tpe_seed(self.seed, s.counters, i) for i in range(len(agents))]
            s.incentive, s.obs = self.oraa.optimise_incentives(s.obs, agents, networks, seeds)
            oraa_best = list(self.oraa.last_best_values)
            phase = "oraa"

        sharing_active = self.sharing is not None and self.schedule.ltos_active(s.counters)
        acting_obs = s.obs
        weights = None
        if sharing_active:
            weights = self.sharing.begin_step(s)
            acting_obs = self.sharing.acting_observation(s.obs, weights)
            phase = "ltos"

        real_greedy = self.real.select_action(acting_obs, agents)
        model_action = (
            self.model.select_action(acting_obs, agents) if self.model is not None else None
        )
        if s.epsilon > randy_random:
            s.greedy = False
            action = make_random_action(agents)
        else:
            s.greedy = True
            action = real_greedy

        if self.schedule.curriculum:
            if s.epsilon <= CURRICULUM_EPSILON and s.counters <= PENALISING_LAST_COUNTER:
                s.incentive = penalising_incentive(agents)
                phase = "penalising"
            if REWARDING_FIRST_COUNTER <= s.counters < BRANCH_COUNTER:
                s.incentive = rewarding_incentive(agents)
                phase = "rewarding"
        if self.schedule.reward_shape_active(s.counters):
            s.incentive = reward_shape_incentive(action, self.targets)
            phase = "reward_shape"

        paid = list(NO_INCENTIVE) if sharing_active else s.incentive
        mean_declared = declared_means(agents, action)
        (next_obs, next_global), (reward, reward_plain), done, _ = self.game.step(
            action, s.greedy, paid
        )

        shaped = None
        transfer = None
        if sharing_active:
            shaped, transfer = self.sharing.share(reward, weights)
            self.real.memory.push_experience(
                self.sharing.transition(
                    s.obs, action, next_obs, reward, shaped, s.global_state, next_global, weights
                )
            )
        else:
            self.real.memory.push(s.obs, action, next_obs, reward, s.global_state, next_global)
        if self.model is not None:
            self.model.memory.push(s.obs, action, next_obs, reward, s.global_state, next_global)

        if self.recorder is not None:
            self._record(
                s, phase, done, paid, reward, reward_plain, shaped, action, mean_declared,
                real_greedy, model_action, oraa_best, weights, transfer,
            )

        s.obs = next_obs
        s.global_state = next_global
        extractor = self.sharing.extractor if sharing_active else None
        self.real.update_policy(agents, s.epsilon, extractor=extractor)
        if self.model is not None:
            self.model.update_policy(agents, s.epsilon)
        self.real.episode_done += 1
        if self.model is not None:
            self.model.episode_done += 1
        if sharing_active:
            self.sharing.end_step(s)
        s.epsilon = next_epsilon(s.epsilon, self.real.episode_done)
        s.step += 1

    def _record(
        self, s, phase, done, paid, reward, reward_plain, shaped, action, mean_declared,
        real_greedy, model_action, oraa_best, weights, transfer,
    ) -> None:
        model_fields = {}
        if model_action is not None:
            model_fields = {
                "model_distance_taken": measure_sum_of_distances_between_matrices(action, model_action),
                "model_distance_greedy": measure_sum_of_distances_between_matrices(
                    real_greedy, model_action
                ),
                "real_greedy_mean": [_mean(row) for row in real_greedy],
                "model_mean": [_mean(row) for row in model_action],
            }
        self.recorder.append(
            step=s.step,
            counter=s.counters,
            epsilon=s.epsilon,
            greedy=s.greedy,
            phase=PHASE_CODES[phase],
            done=done,
            incentive=[float(x) for x in paid],
            reward=[float(x) for x in reward],
            reward_plain=[float(x) for x in reward_plain],
            shaped_reward=shaped,
            mean_action=[_mean(row) for row in action],
            mean_declared=mean_declared,
            global_reward=float(self.game.last_global_pollinators_reward),
            oraa_best=oraa_best,
            sharing_weights=None if weights is None else np.asarray(weights, dtype=np.float64),
            transfer_volume=transfer,
            **model_fields,
        )
