"""The package driver reproduces the reference experiment loop step for step.

``_reference_loop`` transcribes the body of the reference driver loop (the
reward-shape phase on counted steps 1,000-2,099 followed by Model-based ORAA)
with its print statements removed and the TPE sampler seeded.  The package
driver runs the same schedule; both start from the same seed and from injected
epsilon / counter values so that every schedule transition is crossed within a
few dozen steps.
"""

from __future__ import annotations

import random

import numpy as np
import optuna
import pytest
import torch as th

from chapter3_pollinators.env.game import PollinatorsGame
from chapter3_pollinators.experiment.driver import Driver, LoopState
from chapter3_pollinators.experiment.recorder import Recorder
from chapter3_pollinators.experiment.schedule import Schedule
from chapter3_pollinators.learning.pmaddpg import PMADDPG
from chapter3_pollinators.regulators.oraa import tpe_seed

pytestmark = pytest.mark.usefixtures("cuda_required")

REFERENCE_SCHEDULE = Schedule(
    curriculum=True,
    reward_shape_from=1000,
    reward_shape_until=2100,
    oraa_from=2100,
    oraa_networks="model",
)
ALPHAS = (0.4, 1.0, 0.5, 0.6)
TARGETS = [0.2, 0.1, 0.8, 0.9]


def _reference_loop(reference, monkeypatch, seed, steps, epsilon, counters, trials):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    world = reference.gym_pollinator_game.gymDriver()
    obs = world.reset()
    (obs, global_state) = world.reset()
    agents = world.agent_processor.all_agents
    maddpg = reference.MADDPG.MADDPG(4, 12, 1, 16, 9000, 10, agents, False)
    maddpg_fake = reference.MADDPG.MADDPG(4, 12, 1, 16, 9000, 10, agents, False)
    meta = reference.goverment_agent.goverment_agent(None, None, agents)
    incentive = [0, 0, 0, 0]
    for i, agent in enumerate(agents):
        agent.alpha = ALPHAS[i]
    meta.target = list(TARGETS)
    make_random_action = reference.gym_pollinator_game.make_random_action
    measure = reference_measure

    queue: list[int] = []
    real_create_study = optuna.create_study
    real_optimize = optuna.study.Study.optimize

    def seeded_create_study(*args, **kwargs):
        kwargs["sampler"] = optuna.samplers.TPESampler(seed=queue.pop(0))
        return real_create_study(*args, **kwargs)

    def limited_optimize(self, func, n_trials=None, **kwargs):
        return real_optimize(self, func, n_trials=trials, **kwargs)

    monkeypatch.setattr(optuna, "create_study", seeded_create_study)
    monkeypatch.setattr(optuna.study.Study, "optimize", limited_optimize)

    trace = []
    for _ in range(steps):
        randy_random = random.uniform(0, 1)
        if epsilon <= 0.4:
            counters += 1
        if counters >= 2100:
            queue[:] = [tpe_seed(seed, counters, i) for i in range(4)]
            incentive, new_ob = meta.optimise_incentives(
                obs, agents, maddpg_fake.actors, maddpg_fake.critics
            )
            obs = new_ob
        if epsilon > randy_random:
            action = maddpg.select_action(obs, agents)
            action_fake = maddpg_fake.select_action(obs, agents)
            flag_for_real_action = False
            action = make_random_action(agents)
            distance = measure(action, action_fake)
        else:
            flag_for_real_action = True
            action = maddpg.select_action(obs, agents)
            action_fake = maddpg_fake.select_action(obs, agents)
            distance = measure(action, action_fake)
        if epsilon <= 0.8 and counters <= 100:
            incentive = meta.distribute_incetive()
        if counters >= 100 and counters < 1000:
            incentive = meta.distribute_incentive_2()
        if counters >= 1000 and counters < 2100:
            incentive = []
            for i, a in enumerate(agents):
                average_action = sum([x.item() for x in action[i]]) / len(action[i])
                spending = 1 - (abs(meta.target[i] - average_action))
                incentive.append(spending)
        (obs_, global_state_), (reward, reward_without_incentive), done, _ = world.step(
            action, flag_for_real_action, incentive, False
        )
        trace.append(
            (
                counters,
                epsilon,
                flag_for_real_action,
                [float(x) for x in incentive],
                [float(x) for x in reward],
                [float(x) for x in reward_without_incentive],
                [[float(a) for a in row] for row in action],
                distance,
            )
        )
        maddpg.memory.push(obs, action, obs_, reward, global_state, global_state_)
        maddpg_fake.memory.push(obs, action, obs_, reward, global_state, global_state_)
        obs = obs_
        global_state = global_state_
        maddpg.update_policy(agents, epsilon)
        maddpg_fake.update_policy(agents, epsilon)
        maddpg.episode_done += 1
        maddpg_fake.episode_done += 1
        if maddpg.episode_done >= 16:
            if epsilon > 0.4:
                epsilon -= 1e-4
            else:
                epsilon = 0.4
    monkeypatch.setattr(optuna, "create_study", real_create_study)
    monkeypatch.setattr(optuna.study.Study, "optimize", real_optimize)
    return trace, maddpg, maddpg_fake, random.getstate()


def reference_measure(a, b):
    distances = []
    for x, y in zip(a, b):
        for z, c in zip(x, y):
            distances.append(abs(round(z.item(), 1) - round(c.item(), 1)))
    return sum(distances)


def _package_loop(seed, steps, epsilon, counters, trials):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    game = PollinatorsGame()
    game.reset()
    obs, global_state = game.reset()
    agents = game.agent_processor.all_agents
    real = PMADDPG(4, 12, 1, 16, 9000, 10, agents)
    model = PMADDPG(4, 12, 1, 16, 9000, 10, agents)
    for i, agent in enumerate(agents):
        agent.alpha = ALPHAS[i]
    recorder = Recorder()
    state = LoopState(obs=obs, global_state=global_state, epsilon=epsilon, counters=counters)
    driver = Driver(
        game, real, model, state, REFERENCE_SCHEDULE, seed=seed, targets=TARGETS,
        recorder=recorder, oraa_trials=trials,
    )
    for _ in range(steps):
        driver.step()
    return recorder.arrays(), real, model, random.getstate()


SEGMENTS = {
    "policy0": (1, 0, 30),
    "penalising": (0.8, 0, 24),
    "around_counter_100": (0.4, 95, 24),
    "around_counter_1000": (0.4, 990, 24),
    "around_counter_2100": (0.4, 2096, 6),
}


@pytest.mark.parametrize("segment", list(SEGMENTS))
def test_driver_matches_reference_loop(reference, monkeypatch, segment: str) -> None:
    epsilon, counters, steps = SEGMENTS[segment]
    seed = 21
    trials = 25
    trace, r_real, r_model, r_state = _reference_loop(
        reference, monkeypatch, seed, steps, epsilon, counters, trials
    )
    records, p_real, p_model, p_state = _package_loop(seed, steps, epsilon, counters, trials)
    assert p_state == r_state
    for t, (counter, eps, greedy, incentive, reward, plain, action, distance) in enumerate(trace):
        assert records["counter"][t] == counter, t
        assert records["epsilon"][t] == eps, t
        assert bool(records["greedy"][t]) == greedy, t
        assert records["incentive"][t].tolist() == incentive, t
        assert records["reward"][t].tolist() == reward, t
        assert records["reward_plain"][t].tolist() == plain, t
        assert records["mean_action"][t].tolist() == [sum(row) / len(row) for row in action], t
        assert records["model_distance_taken"][t] == distance, t
    for p_net, r_net in zip(
        p_real.actors + p_real.critics + p_model.actors + p_model.critics,
        r_real.actors + r_real.critics + r_model.actors + r_model.critics,
    ):
        for (key, p_value), r_value in zip(p_net.state_dict().items(), r_net.state_dict().values()):
            assert th.equal(p_value, r_value), key


def test_segments_cross_the_intended_phases() -> None:
    """Sanity check of the segment choice: each regulator acts at least once."""
    _, _, steps = SEGMENTS["around_counter_100"]
    assert 95 + steps > 100
    assert 990 + SEGMENTS["around_counter_1000"][2] > 1000
    assert 2096 + SEGMENTS["around_counter_2100"][2] > 2100
