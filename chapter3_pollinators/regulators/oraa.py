"""ORAA, the Online Regulatory Agent Algorithm.

At every step and for every landowner ``i`` ORAA runs a fresh Optuna study
(the default TPE sampler, seeded, 500 trials) over one scalar incentive
``x in [-b, b]``, with the bound ``b = 1`` unless set otherwise.  A trial writes ``x`` on every cell of the landowner's
property in the incentive layer of its current observation (keeping its habitat
layer), feeds that observation to a simulation actor and scores

    | target_i - mean_c actor_i(observation with x)_c |,

with no cost term.  The incentive paid is the best ``x``; it is added once to
the landowner's reward (``incentive_i`` in the game's reward) and written into
the landowner's observation for this step.

The simulation actors are either the landowners' own actors (Omniscient) or the
actors of a second, independently initialised P-MADDPG trained on the same
transitions (Model-based).  The critics are not used.

Each study's sampler is seeded with :func:`tpe_seed` from the run seed, the
counted step and the landowner, so ORAA is reproducible and draws from none of
the global generators.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import optuna
import torch as th

#: Trials per study.
N_TRIALS = 500
#: Bound of the incentive search, ``x in [-DEFAULT_BOUND, DEFAULT_BOUND]``.
DEFAULT_BOUND = 1.0
INCENTIVE_LOW, INCENTIVE_HIGH = -DEFAULT_BOUND, DEFAULT_BOUND
_ORAA_STREAM = 0x0A2A


def tpe_seed(run_seed: int, counter: int, agent_index: int) -> int:
    """The TPE sampler seed for one study, a 32-bit integer."""
    sequence = np.random.SeedSequence([int(run_seed), int(counter), int(agent_index), _ORAA_STREAM])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def get_agents_land_positions(agent, other_array: np.ndarray, incentive: float) -> np.ndarray:
    """An incentive layer with ``incentive`` on the landowner's cells, oriented like the game's."""
    incentive_arr = np.zeros_like(other_array)
    for land in agent.cells:
        incentive_arr[land.x, land.y] = incentive
    return np.rot90(incentive_arr)


class ORAA:
    """The regulatory agent's per-step incentive search."""

    def __init__(self, target: Sequence[float], n_trials: int = N_TRIALS, bound: float = DEFAULT_BOUND) -> None:
        self.target = list(target)
        self.n_trials = n_trials
        self.bound = float(bound)
        self.agent_networks = None
        self.counter_networks = 0
        self.interpreted_agent = None
        self.interpreted_obs: np.ndarray | None = None
        self.decisions: dict[int, th.Tensor] = {}
        self.new_state: th.Tensor | None = None
        self.last_best_values: list[float] = []

    def optimise_incentives(self, obs, agents, agent_networks, sampler_seeds: Sequence[int]):
        """Best incentive per landowner, and the observation with those incentives written in."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.agent_networks = agent_networks
        final_incentive = []
        new_observation = []
        best_values = []
        for i, agent in enumerate(agents):
            self.counter_networks = i
            self.interpreted_obs = np.array(obs)[self.counter_networks][0][0]
            self.interpreted_agent = agent
            self._prepare_device_inputs(agent)
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=int(sampler_seeds[i])),
            )
            study.optimize(self.objective, n_trials=self.n_trials)
            best_incentive = study.best_params["x"]
            final_incentive.append(best_incentive)
            best_values.append(float(study.best_value))
            new_obs_incentive = get_agents_land_positions(agent, self.interpreted_obs, best_incentive)
            new_observation.append(np.stack((self.interpreted_obs, new_obs_incentive)))
        self.last_best_values = best_values
        stacked = np.stack(list(np.expand_dims(x, axis=0) for x in new_observation))
        return final_incentive, stacked.tolist()

    def _prepare_device_inputs(self, agent) -> None:
        """The landowner's habitat layer and the mask of its cells, on the GPU, once per study."""
        self._habitat = th.from_numpy(self.interpreted_obs.copy()).float().unsqueeze(0).cuda()
        cells = get_agents_land_positions(agent, self.interpreted_obs, 1.0) != 0
        self._cells = th.from_numpy(cells.copy()).cuda()

    def objective(self, trial: optuna.Trial) -> float:
        """``|target_i - mean(actor_i(observation with x on the landowner's cells))|``.

        The observation is assembled on the GPU: the habitat layer, and an
        incentive layer holding ``x`` (as float32) on the landowner's cells and
        zero elsewhere.
        """
        x = trial.suggest_float("x", -self.bound, self.bound)
        multiplier = 1
        value = th.full((), x * multiplier, dtype=th.float32, device=self._cells.device)
        incentive_layer = th.where(self._cells, value, 0.0)
        self.new_state = th.cat([self._habitat, incentive_layer.unsqueeze(0)], dim=0).unsqueeze(0)
        self.decisions[self.counter_networks] = (
            self.agent_networks[self.counter_networks](self.new_state).squeeze().data
        )
        return abs(
            self.target[self.counter_networks] - self.decisions[self.counter_networks].mean().item()
        )
