"""Full-state checkpoints of a run.

A checkpoint holds everything a run needs to continue exactly as if it had not
stopped: the game (every cell, landowner, register and manager), the loop
variables (current observation, global state, last incentive, epsilon, the step
counters), every network, target network and Adam state of the landowners'
learner (and optionally of the model learner), the replay memory, and the state
of Python's, NumPy's and torch's generators.  The replay memory is stored as
arrays and rebuilt into the same Python objects on loading.

Checkpoints are pickle-based files that the campaign writes and reads itself;
load only checkpoints produced by this package.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch as th

from chapter3_pollinators.env.properties import PROPERTY_SIZES
from chapter3_pollinators.experiment import atomic
from chapter3_pollinators.experiment.seeding import generator_states, restore_generator_states
from chapter3_pollinators.learning.memory import Experience, ReplayMemory
from chapter3_pollinators.learning.pmaddpg import PMADDPG

CHECKPOINT_FORMAT = 1
_ENV_OBS, _LIST_OBS = 0, 1


def learner_state(learner: PMADDPG) -> dict:
    return {
        "actors": [n.state_dict() for n in learner.actors],
        "critics": [n.state_dict() for n in learner.critics],
        "actors_target": [n.state_dict() for n in learner.actors_target],
        "critics_target": [n.state_dict() for n in learner.critics_target],
        "actor_optimizer": [o.state_dict() for o in learner.actor_optimizer],
        "critic_optimizer": [o.state_dict() for o in learner.critic_optimizer],
        "steps_done": learner.steps_done,
        "episode_done": learner.episode_done,
    }


def load_learner_state(learner: PMADDPG, state: dict) -> None:
    for name in ("actors", "critics", "actors_target", "critics_target"):
        for network, values in zip(getattr(learner, name), state[name]):
            network.load_state_dict(values)
    for name in ("actor_optimizer", "critic_optimizer"):
        for optimizer, values in zip(getattr(learner, name), state[name]):
            optimizer.load_state_dict(values)
    learner.steps_done = state["steps_done"]
    learner.episode_done = state["episode_done"]


def _pack_obs(obs) -> tuple[np.ndarray, int]:
    kind = _ENV_OBS if isinstance(obs[0], list) and isinstance(obs[0][0], np.ndarray) else _LIST_OBS
    return np.asarray(obs, dtype=np.float64).reshape(4, 2, 4, 4), kind


def _unpack_obs(values: np.ndarray, kind: int):
    if kind == _ENV_OBS:
        return [[values[agent].copy()] for agent in range(values.shape[0])]
    return values.reshape(values.shape[0], 1, 2, 4, 4).tolist()


def compact_memory(memory: ReplayMemory) -> dict:
    """The replay memory as arrays (see :func:`restore_memory`)."""
    n = len(memory.memory)
    states = np.empty((n, 4, 2, 4, 4))
    next_states = np.empty((n, 4, 2, 4, 4))
    state_kind = np.empty(n, dtype=np.int8)
    next_kind = np.empty(n, dtype=np.int8)
    actions = np.empty((n, sum(PROPERTY_SIZES)), dtype=np.float32)
    rewards = np.empty((n, 4))
    reward_is_int = np.zeros((n, 4), dtype=np.bool_)
    global_states = np.empty((n, 2, 4, 4))
    next_global_states = np.empty((n, 2, 4, 4))
    for k, experience in enumerate(memory.memory):
        if type(experience) is not Experience:
            raise TypeError("only landowner transitions can be checkpointed")
        states[k], state_kind[k] = _pack_obs(experience.states)
        next_states[k], next_kind[k] = _pack_obs(experience.next_states)
        actions[k] = [float(a) for row in experience.actions for a in row]
        rewards[k] = experience.rewards
        reward_is_int[k] = [type(r) is int for r in experience.rewards]
        global_states[k] = experience.global_state
        next_global_states[k] = experience.next_global_state
    return {
        "capacity": memory.capacity,
        "position": memory.position,
        "states": states,
        "next_states": next_states,
        "state_kind": state_kind,
        "next_kind": next_kind,
        "actions": actions,
        "rewards": rewards,
        "reward_is_int": reward_is_int,
        "global_states": global_states,
        "next_global_states": next_global_states,
    }


def restore_memory(blob: dict) -> ReplayMemory:
    memory = ReplayMemory(int(blob["capacity"]))
    bounds = np.cumsum((0,) + PROPERTY_SIZES)
    for k in range(blob["states"].shape[0]):
        flat = blob["actions"][k]
        actions = [
            [th.tensor(float(v)) for v in flat[bounds[a] : bounds[a + 1]]] for a in range(4)
        ]
        rewards = [
            int(r) if is_int else float(r)
            for r, is_int in zip(blob["rewards"][k], blob["reward_is_int"][k])
        ]
        memory.memory.append(
            Experience(
                _unpack_obs(blob["states"][k], int(blob["state_kind"][k])),
                actions,
                _unpack_obs(blob["next_states"][k], int(blob["next_kind"][k])),
                rewards,
                blob["global_states"][k].copy(),
                blob["next_global_states"][k].copy(),
            )
        )
    memory.position = int(blob["position"])
    return memory


def _same_memory(a: ReplayMemory, b: ReplayMemory) -> bool:
    if a.position != b.position or len(a) != len(b):
        return False
    return all(x.actions is y.actions and x.rewards is y.rewards for x, y in zip(a.memory, b.memory))


def save_checkpoint(path: Path, run, include_model: bool, meta: dict | None = None) -> None:
    """Write ``run`` to ``path`` (atomically)."""
    if include_model and run.model is not None and not _same_memory(run.real.memory, run.model.memory):
        raise ValueError("the model learner is expected to hold the same transitions")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "meta": dict(meta or {}),
        "seed": run.seed,
        "game": pickle.dumps(run.game, protocol=pickle.HIGHEST_PROTOCOL),
        "loop": pickle.dumps(run.state, protocol=pickle.HIGHEST_PROTOCOL),
        "real": learner_state(run.real),
        "model": learner_state(run.model) if include_model and run.model is not None else None,
        "memory": compact_memory(run.real.memory),
        "generators": generator_states(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    th.save(payload, tmp)
    atomic.replace_file(tmp, path)


def load_checkpoint(path: Path):
    """Rebuild the run saved at ``path``; the generators are restored last."""
    from chapter3_pollinators.experiment.run import Run, build_learner

    payload = th.load(Path(path), weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format in {path}")
    game = pickle.loads(payload["game"])
    state = pickle.loads(payload["loop"])
    agents = game.agent_processor.all_agents
    real = build_learner(agents)
    load_learner_state(real, payload["real"])
    real.memory = restore_memory(payload["memory"])
    model = None
    if payload["model"] is not None:
        model = build_learner(agents)
        load_learner_state(model, payload["model"])
        model.memory = ReplayMemory(real.memory.capacity)
        model.memory.memory = [Experience(*e) for e in real.memory.memory]
        model.memory.position = real.memory.position
    restore_generator_states(payload["generators"])
    return Run(seed=payload["seed"], game=game, real=real, model=model, state=state, meta=payload["meta"])
