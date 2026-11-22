"""LToS (Yi et al., "Learning to Share in Networked Multi-Agent RL") over the P-MADDPG landowners.

Each landowner ``i`` has a high-level policy ``phi_i`` (:class:`SharingActor`)
that maps the habitat of its neighbourhood to its outgoing sharing weights
``w_i^out`` on the simplex over ``N_i``.  The low level is the landowners'
P-MADDPG learner, unchanged, trained on the shaped reward.

At every step of the evaluation window:

1. every ``action_interval`` steps each landowner draws
   ``w_i^out = softmax(phi_i(o_i) + eps * OU_i)``; the weights are held in
   between (Appendix C, different time scales);
2. landowner ``j`` observes its incoming weights ``w_j^in`` in its incentive
   layer: ``W[i, j]`` is written on the cells of each ``i`` in ``N_j`` (the
   layer that carries the regulator's incentive in every other condition), so
   its actor and critic are conditioned on ``w_j^in`` without any change to the
   networks;
3. the landowners act and the game pays no external incentive; the rewards are
   shared, ``r_j^w = sum_i W[i, j] r_i``;
4. the landowners' learner is updated on ``r^w`` exactly as in every other
   condition; in its targets the next state carries the incoming weights the
   *target* high level would set there (Algorithm 1, line 15);
5. every ``update_interval`` steps the high level is updated (Algorithm 1,
   lines 18-22): a synchronised sample of past steps is drawn for all
   landowners; with ``w = phi(o)`` (no noise), each landowner ``j`` computes
   ``g_j^in = d Q_j(s; w_j^in, a_j*) / d w_j^in`` at its greedy action ``a_j*``
   with its own critic, the gradients are exchanged so that ``i`` receives
   ``g_i^out[j] = g_j^in[i]``, and ``phi_i`` takes an SGD step along
   ``(d phi_i / d theta_i)^T g_i^out``; the target high level then follows by
   soft update.

LToS draws its initial weights, its noise and its samples from generators of
its own, seeded from the run seed, so the global generators that drive the game
and the landowners are not touched.
"""

from __future__ import annotations

import random
from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch as th

from chapter3_pollinators.env.properties import PROPERTY_CELLS
from chapter3_pollinators.learning.memory import Experience
from chapter3_pollinators.ltos.batched import actor_batch, critic_batch
from chapter3_pollinators.ltos.config import LToSConfig
from chapter3_pollinators.ltos.graph import neighbourhoods, property_adjacency
from chapter3_pollinators.ltos.high_level import OrnsteinUhlenbeckNoise, SharingActor, soft_update
from chapter3_pollinators.ltos.sharing import (
    exchange_gradients,
    property_masks,
    shaped_rewards,
    transfer_volume,
    weight_matrix,
)

_LTOS_STREAM = 0x170C5

#: A landowner transition of the sharing phase: raw observations and states, the
#: shaped rewards and the weights in force; the incoming weights are written in
#: when a batch is built.
SharingExperience = namedtuple("SharingExperience", Experience._fields + ("weights", "uid"))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


class RewardSharing:
    """The LToS high level and its hooks into the experiment loop."""

    def __init__(self, real, seed: int, config: LToSConfig | None = None, property_cells=PROPERTY_CELLS):
        self.real = real
        self.config = config or LToSConfig()
        self.hoods = neighbourhoods(property_adjacency(property_cells))
        self.masks = property_masks(property_cells)
        self.hood_masks = np.stack([self.masks[list(h)].sum(axis=0) for h in self.hoods])
        streams = np.random.SeedSequence([int(seed), _LTOS_STREAM]).spawn(3)
        generator = th.Generator().manual_seed(int(streams[0].generate_state(1)[0]))
        self.actors = [
            SharingActor(
                len(hood), hood.index(i), self.config.selfishness, generator,
                self.config.hidden_units, self.config.init_std,
            )
            for i, hood in enumerate(self.hoods)
        ]
        self.targets = [deepcopy(actor) for actor in self.actors]
        self.optimisers = [
            th.optim.SGD(actor.parameters(), lr=self.config.learning_rate) for actor in self.actors
        ]
        noise_rng = np.random.default_rng(streams[1])
        self.noise = [
            OrnsteinUhlenbeckNoise(
                len(hood), self.config.ou_sigma, noise_rng, self.config.ou_theta, self.config.ou_dt
            )
            for hood in self.hoods
        ]
        self.sampler = random.Random(int(streams[2].generate_state(1)[0]))
        self.buffer: list[np.ndarray] = []
        self.steps = 0
        self.updates = 0
        self.target_version = 0
        self.current: np.ndarray | None = None
        self._next_weights: dict[int, tuple[int, np.ndarray]] = {}
        self._gradient_norms: list[float] = []
        self._first_weights: np.ndarray | None = None

    @classmethod
    def for_run(cls, run, seed: int, config: LToSConfig | None = None) -> RewardSharing:
        return cls(run.real, seed=seed, config=config)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def high_observations(self, habitat: np.ndarray) -> np.ndarray:
        """``(..., 4, 4)`` habitat layers -> ``(..., 4, 16)``: each neighbourhood's habitat."""
        masked = habitat[..., None, :, :] * self.hood_masks
        return masked.reshape(*habitat.shape[:-2], len(self.hoods), 16).astype(np.float32)

    def incoming(self, w: np.ndarray, j: int) -> np.ndarray:
        return np.tensordot(w[:, j], self.masks, axes=1)

    def _paint(self, layers: np.ndarray, w: np.ndarray, j: int) -> np.ndarray:
        painted = np.array(layers, dtype=np.float64)
        painted[1] = self.incoming(w, j)
        return painted

    def acting_observation(self, obs, w: np.ndarray):
        return [[self._paint(np.asarray(obs[j])[0], w, j)] for j in range(len(self.hoods))]

    def policy_weights(self, habitat: np.ndarray, target: bool = False) -> np.ndarray:
        """``W`` the (target) high level sets for one habitat layer, without noise."""
        x = th.from_numpy(self.high_observations(habitat))
        nets = self.targets if target else self.actors
        with th.no_grad():
            outs = [th.softmax(net(x[i : i + 1]), dim=-1)[0].numpy() for i, net in enumerate(nets)]
        return weight_matrix(outs, self.hoods)

    # ------------------------------------------------------------------
    # hooks called by the driver
    # ------------------------------------------------------------------
    def begin_step(self, state) -> np.ndarray:
        if self.steps % self.config.action_interval == 0:
            x = th.from_numpy(self.high_observations(np.asarray(state.global_state)[0]))
            with th.no_grad():
                logits = [net(x[i : i + 1])[0].double().numpy() for i, net in enumerate(self.actors)]
            outs = [
                _softmax(logits[i] + state.epsilon * self.noise[i].sample())
                for i in range(len(self.hoods))
            ]
            self.current = weight_matrix(outs, self.hoods)
            if self._first_weights is None:
                self._first_weights = self.current.copy()
        return self.current

    def share(self, reward, w: np.ndarray) -> tuple[list[float], float]:
        return shaped_rewards(reward, w), transfer_volume(reward, w)

    def transition(self, obs, action, next_obs, reward, shaped, global_state, next_global, w):
        self.buffer.append(np.asarray(global_state)[0].copy())
        if len(self.buffer) > self.config.capacity:
            self.buffer.pop(0)
        return SharingExperience(
            obs, action, next_obs, list(shaped), global_state, next_global, w.copy(), self.steps
        )

    def end_step(self, state) -> None:
        self.steps += 1
        if self.steps % self.config.update_interval == 0:
            self.update_high_level()

    # ------------------------------------------------------------------
    # landowner batches
    # ------------------------------------------------------------------
    def _target_weights(self, experience: SharingExperience) -> np.ndarray:
        cached = self._next_weights.get(experience.uid)
        if cached is not None and cached[0] == self.target_version:
            return cached[1]
        w_next = self.policy_weights(np.asarray(experience.next_global_state)[0], target=True)
        self._next_weights[experience.uid] = (self.target_version, w_next)
        return w_next

    def extractor(self, transitions: list, agent_index: int) -> tuple:
        """The arrays of one landowner's update, with the incoming weights written in."""
        own, own_next, states, next_states = [], [], [], []
        for x in transitions:
            if isinstance(x, SharingExperience):
                w_next = self._target_weights(x)
                own.append(self._paint(np.asarray(x.states[agent_index])[0], x.weights, agent_index)[None])
                own_next.append(
                    self._paint(np.asarray(x.next_states[agent_index])[0], w_next, agent_index)[None]
                )
                states.append(self._paint(x.global_state, x.weights, agent_index))
                next_states.append(self._paint(x.next_global_state, w_next, agent_index))
            else:
                own.append(np.asarray(x.states[agent_index]))
                own_next.append(np.asarray(x.next_states[agent_index]))
                states.append(x.global_state)
                next_states.append(x.next_global_state)
        actions = np.stack([x.actions[agent_index] for x in transitions])
        rewards = np.stack([x.rewards[agent_index] for x in transitions])
        return (
            np.stack(own).squeeze(1),
            np.stack(own_next).squeeze(1),
            np.asarray(states),
            np.asarray(next_states),
            actions,
            rewards,
        )

    # ------------------------------------------------------------------
    # high-level update
    # ------------------------------------------------------------------
    def incoming_gradients(self, habitat: np.ndarray) -> list[np.ndarray]:
        """``g_j^in`` for a batch of habitat layers ``(B, 4, 4)``, one array per landowner."""
        x = th.from_numpy(self.high_observations(habitat))
        with th.no_grad():
            outs = [th.softmax(net(x[:, i]), dim=-1).numpy() for i, net in enumerate(self.actors)]
        w = np.stack([weight_matrix([o[b] for o in outs], self.hoods) for b in range(habitat.shape[0])])
        device = self.real.critics[0].FC1.weight.device
        masks = th.from_numpy(self.masks).float().to(device)
        hab = th.from_numpy(habitat).float().to(device)
        g_in = []
        for j, hood in enumerate(self.hoods):
            w_in = th.from_numpy(w[:, list(hood), j]).float().to(device).requires_grad_(True)
            layer = th.einsum("bk,kxy->bxy", w_in, masks[list(hood)])
            state = th.stack([hab, layer], dim=1)
            with th.no_grad():
                own = th.stack([hab * masks[j], layer], dim=1)
                greedy = actor_batch(self.real.actors[j], own)
            value = critic_batch(self.real.critics[j], state, greedy)
            (gradient,) = th.autograd.grad(value.sum(), w_in)
            g_in.append(gradient.detach().cpu().double().numpy())
        return g_in

    def update_high_level(self) -> None:
        n = min(self.config.sample_size, len(self.buffer))
        if n == 0:
            return
        indices = self.sampler.sample(range(len(self.buffer)), n)
        habitat = np.stack([self.buffer[k] for k in indices])
        size = self.config.batch_size
        for start in range(0, n, size):
            batch = habitat[start : start + size]
            g_out = exchange_gradients(self.incoming_gradients(batch), self.hoods)
            self._gradient_norms.append(float(np.mean([np.abs(g).mean() for g in g_out])))
            x = th.from_numpy(self.high_observations(batch))
            for i, (actor, optimiser) in enumerate(zip(self.actors, self.optimisers)):
                weights = th.softmax(actor(x[:, i]), dim=-1)
                loss = -(weights * th.from_numpy(g_out[i]).float()).sum(dim=-1).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        for target, actor in zip(self.targets, self.actors):
            soft_update(target, actor, self.config.tau)
        self.target_version += 1
        self.updates += 1

    def summary(self) -> dict:
        from dataclasses import asdict

        return {
            "neighbourhoods": [list(h) for h in self.hoods],
            "config": asdict(self.config),
            "high_level_updates": self.updates,
            "steps": self.steps,
            "first_weights": None if self._first_weights is None else self._first_weights.tolist(),
            "last_weights": None if self.current is None else self.current.tolist(),
            "mean_abs_exchanged_gradient": (
                float(np.mean(self._gradient_norms)) if self._gradient_norms else None
            ),
        }
