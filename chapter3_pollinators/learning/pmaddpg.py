"""P-MADDPG: the landowners' learner.

One actor and one critic per landowner.  The actor maps the landowner's own
observation (its cells only) to one habitat share per cell; the critic is
``Q_i(s, a_i)``, conditioned on the global state and the landowner's own action.

Update (called once per environment step, after 16 steps of collection): for
each landowner in turn a mini-batch of 16 transitions is drawn from the shared
replay memory (capacity 9,000); the critic is regressed on

    y = r_i + gamma * Q'_i(s', mu'_i(o'_i)),    gamma = 0.1,

with the target networks ``Q'``, ``mu'`` and no terminal mask, and the actor
ascends ``Q_i(s, mu_i(o_i))``.  Both use Adam with learning rate 1e-4.  The
target networks are hard copies refreshed whenever the update counter, which
rises by one per landowner update, is a multiple of 250, i.e. every 125
environment steps.  Each sample of a mini-batch goes through the networks on
its own.  Networks live on the GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam

from chapter3_pollinators.learning.memory import Experience, ReplayMemory
from chapter3_pollinators.learning.networks import Actor, Critic

#: Discount factor of the landowners.
GAMMA = 0.1
#: Adam step size of actors and critics.
LEARNING_RATE = 1e-4
#: The target networks are copied when the update counter is a multiple of this.
TARGET_COPY_INTERVAL = 250

#: ``(transitions, agent_index) -> (S, S', s, s', a, r)`` arrays for one landowner.
BatchExtractor = Callable[[list, int], tuple]


def hard_update(target: nn.Module, source: nn.Module) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def extract_batch(transitions: list, agent_index: int) -> tuple:
    """The arrays one landowner's update reads from a list of transitions."""
    batch = Experience(*zip(*transitions))
    own_states = np.stack([x[agent_index] for x in batch.states]).squeeze(1)
    own_next_states = np.stack([x[agent_index] for x in batch.next_states]).squeeze(1)
    global_states = np.asarray(batch.global_state)
    global_next_states = np.asarray(batch.next_global_state)
    actions = np.stack([x[agent_index] for x in batch.actions])
    rewards = np.stack([x[agent_index] for x in batch.rewards])
    return own_states, own_next_states, global_states, global_next_states, actions, rewards


class PMADDPG:
    """Four P-MADDPG landowners sharing one replay memory."""

    def __init__(
        self,
        n_agents: int,
        dim_obs: int,
        dim_act: int,
        batch_size: int,
        capacity: int,
        episodes_before_train: int,
        worlds_all_agents,
    ) -> None:
        self.initialise_networks(batch_size, dim_act, worlds_all_agents)
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train
        self.GAMMA = GAMMA
        self.critic_optimizer = [Adam(x.parameters(), lr=LEARNING_RATE) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(), lr=LEARNING_RATE) for x in self.actors]
        if self.use_cuda:
            for network in self.actors + self.critics + self.actors_target + self.critics_target:
                network.cuda()
        self.steps_done = 0
        self.episode_done = 0

    def initialise_networks(self, batch_size: int, dim_act: int, worlds_all_agents) -> None:
        self.actors = [Actor(len(a.cells), dim_act) for a in worlds_all_agents]
        self.critics = [Critic(len(a.cells), len(a.cells), batch_size) for a in worlds_all_agents]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def update_policy(self, all_agents, epsilon, randy_random=None, extractor=None):
        """One update of every landowner; ``(critic_losses, actor_losses)`` or ``(None, None)``.

        ``extractor`` (default :func:`extract_batch`) turns the sampled
        transitions into the arrays a landowner's update reads.
        """
        if self.episode_done <= self.batch_size:
            return None, None
        extract = extract_batch if extractor is None else extractor
        c_loss = []
        a_loss = []
        for agent_index, agent in enumerate(all_agents):
            transitions = self.memory.sample(self.batch_size)
            arrays = extract(transitions, agent_index)
            loss_q, actor_loss = self.update_agent(agent_index, agent, *arrays)
            c_loss.append(loss_q)
            a_loss.append(actor_loss)
            self.steps_done += 1

        if self.steps_done % TARGET_COPY_INTERVAL == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                hard_update(self.critics_target[i], self.critics[i])
                hard_update(self.actors_target[i], self.actors[i])
        return c_loss, a_loss

    def update_agent(
        self,
        agent_index: int,
        agent,
        own_states: np.ndarray,
        own_next_states: np.ndarray,
        global_states: np.ndarray,
        global_next_states: np.ndarray,
        action: np.ndarray,
        reward_batch: np.ndarray,
    ):
        """Critic then actor step of one landowner on one mini-batch.

        Every sample goes through the networks on its own, one forward pass per
        sample; the mini-batch arrays are moved to the GPU once and the
        per-sample outputs are collected on the GPU.
        """
        batch = self.batch_size
        critic = self.critics[agent_index]
        actor = self.actors[agent_index]
        reward_batch = th.from_numpy(reward_batch).cuda()
        states = th.from_numpy(global_states).cuda()
        next_states = th.from_numpy(global_next_states).cuda()
        actions = th.from_numpy(action).cuda()
        next_own = th.from_numpy(own_next_states).float().cuda()

        self.critic_optimizer[agent_index].zero_grad()
        current_q = th.cat([critic(states[i], actions[i]) for i in range(batch)])
        next_actions = [self.actors_target[agent_index](next_own[i]) for i in range(batch)]
        target_q = th.cat(
            [self.critics_target[agent_index](next_states[i], next_actions[i]) for i in range(batch)]
        )
        target_q = (target_q * self.GAMMA) + reward_batch

        loss_q = nn.MSELoss()(current_q.float(), target_q.float().detach())
        loss_q.backward()
        self.critic_optimizer[agent_index].step()

        self.actor_optimizer[agent_index].zero_grad()
        own = th.from_numpy(own_states.copy()).float().cuda()
        states_32 = th.from_numpy(global_states.copy()).float().cuda()
        action_i = th.stack([actor(own[i]) for i in range(batch)])
        actor_loss = th.cat([-critic(states_32[i], action_i[i]) for i in range(batch)])
        actor_loss = actor_loss.view(batch, -1).mean()
        actor_loss.backward()
        self.actor_optimizer[agent_index].step()
        return loss_q, actor_loss

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------
    def select_action(self, state_batch, all_agents) -> list[list[th.Tensor]]:
        """Greedy actions: one list of 0-d float32 tensors per landowner."""
        actions = []
        for agent_index, _agent in enumerate(all_agents):
            sb = np.asarray(state_batch[agent_index])
            decisions = (
                self.actors[agent_index](th.from_numpy(np.array(sb)).unsqueeze(0).float().cuda())
                .squeeze()
                .data.cpu()
            )
            actions.append([th.tensor(x) for x in decisions.tolist()])
        return actions
