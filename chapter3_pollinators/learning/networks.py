"""Actor and critic networks of the P-MADDPG landowners.

Both networks flatten their input, so one forward pass handles one sample.

* Actor: the landowner's ``2 x 4 x 4`` observation (32 values) -> 1024 -> 2048
  -> ``M`` outputs with a sigmoid, ``M`` being the number of cells it owns.
* Critic: the global state (32 values) concatenated with the landowner's own
  ``M`` actions -> 512 -> 512 -> 1, i.e. ``Q_i(s, a_i)``.

Each network also defines a layer ``FC3`` that the forward pass does not use.
It is kept because creating it draws from torch's generator, so it is part of
the initialisation stream.
"""

from __future__ import annotations

import torch as th
import torch.nn as nn
import torch.nn.functional as F

#: Size of a flattened observation or global state: 2 layers x 4 x 4 cells.
STATE_SIZE = 32


class Critic(nn.Module):
    """``Q_i(s, a_i)`` for one landowner with ``dim_observation`` cells."""

    def __init__(self, n_agent: int, dim_observation: int, batch_size: int) -> None:
        super().__init__()
        self.n_agent = n_agent
        self.dim_observation = 48
        self.dim_action = 4
        self.FC1 = nn.Linear(STATE_SIZE + dim_observation, 512)
        self.FC2 = nn.Linear(512, 512)
        self.FC3 = nn.Linear(1024, 512)
        self.FC4 = nn.Linear(512, 1)

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        obs = th.flatten(obs.float())
        acts = th.flatten(acts).float()
        combined = th.cat([obs, acts], 0)
        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))
        return self.FC4(result)


class Actor(nn.Module):
    """Deterministic policy over the habitat share of each of ``dim_observation`` cells."""

    def __init__(self, dim_observation: int, dim_action: int) -> None:
        super().__init__()
        self.FC1 = nn.Linear(STATE_SIZE, 1024)
        self.FC2 = nn.Linear(1024, 2048)
        self.FC3 = nn.Linear(1024, 512)
        self.FC4 = nn.Linear(2048, dim_observation)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs = th.flatten(obs)
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC4(result)
        return th.sigmoid(result)
