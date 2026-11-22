"""The landowners' actor and critic evaluated on a batch at once.

The networks themselves process one flattened sample per call; these helpers
apply the same layers to a batch, which the high-level update needs.
"""

from __future__ import annotations

import torch as th
import torch.nn.functional as F


def actor_batch(actor, observations: th.Tensor) -> th.Tensor:
    """``(B, 2, 4, 4)`` observations -> ``(B, M)`` actions."""
    x = observations.reshape(observations.shape[0], -1).float()
    x = F.relu(actor.FC1(x))
    x = F.relu(actor.FC2(x))
    return th.sigmoid(actor.FC4(x))


def critic_batch(critic, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
    """``(B, 2, 4, 4)`` states and ``(B, M)`` actions -> ``(B,)`` values ``Q_i(s, a_i)``."""
    x = th.cat([states.reshape(states.shape[0], -1).float(), actions.float()], dim=1)
    x = F.relu(critic.FC1(x))
    x = F.relu(critic.FC2(x))
    return critic.FC4(x).squeeze(-1)
