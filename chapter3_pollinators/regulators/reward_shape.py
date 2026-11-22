"""The reward-shape baseline.

Each step landowner ``i`` receives ``1 - |target_i - mean_c a_{i,c}|``, where
``a_{i,c}`` is the continuous action it takes on cell ``c`` this step (before the
snap to 0.1), so the incentive is never negative and is largest on target.
"""

from __future__ import annotations


def reward_shape_incentive(action, target) -> list[float]:
    incentive = []
    for i in range(len(action)):
        average_action = sum([x.item() for x in action[i]]) / len(action[i])
        incentive.append(1 - (abs(target[i] - average_action)))
    return incentive


def target_gaps(action, target) -> list[float]:
    """``target_i - mean_c a_{i,c}`` per landowner (signed)."""
    return [
        target[i] - sum([x.item() for x in action[i]]) / len(action[i]) for i in range(len(action))
    ]
