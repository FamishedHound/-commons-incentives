"""The training curriculum's threshold rules ("training policies").

Both rules look at a landowner's mean actual habitat ``m`` over its cells and
use fixed thresholds 0.5 and 0.8:

* penalising rule (Policy #1):  ``-1`` if ``m >= 0.8``, ``-0.6`` if ``m >= 0.5``, else ``0``;
* rewarding rule (Policy #2):   ``+0.6`` if ``m >= 0.8``, ``+0.1`` if ``m >= 0.5``, else ``-0.25``.

The schedule that applies them is in :mod:`chapter3_pollinators.experiment.schedule`.
"""

from __future__ import annotations


def _mean_habitat(agent) -> float:
    all_pollinators = 0
    for land in agent.cells:
        all_pollinators += land.actual_habitat / 100
    return all_pollinators / len(agent.cells)


def penalising_incentive(all_agents) -> list[float]:
    """Policy #1: penalise high habitat."""
    incentive = []
    for agent in all_agents:
        mean = _mean_habitat(agent)
        if mean >= 0.8:
            value = -1
        elif mean >= 0.5:
            value = -0.6
        else:
            value = 0
        incentive.append(value)
    return incentive


def rewarding_incentive(all_agents) -> list[float]:
    """Policy #2: reward high habitat."""
    incentive = []
    for agent in all_agents:
        mean = _mean_habitat(agent)
        if mean >= 0.8:
            value = 0.6
        elif mean >= 0.5:
            value = 0.1
        else:
            value = -0.25
        incentive.append(value)
    return incentive
