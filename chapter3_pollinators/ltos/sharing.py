"""Reward-sharing arithmetic of LToS.

``W[i, j]`` is the share of landowner ``i``'s reward that it gives to ``j``; row
``i`` is landowner ``i``'s outgoing weights ``w_i^out`` (on the simplex over its
neighbourhood ``N_i``, zero elsewhere) and column ``j`` holds ``j``'s incoming
weights ``w_j^in``.  The shaped reward of ``j`` is the sum of the shares it
receives, its own included:

    r_j^w = sum_{i in N_j} W[i, j] * r_i,

so ``sum_j r_j^w = sum_i r_i``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chapter3_pollinators.env.constants import GRID_SIZE


def weight_matrix(outgoing: Sequence[Sequence[float]], hoods: Sequence[Sequence[int]]) -> np.ndarray:
    """Assemble ``W`` from each landowner's outgoing weights over its neighbourhood."""
    n = len(hoods)
    w = np.zeros((n, n))
    for i, hood in enumerate(hoods):
        for slot, j in enumerate(hood):
            w[i, j] = outgoing[i][slot]
    return w


def shaped_rewards(own: Sequence[float], w: np.ndarray) -> list[float]:
    own = np.asarray(own, dtype=np.float64)
    return [float(x) for x in w.T @ own]


def net_received(own: Sequence[float], w: np.ndarray) -> list[float]:
    """Reward received from others minus reward given to others, per landowner."""
    return [s - float(r) for s, r in zip(shaped_rewards(own, w), own)]


def transfer_volume(own: Sequence[float], w: np.ndarray) -> float:
    """Total reward moved between landowners: ``sum_i sum_{j != i} W[i, j] r_i``."""
    own = np.asarray(own, dtype=np.float64)
    return float(((1.0 - np.diag(w)) * own).sum())


def property_masks(property_cells) -> np.ndarray:
    """``masks[i]``: 1 on landowner ``i``'s cells, oriented like the game's observation layers."""
    masks = np.zeros((len(property_cells), GRID_SIZE, GRID_SIZE))
    for i, cells in enumerate(property_cells):
        layer = np.zeros((GRID_SIZE, GRID_SIZE))
        for x, y in cells:
            layer[x, y] = 1.0
        masks[i] = np.rot90(layer)
    return masks


def incoming_layer(w: np.ndarray, j: int, property_cells) -> np.ndarray:
    """Landowner ``j``'s incentive layer under sharing: ``W[i, j]`` on the cells of each ``i``."""
    return np.tensordot(w[:, j], property_masks(property_cells), axes=1)


def exchange_gradients(g_in: Sequence[np.ndarray], hoods: Sequence[Sequence[int]]) -> list[np.ndarray]:
    """Send each component of ``g_j^in`` back to the landowner that set that weight.

    ``g_in[j][:, k]`` is the gradient of ``j``'s value with respect to the weight
    it receives from ``hoods[j][k]``; the result ``g_out[i][:, s]`` is the
    gradient with respect to the weight ``i`` gives to ``hoods[i][s]``.
    """
    g_out = [np.zeros((g_in[0].shape[0], len(hood))) for hood in hoods]
    for j, hood_j in enumerate(hoods):
        for k, i in enumerate(hood_j):
            g_out[i][:, hoods[i].index(j)] = g_in[j][:, k]
    return g_out
