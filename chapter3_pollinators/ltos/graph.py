"""The landowners' graph: two landowners are neighbours when their properties share a cell edge."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def property_adjacency(property_cells: Sequence[Sequence[tuple[int, int]]]) -> np.ndarray:
    """Boolean ``n x n`` adjacency; True when some cells of i and j are edge-adjacent."""
    n = len(property_cells)
    adjacency = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            adjacency[i, j] = any(
                abs(xa - xb) + abs(ya - yb) == 1
                for xa, ya in property_cells[i]
                for xb, yb in property_cells[j]
            )
    return adjacency


def neighbourhoods(adjacency: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """``N_i``: landowner ``i`` and its neighbours, in increasing index order."""
    n = adjacency.shape[0]
    return tuple(tuple(j for j in range(n) if j == i or adjacency[i, j]) for i in range(n))
