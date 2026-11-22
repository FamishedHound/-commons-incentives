"""The pollination model and the pollinator register.

A cell ``c`` with habitat share ``h_c`` (a fraction) is pollinated with
probability

    P(c) = 1 / (1 + exp(-k * (s_c - x0))),
    s_c  = w(0) * h_c + sum_{c' registered, c' != c, d(c, c') <= 1.9} w(d(c, c')) * h_c',
    w(d) = 0.5 * exp(-d / 1.1),

with ``k = 8.4`` and ``x0 = 0.3``; ``d`` is the Euclidean distance between cell
coordinates, so the sum runs over the (up to eight) adjacent cells.  Whether a
cell is pollinated is drawn from Python's ``random`` every step.

The *register* (``all_pollinators``) is the set of cells taking part in the
model.  A register created with the game seeds ``NUMBER_OF_RANDOM_POLLINATORS``
random cells with 10 % habitat.
"""

from __future__ import annotations

import math
import random

from scipy.spatial import distance

from chapter3_pollinators.env.constants import (
    GRID_SIZE,
    LOGISTIC_K,
    LOGISTIC_X0,
    NUMBER_OF_RANDOM_POLLINATORS,
    POLLINATION_A,
    POLLINATION_C,
    POLLINATION_RADIUS,
    SEED_HABITAT,
)
from chapter3_pollinators.env.land import Grid, LandCell


class PollinatorsProcessor:
    """Pollinator register and pollination draws for one grid."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.buffer_lands: list[LandCell] = []
        self.all_pollinators = set(
            [
                (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
                for _ in range(NUMBER_OF_RANDOM_POLLINATORS)
            ]
        )
        for coords in self.all_pollinators:
            cell = self.get_pollinator(coords)
            cell.is_pollinator = True
            cell.declared_habitat = SEED_HABITAT
            cell.actual_habitat = SEED_HABITAT
            self.grid.all_cells[coords].is_pollinator = True
            self.grid.all_cells[coords].declared_habitat = SEED_HABITAT
            self.grid.all_cells[coords].actual_habitat = SEED_HABITAT

    def set_active_pollinator(self, land: LandCell) -> None:
        self.all_pollinators.add((land.x, land.y))

    def get_pollinator(self, coords: tuple[int, int]) -> LandCell | None:
        for x, y in self.grid.all_cells.keys():
            if x == coords[0] and y == coords[1]:
                return self.grid.all_cells[(x, y)]
        return None

    def find_closest_pollinator_to_land(self, current_point, closeness):
        """Registered cells strictly closer than ``closeness``, with their distances."""
        closest = list(
            filter(lambda c: distance.euclidean(c, current_point) < closeness, self.all_pollinators)
        )
        distances = list(map(lambda c: distance.euclidean(c, current_point), closest))
        return closest, distances

    def clear_pollinators(self) -> None:
        """Drop registered cells without habitat, then register every cell with habitat."""
        to_delete = []
        for coords in self.all_pollinators:
            cell = self.grid.all_cells[(coords[0], coords[1])]
            if cell.actual_habitat == 0 or cell.actual_habitat == -1:
                to_delete.append(coords)
        for coords in to_delete:
            self.all_pollinators.remove(coords)
        for coords, cell in self.grid.all_cells.items():
            if cell.actual_habitat > 0:
                self.all_pollinators.add(coords)

    def check_for_failed_pollinators_during_exploration(self) -> bool:
        """True when no registered cell holds any habitat (the pollinators died out)."""
        for coords in self.all_pollinators:
            if self.grid.all_cells[(coords[0], coords[1])].actual_habitat != 0:
                return False
        return True

    def logits(self, x: float) -> float:
        k = LOGISTIC_K
        xo = LOGISTIC_X0
        return 1 / (1 + math.exp(-k * (x - xo)))

    def get_pollinated(self, land: LandCell) -> bool:
        """Draw whether ``land`` is pollinated this step."""
        pollinator_distance = {
            c: distance.euclidean(c, (land.x, land.y))
            for c in self.all_pollinators
            if c != (land.x, land.y)
        }
        within = dict(
            filter(
                lambda elem: self.distance_less_than(elem[1], POLLINATION_RADIUS),
                pollinator_distance.items(),
            )
        )
        weights = []
        for other, dist in within.items():
            habitat = self.get_pollinator(other).actual_habitat
            result = self.sample_pollination(dist)
            weights.append(result * habitat / 100)
        result_from_this_land = self.sample_pollination(0)
        weights.append(result_from_this_land * land.actual_habitat / 100)
        sum_of_weights = sum(weights)
        draw = random.uniform(0, 1)
        probability = self.logits(sum_of_weights)
        return draw < probability

    @staticmethod
    def distance_less_than(number: float, less_than: float) -> bool:
        return number <= less_than

    @staticmethod
    def sample_pollination(x: float, mode: int = 0) -> float:
        c = POLLINATION_C
        a = POLLINATION_A
        return c * math.exp(-x / a)
