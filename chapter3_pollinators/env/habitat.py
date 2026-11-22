"""The habitat update applied after the landowners' declarations.

Declared habitat is in place immediately: a declaration sets the cell's actual
habitat in the same step, so the update below never changes a cell; it draws
one number from Python's ``random`` for every registered cell closer than
``SPREAD_RADIUS`` to the cell, and registers every cell that holds habitat.
"""

from __future__ import annotations

import math
from random import uniform

from chapter3_pollinators.env.constants import SPREAD_RADIUS
from chapter3_pollinators.env.land import LandCell
from chapter3_pollinators.env.pollination import PollinatorsProcessor


class EnvironmentalManager:
    """Applies the habitat update against the register it was created with."""

    def __init__(self, pollinators_processor: PollinatorsProcessor) -> None:
        self.pollinators_processor = pollinators_processor
        self.all_pollinators = self.pollinators_processor.all_pollinators

    def process_declared_lands(self) -> None:
        lands_to_process = [land for land in self.pollinators_processor.grid.all_cells.values()]
        for land in lands_to_process:
            if land.actual_habitat != -1:
                current_point = (land.x, land.y)
                closest, distances = self.pollinators_processor.find_closest_pollinator_to_land(
                    current_point, SPREAD_RADIUS
                )
                if closest:
                    self.calculate_environmental_bag(
                        zip(closest, distances),
                        self.pollinators_processor.get_pollinator(current_point),
                    )

    def calculate_environmental_bag(self, closest_with_distance, land: LandCell) -> None:
        for closest, euclidean_distance in closest_with_distance:
            probability = math.exp(-1 * euclidean_distance)
            self.sample_pollinator_to_create_new_one(
                land, probability, self.pollinators_processor.get_pollinator(closest)
            )

    def sample_pollinator_to_create_new_one(self, land: LandCell, probability, pollinator) -> None:
        draw = uniform(0, 1)  # noqa: F841 - part of the random stream
        actual = land.actual_habitat
        declared = land.declared_habitat
        if declared > actual and actual < 100:
            land.actual_habitat = land.declared_habitat
        elif declared < actual:
            land.actual_habitat = declared
        if actual > 0:
            self.pollinators_processor.all_pollinators.add((land.x, land.y))
