"""The fixed partition of the landscape into four properties.

Cells are ``(x, y)`` grid coordinates.  The four properties are disjoint and
together cover all 16 cells:

========  ===================================  =====
landowner cells                                size
========  ===================================  =====
0         (0,0) (0,1) (0,2) (0,3)              4
1         (1,0) (1,1) (1,2) (2,0) (3,0)        5
2         (2,1) (2,2) (3,1) (3,2)              4
3         (1,3) (2,3) (3,3)                    3
========  ===================================  =====
"""

from __future__ import annotations

from chapter3_pollinators.env.constants import NUMBER_OF_AGENTS
from chapter3_pollinators.env.land import Grid
from chapter3_pollinators.env.landowner import Landowner

#: Cells of each property, in the order the landowner's actions address them.
PROPERTY_CELLS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((1, 0), (1, 1), (1, 2), (2, 0), (3, 0)),
    ((2, 1), (2, 2), (3, 1), (3, 2)),
    ((1, 3), (2, 3), (3, 3)),
)

#: The nominal position of each landowner (the first cell of its property).
AGENT_POSITIONS: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (2, 1), (1, 3))

#: Number of cells of each property.
PROPERTY_SIZES: tuple[int, ...] = tuple(len(cells) for cells in PROPERTY_CELLS)


class AgentProcessor:
    """Creates the landowners and hands each its property."""

    def __init__(self, grid: Grid) -> None:
        self.all_agents: list[Landowner] = []
        self.grid = grid
        self.agents_pos_memory: set[tuple[int, int]] = set()
        self.distribute_land()

    def distribute_land(self) -> None:
        for index in range(NUMBER_OF_AGENTS):
            pos_x, pos_y = AGENT_POSITIONS[index]
            agent = Landowner(index, pos_x, pos_y, 25, "RuleBasedAgent")
            for coords in PROPERTY_CELLS[index]:
                cell = self.grid.all_cells[coords]
                agent.cells.append(cell)
                cell.set_owned(True)
                cell.set_owner(agent)
            self.all_agents.append(agent)

    def clear_empty_agents(self) -> None:
        for agent in self.all_agents:
            if len(agent.cells) == 0:
                self.all_agents.remove(agent)
