"""Cells of the landscape and the grid that holds them."""

from __future__ import annotations

from chapter3_pollinators.env.constants import GRID_SIZE


class LandCell:
    """One cell of the landscape.

    ``declared_habitat`` is the habitat share (percent) its owner declares and
    ``actual_habitat`` the share actually holding pollinators; in this game the
    two are equal from the moment of the declaration.  ``last_income`` is the
    crop income of the last step and ``was_pollinated`` whether the cell was
    pollinated in it.
    """

    def __init__(self, x: int, y: int) -> None:
        self.is_owned = False
        self.owner = None
        self.x = x
        self.y = y
        self.is_pollinator = False
        self.declared_habitat = 0
        self.actual_habitat = 0
        self.was_pollinated = False
        self.dead = False
        self.last_income = 0

    def set_owner(self, owner) -> None:
        self.owner = owner
        owner.assigned_cells += 1

    def set_owned(self, is_owned: bool) -> None:
        self.is_owned = is_owned

    def __eq__(self, other: object) -> bool:
        return self.x == other.x and self.y == other.y  # type: ignore[attr-defined]

    __hash__ = None  # type: ignore[assignment]


class Grid:
    """The ``GRID_SIZE x GRID_SIZE`` landscape, keyed by ``(x, y)`` in row-major order."""

    def __init__(self) -> None:
        self.all_cells = {
            (x, y): LandCell(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
        }

    def get_cell(self, coords: tuple[int, int]) -> LandCell | None:
        for key, cell in self.all_cells.items():
            if key == coords:
                return cell
        return None
