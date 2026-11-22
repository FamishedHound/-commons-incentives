"""A landowner: the decentralised agent that owns a property."""

from __future__ import annotations

import random

from chapter3_pollinators.env.constants import DEFAULT_ALPHA, HABITAT_LEVELS, STARTING_GOLD


class Landowner:
    """A landowner with its property, savings and reward weight ``alpha``.

    ``alpha`` weights the local (income) term of the reward against the global
    (pollination) term.  The display colours and the ``trust_factor`` are drawn
    from Python's ``random`` when the landowner is created; they play no role in
    the dynamics but are part of the game's random stream.
    """

    def __init__(
        self,
        agent_id: int,
        pos_x: int,
        pos_y: int,
        number_of_lands: int,
        agent_type: str | None = None,
    ) -> None:
        self.agent_id = (pos_x, pos_y)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.agent_type = agent_type
        self.id = agent_id
        self.cells: list = []
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.predefined_number_of_lands = number_of_lands
        self.assigned_cells = 0
        self.money = 0
        self.is_dead = False
        self.last_income = None
        self.money = STARTING_GOLD
        self.alpha = DEFAULT_ALPHA
        self.trust_factor = round(random.uniform(0.0, 1), 2)

    def select_action(self, output) -> int:
        """Snap a continuous action in [0, 1] to the nearest habitat level, in percent."""
        return (
            min(range(len(HABITAT_LEVELS)), key=lambda i: abs(HABITAT_LEVELS[i] - output)) * 10
        )

    def make_a_decision(self, action, epsilon) -> list[int]:
        """Declare one habitat level per cell; the habitat is in place at once."""
        decisions = []
        for i, land in enumerate(self.cells):
            level = self.select_action(action[i])
            land.declared_habitat = level
            land.actual_habitat = level
            decisions.append(level)
        return decisions

    def __eq__(self, other: object) -> bool:
        return self.pos_x == other.pos_x and self.pos_y == other.pos_y  # type: ignore[attr-defined]

    __hash__ = None  # type: ignore[assignment]
