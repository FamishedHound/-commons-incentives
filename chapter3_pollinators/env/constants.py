"""Fixed parameters of the Pollinators' Game.

All habitat quantities inside the game are integer percentages (0, 10, ..., 100)
of a cell given over to pollinator habitat; observations divide them by 100.
"""

from __future__ import annotations

#: The landscape is a square grid of ``GRID_SIZE x GRID_SIZE`` cells.
GRID_SIZE = 4

#: Number of landowners; the fixed property partition is in :mod:`.properties`.
NUMBER_OF_AGENTS = 4

#: Number of cells seeded with habitat when a pollinator register is created.
NUMBER_OF_RANDOM_POLLINATORS = 1

#: Habitat (percent) of a cell seeded by a new pollinator register.
SEED_HABITAT = 10

#: Upkeep paid per cell per step, in gold coins (also the 0.4 offset of the reward).
LAND_UPKEEP = 40

#: Gold coins a landowner starts with, and is reset to when upkeep exceeds savings.
STARTING_GOLD = 10_000

#: Income of a pollinated cell with no habitat, in gold coins.
MAXIMAL_INCOME = 100

#: Default weight of the local (income) term in a landowner's reward.
DEFAULT_ALPHA = 0.6

#: The habitat levels a landowner can declare, as fractions of a cell.
HABITAT_LEVELS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

# --- pollination model -------------------------------------------------------
#: Neighbour weight ``w(d) = POLLINATION_C * exp(-d / POLLINATION_A)``.
POLLINATION_C = 0.5
POLLINATION_A = 1.1
#: Logistic link ``P = 1 / (1 + exp(-LOGISTIC_K * (s - LOGISTIC_X0)))``.
LOGISTIC_K = 8.4
LOGISTIC_X0 = 0.3
#: Habitat in cells at Euclidean distance <= this radius contributes to pollination.
POLLINATION_RADIUS = 1.9
#: Registered cells at distance < this radius take part in the habitat update.
SPREAD_RADIUS = 3
