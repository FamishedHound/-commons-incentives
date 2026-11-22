"""Crop income and land upkeep.

Every step each pollinated cell earns ``(100 - h) / 100 * MAXIMAL_INCOME`` gold
coins, ``h`` being its habitat share in percent; an unpollinated cell earns
nothing.  The income is added to the landowner's savings and the upkeep
(``LAND_UPKEEP`` per cell) is deducted; a landowner whose upkeep exceeds its
savings is given ``STARTING_GOLD`` again.  The reward only uses the cells'
income (see :meth:`chapter3_pollinators.env.game.PollinatorsGame._get_reward`).
"""

from __future__ import annotations

import random

from chapter3_pollinators.env.constants import LAND_UPKEEP, MAXIMAL_INCOME, STARTING_GOLD
from chapter3_pollinators.env.pollination import PollinatorsProcessor


class EconomyManager:
    """Pays income and charges upkeep, using the register it was created with."""

    def __init__(self, all_agents, pollinators_processor: PollinatorsProcessor) -> None:
        self.upcost_price = LAND_UPKEEP
        self.all_agents = all_agents
        self.land_fee = LAND_UPKEEP
        self.pollinators_processor = pollinators_processor

    def deduce_land_fee(self) -> None:
        for agent in self.all_agents:
            income = self.add_income(agent)
            agent.money += income
            agent.income = income
            fee_to_pay = len(agent.cells) * self.land_fee
            if fee_to_pay > agent.money:
                self.handle_dead_situation(agent)
            else:
                agent.money -= fee_to_pay
            agent.last_income = income

    def handle_dead_situation(self, agent) -> None:
        agent.money = STARTING_GOLD
        agent.is_dead = False
        agent.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        agent.color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def add_income(self, agent) -> float:
        total_gross_income = 0
        for land in agent.cells:
            if self.pollinators_processor.get_pollinated(land):
                this_land_income = (100 - land.actual_habitat) / 100 * MAXIMAL_INCOME
                total_gross_income += this_land_income
                land.was_pollinated = True
                land.last_income = this_land_income
            else:
                land.was_pollinated = False
                land.last_income = 0
        return total_gross_income
