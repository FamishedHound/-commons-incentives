"""The training schedule and the regulators active at each step.

A run is one continuous sequence of environment steps (there are no episodes).

* Exploration: epsilon starts at 1; once the learner has collected 16 steps it
  falls by 1e-4 per step down to a floor of 0.4.  Each step a single draw
  decides for all four landowners: with probability epsilon every landowner
  takes a uniform random action, otherwise every landowner acts greedily.
* Counted steps: the step counter rises by one at the start of every step taken
  with epsilon at or below 0.4, i.e. from roughly environment step 6,016 on.
* Curriculum (warm-up): no incentive while epsilon > 0.8 (Policy #0); the
  penalising threshold rule while epsilon <= 0.8 and the counter is below 100
  (Policy #1); the rewarding threshold rule for counted steps 100-999
  (Policy #2).
* Evaluation window: counted steps 1,000-2,199 (1,200 steps), during which the
  condition's regulator is active.
"""

from __future__ import annotations

from dataclasses import dataclass

EPSILON_START = 1
EPSILON_DECAY = 1e-4
EPSILON_FLOOR = 0.4
#: The penalising rule starts once epsilon has fallen to this value.
CURRICULUM_EPSILON = 0.8
#: The penalising rule applies while the counter is at most this value ...
PENALISING_LAST_COUNTER = 100
#: ... and the rewarding rule from this counter (it takes precedence at 100).
REWARDING_FIRST_COUNTER = 100
#: First counted step of the evaluation window (the branch point).
BRANCH_COUNTER = 1000
#: Length of the evaluation window in counted steps.
WINDOW_STEPS = 1200
#: Last counted step of the evaluation window.
LAST_COUNTER = BRANCH_COUNTER + WINDOW_STEPS - 1

#: Landowner learner settings.
N_AGENTS = 4
BATCH_SIZE = 16
CAPACITY = 9000
EPISODES_BEFORE_TRAIN = 10
DIM_OBS = 12
DIM_ACT = 1

#: Integer codes of the regulator acting at a step (stored with every record).
PHASE_CODES = {
    "none": 0,
    "penalising": 1,
    "rewarding": 2,
    "reward_shape": 3,
    "oraa": 4,
    "ltos": 5,
}


@dataclass(frozen=True)
class Schedule:
    """Which regulator acts at which counted step.

    ``curriculum`` enables the two threshold rules.  ``reward_shape_from`` /
    ``reward_shape_until`` bound the reward-shape window (until is exclusive,
    ``None`` = open), ``oraa_from`` starts ORAA with the ``oraa_networks``
    (``"model"`` or ``"real"``) searching incentives in ``[-oraa_bound,
    oraa_bound]``, and ``ltos_from`` starts reward sharing.
    """

    curriculum: bool = True
    reward_shape_from: int | None = None
    reward_shape_until: int | None = None
    oraa_from: int | None = None
    oraa_networks: str = "model"
    oraa_bound: float = 1.0
    ltos_from: int | None = None

    def oraa_active(self, counter: int) -> bool:
        return self.oraa_from is not None and counter >= self.oraa_from

    def ltos_active(self, counter: int) -> bool:
        return self.ltos_from is not None and counter >= self.ltos_from

    def reward_shape_active(self, counter: int) -> bool:
        if self.reward_shape_from is None or counter < self.reward_shape_from:
            return False
        return self.reward_shape_until is None or counter < self.reward_shape_until


#: The warm-up: Policy #0, then the curriculum, up to counted step 999.
WARMUP = Schedule(curriculum=True)

#: Schedules of the evaluation window, per condition.
CONDITION_SCHEDULES = {
    "no_regulator": Schedule(curriculum=False),
    "reward_shape": Schedule(curriculum=True, reward_shape_from=BRANCH_COUNTER),
    "oraa_mb": Schedule(curriculum=True, oraa_from=BRANCH_COUNTER, oraa_networks="model"),
    "oraa_omni": Schedule(curriculum=True, oraa_from=BRANCH_COUNTER, oraa_networks="real"),
    "ltos": Schedule(curriculum=True, ltos_from=BRANCH_COUNTER),
    "oraa_mb_b3": Schedule(curriculum=True, oraa_from=BRANCH_COUNTER, oraa_networks="model", oraa_bound=3.0),
    "oraa_omni_b3": Schedule(curriculum=True, oraa_from=BRANCH_COUNTER, oraa_networks="real", oraa_bound=3.0),
}


def next_epsilon(epsilon: float, collected: int) -> float:
    """Epsilon after a step, ``collected`` being the number of steps taken so far."""
    if collected >= BATCH_SIZE:
        if epsilon > EPSILON_FLOOR:
            epsilon -= EPSILON_DECAY
        else:
            epsilon = EPSILON_FLOOR
    return epsilon
