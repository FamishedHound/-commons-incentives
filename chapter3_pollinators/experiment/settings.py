"""Communities, governments, conditions and the priority groups of the campaign.

Communities (the landowners' ``alpha``, agent order 1-4 = landowners 0-3):

* HM1-HM6: every landowner has the same alpha, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0;
* HT1-HT4: per-landowner alphas of Appendix Table B.2.

Governments (the regulator's habitat targets):

* HMC: 0.35 for every landowner;
* HTC: the per-landowner targets of Appendix Table B.2 (for HM communities the
  table's HM# column, 0.2 / 0.1 / 0.8 / 0.9).

Conditions: the five of the main experiment, and the two forms of ORAA with the
bound of its incentive search widened from 1 to 3 (``oraa_mb_b3``,
``oraa_omni_b3``; priority group 4).
"""

from __future__ import annotations

from dataclasses import dataclass

#: alpha of landowners 0-3 in each community.
COMMUNITIES: dict[str, tuple[float, float, float, float]] = {
    "HM1": (0.0, 0.0, 0.0, 0.0),
    "HM2": (0.2, 0.2, 0.2, 0.2),
    "HM3": (0.4, 0.4, 0.4, 0.4),
    "HM4": (0.6, 0.6, 0.6, 0.6),
    "HM5": (0.8, 0.8, 0.8, 0.8),
    "HM6": (1.0, 1.0, 1.0, 1.0),
    "HT1": (0.4, 1.0, 0.5, 0.6),
    "HT2": (0.4, 0.4, 0.9, 0.9),
    "HT3": (0.4, 1.0, 1.0, 0.5),
    "HT4": (1.0, 0.9, 0.8, 0.4),
}

HM_COMMUNITIES = ("HM1", "HM2", "HM3", "HM4", "HM5", "HM6")
HT_COMMUNITIES = ("HT1", "HT2", "HT3", "HT4")

#: Common target of the homogeneity-based government.
HMC_TARGET = (0.35, 0.35, 0.35, 0.35)

#: Per-landowner targets of the heterogeneity-based government.
HTC_TARGETS: dict[str, tuple[float, float, float, float]] = {
    "HT1": (0.2, 0.1, 0.8, 0.9),
    "HT2": (0.0, 0.1, 1.0, 1.0),
    "HT3": (1.0, 0.8, 0.5, 0.9),
    "HT4": (0.5, 0.2, 0.7, 0.6),
    "HM": (0.2, 0.1, 0.8, 0.9),
}

GOVERNMENTS = ("HMC", "HTC")

#: The five conditions of the main experiment (regulator during the evaluation window).
MAIN_CONDITIONS = ("no_regulator", "reward_shape", "oraa_mb", "oraa_omni", "ltos")
#: ORAA with a widened incentive bound: condition -> (ORAA form, bound).
BOUND_VARIANTS = {"oraa_mb_b3": ("oraa_mb", 3.0), "oraa_omni_b3": ("oraa_omni", 3.0)}
BOUND_CONDITIONS = tuple(BOUND_VARIANTS)
#: Every condition.
CONDITIONS = MAIN_CONDITIONS + BOUND_CONDITIONS
#: ORAA's incentive bound in the main experiment: x in [-1, 1].
DEFAULT_ORAA_BOUND = 1.0

CONDITION_LABELS = {
    "no_regulator": "No regulator",
    "reward_shape": "Reward-shape",
    "oraa_mb": "ORAA (Model-based)",
    "oraa_omni": "ORAA (Omniscient)",
    "ltos": "LToS",
    "oraa_mb_b3": "ORAA (Model-based, bound 3)",
    "oraa_omni_b3": "ORAA (Omniscient, bound 3)",
}


def base_condition(condition: str) -> str:
    """The condition without its bound variant (``oraa_mb_b3`` -> ``oraa_mb``)."""
    return BOUND_VARIANTS.get(condition, (condition, None))[0]


def oraa_bound(condition: str) -> float | None:
    """ORAA's incentive bound under ``condition``; ``None`` without ORAA."""
    if condition in BOUND_VARIANTS:
        return BOUND_VARIANTS[condition][1]
    return DEFAULT_ORAA_BOUND if condition in ("oraa_mb", "oraa_omni") else None


def community_alphas(community: str) -> tuple[float, float, float, float]:
    try:
        return COMMUNITIES[community]
    except KeyError as error:
        raise ValueError(f"unknown community {community!r}") from error


def government_targets(community: str, government: str) -> tuple[float, float, float, float]:
    """The habitat targets of ``government`` for ``community``."""
    community_alphas(community)
    if government == "HMC":
        return HMC_TARGET
    if government == "HTC":
        return HTC_TARGETS["HM" if community.startswith("HM") else community]
    raise ValueError(f"unknown government {government!r}")


@dataclass(frozen=True)
class Group:
    """A priority group of the campaign: communities x one government x conditions."""

    name: str
    priority: int
    communities: tuple[str, ...]
    government: str
    conditions: tuple[str, ...]

    @property
    def file_stem(self) -> str:
        """The group's name in output file names (``HT+HTC`` -> ``HT_HTC``)."""
        return file_stem(self.name)


def file_stem(name: str) -> str:
    return name.replace("+", "_").replace(" ", "_")


#: The campaign's priority groups, in run order.
GROUPS: tuple[Group, ...] = (
    Group("HT+HTC", 1, HT_COMMUNITIES, "HTC", MAIN_CONDITIONS),
    Group("HT+HMC", 2, HT_COMMUNITIES, "HMC", ("reward_shape", "oraa_mb", "oraa_omni", "no_regulator")),
    Group("HM+HMC", 3, HM_COMMUNITIES, "HMC", ("reward_shape", "oraa_mb", "no_regulator")),
    Group("HM+HTC", 3, HM_COMMUNITIES, "HTC", ("reward_shape", "oraa_mb", "no_regulator")),
    Group("HT+HTC bound 3", 4, HT_COMMUNITIES, "HTC", BOUND_CONDITIONS),
)

#: The sensitivity analysis on ORAA's bound: HT+HTC with both forms of ORAA at
#: bounds 1 and 3, beside the run without a regulator and reward-shape.
BOUND_COMPARISON = Group(
    "HT+HTC bounds", 4, HT_COMMUNITIES, "HTC",
    ("no_regulator", "reward_shape", "oraa_mb", "oraa_omni") + BOUND_CONDITIONS,
)

#: The groups of the analysis: the first three priority levels and the bound comparison.
ANALYSIS_GROUPS: tuple[Group, ...] = GROUPS[:4] + (BOUND_COMPARISON,)


def group_by_name(name: str) -> Group:
    for group in GROUPS + (BOUND_COMPARISON,):
        if group.name == name:
            return group
    raise ValueError(f"unknown group {name!r}")
