"""Per-group summaries: mean +- SE over seeds per setting and condition, and paired tests.

Paired t-tests compare two conditions within one setting over the seeds both
have; the conditions of a setting share their seeds and, for the regulated
conditions, their warm-up checkpoint.  p-values are not corrected for multiple
comparisons.  ``pooled`` entries average each seed's values over the group's settings that
have every condition (listed as ``pooled_settings``) and are reported beside,
never instead of, the per-setting ones.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Iterable
from pathlib import Path

from chapter3_pollinators.analysis.load import RunResult
from chapter3_pollinators.analysis.stats import mean_se, paired_test
from chapter3_pollinators.experiment.settings import CONDITION_LABELS, Group, file_stem

#: The metrics of the result tables and of the paired tests.
PRIMARY_METRICS = (
    "reward_per_agent",
    "pollination_difference_pp",
    "budget_net",
    "budget_gross",
    "budget_penalties",
    "cumulative_net_budget",
)
#: Further metrics reported in the summaries when a run has them.
EXTRA_METRICS = (
    "received_reward_per_agent",
    "pollination_difference",
    "pollination_difference_action",
    "signed_target_gap",
    "mean_declared_habitat",
    "global_pollination_reward",
    "greedy_fraction",
    "transfer_volume",
    "selfishness",
    "model_distance_greedy",
    "model_distance_taken",
    "model_mean_action_error",
    "oraa_best_objective",
)
CONDITION_ORDER = ("no_regulator", "reward_shape", "oraa_mb", "oraa_omni", "oraa_mb_b3", "oraa_omni_b3", "ltos")
#: Orientation of a paired comparison: ``a`` is the condition earlier in this list.
PAIR_ORDER = ("oraa_mb_b3", "oraa_omni_b3", "oraa_mb", "oraa_omni", "ltos", "reward_shape", "no_regulator")

#: The comparisons shown in the paired-test tables, as (a, b) for a - b.
MAIN_COMPARISONS = {
    "HT+HTC": (
        ("oraa_mb", "no_regulator"), ("reward_shape", "no_regulator"),
        ("oraa_omni", "no_regulator"), ("ltos", "no_regulator"),
        ("oraa_mb", "reward_shape"), ("oraa_mb", "oraa_omni"), ("oraa_mb", "ltos"),
    ),
    "HT+HMC": (
        ("oraa_mb", "no_regulator"), ("reward_shape", "no_regulator"),
        ("oraa_omni", "no_regulator"), ("oraa_mb", "reward_shape"), ("oraa_mb", "oraa_omni"),
    ),
    "HM+HMC": (("oraa_mb", "no_regulator"), ("reward_shape", "no_regulator"), ("oraa_mb", "reward_shape")),
    "HM+HTC": (("oraa_mb", "no_regulator"), ("reward_shape", "no_regulator"), ("oraa_mb", "reward_shape")),
    "HT+HTC bounds": (
        ("reward_shape", "no_regulator"), ("oraa_mb", "no_regulator"), ("oraa_omni", "no_regulator"),
        ("oraa_mb_b3", "no_regulator"), ("oraa_omni_b3", "no_regulator"),
        ("oraa_mb_b3", "oraa_mb"), ("oraa_omni_b3", "oraa_omni"),
    ),
}


def select(results: Iterable[RunResult], group: Group) -> list[RunResult]:
    return [
        r for r in results
        if r.community in group.communities and r.government == group.government
        and r.condition in group.conditions
    ]


def _metrics_of(result: RunResult, greedy: bool) -> dict | None:
    return result.metrics_greedy if greedy else result.metrics


def summarise(results: Iterable[RunResult], metric: str, greedy: bool = False) -> dict:
    """``{(community, condition): {mean, se, n, values}}`` for one metric."""
    values: dict[tuple[str, str], dict[int, float]] = {}
    for r in results:
        metrics = _metrics_of(r, greedy)
        if metrics is None or metric not in metrics:
            continue
        values.setdefault((r.community, r.condition), {})[r.seed] = float(metrics[metric])
    out = {}
    for key, per_seed in values.items():
        mean, se, n = mean_se(per_seed.values())
        out[key] = {"mean": mean, "se": se, "n": n, "values": dict(sorted(per_seed.items()))}
    return out


def _pairs(conditions: Iterable[str]) -> list[tuple[str, str]]:
    ordered = [c for c in PAIR_ORDER if c in set(conditions)]
    return [(a, b) for a, b in itertools.combinations(ordered, 2)]


def _pooled_values(summary: dict, communities: Iterable[str], condition: str) -> dict[int, float]:
    per_setting = [summary.get((c, condition), {}).get("values", {}) for c in communities]
    if not per_setting or any(not v for v in per_setting):
        return {}
    seeds = set.intersection(*(set(v) for v in per_setting))
    return {s: sum(v[s] for v in per_setting) / len(per_setting) for s in sorted(seeds)}


def group_summary(results: Iterable[RunResult], group: Group, greedy: bool = False) -> dict:
    chosen = select(results, group)
    present = [c for c in CONDITION_ORDER if c in group.conditions]
    tables = {m: summarise(chosen, m, greedy) for m in PRIMARY_METRICS + EXTRA_METRICS}
    cells = []
    for community in group.communities:
        for condition in present:
            n = tables["reward_per_agent"].get((community, condition), {}).get("n", 0)
            if n == 0:
                continue
            cell = {"community": community, "condition": condition,
                    "label": CONDITION_LABELS[condition], "n": n,
                    "seeds": sorted(tables["reward_per_agent"][(community, condition)]["values"])}
            for metric, table in tables.items():
                if (community, condition) in table:
                    cell[metric] = table[(community, condition)]
            cells.append(cell)
    tests = []
    pooled_cells, pooled_tests = [], []
    pooled_settings = [
        c for c in group.communities
        if all((c, cond) in tables["reward_per_agent"] for cond in present)
    ]
    for metric in PRIMARY_METRICS:
        table = tables[metric]
        for community in group.communities:
            for a, b in _pairs(present):
                if (community, a) in table and (community, b) in table:
                    result = paired_test(table[(community, a)]["values"], table[(community, b)]["values"])
                    tests.append({"community": community, "metric": metric, "a": a, "b": b, **result})
        pooled = {c: _pooled_values(table, pooled_settings, c) for c in present} if pooled_settings else {}
        for condition, values in pooled.items():
            if values:
                mean, se, n = mean_se(values.values())
                pooled_cells.append({"metric": metric, "condition": condition, "mean": mean,
                                     "se": se, "n": n, "values": values})
        for a, b in _pairs(present):
            if pooled.get(a) and pooled.get(b):
                pooled_tests.append({"community": "pooled", "metric": metric, "a": a, "b": b,
                                     **paired_test(pooled[a], pooled[b])})
    return {
        "group": group.name,
        "government": group.government,
        "communities": list(group.communities),
        "conditions": present,
        "greedy_only": greedy,
        "metrics": list(PRIMARY_METRICS),
        "cells": cells,
        "paired_tests": tests,
        "pooled_settings": pooled_settings,
        "pooled": pooled_cells,
        "pooled_tests": pooled_tests,
    }


def write_group_outputs(summary: dict, out_dir: Path, suffix: str = "") -> list[Path]:
    """``summary.json``, cells / per-seed / paired-test CSVs for one group."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = file_stem(summary["group"]) + suffix
    paths = []
    path = out_dir / f"summary_{name}.json"
    path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    paths.append(path)

    metrics = PRIMARY_METRICS + EXTRA_METRICS
    path = out_dir / f"cells_{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "community", "condition", "n", "metric", "mean", "se"])
        for cell in summary["cells"]:
            for metric in metrics:
                if metric in cell:
                    writer.writerow([summary["group"], cell["community"], cell["condition"], cell["n"],
                                     metric, cell[metric]["mean"], cell[metric]["se"]])
    paths.append(path)

    path = out_dir / f"per_seed_{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "community", "condition", "seed", "metric", "value"])
        for cell in summary["cells"]:
            for metric in metrics:
                for seed, value in cell.get(metric, {}).get("values", {}).items():
                    writer.writerow([summary["group"], cell["community"], cell["condition"], seed,
                                     metric, value])
    paths.append(path)

    path = out_dir / f"paired_tests_{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "community", "metric", "a", "b", "n", "mean_diff", "se_diff",
                         "t", "df", "p"])
        for test in summary["paired_tests"] + summary["pooled_tests"]:
            writer.writerow([summary["group"], test["community"], test["metric"], test["a"], test["b"],
                             test["n"], test["mean_diff"], test["se_diff"], test["t"], test["df"],
                             test["p"]])
    paths.append(path)
    return paths
