"""Aggregation, paired tests, tables and figures on a synthetic campaign."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats as scipy_stats

from chapter3_pollinators.analysis.aggregate import group_summary, paired_test, summarise
from chapter3_pollinators.analysis.figures import write_group_figures
from chapter3_pollinators.analysis.load import load_results, load_series
from chapter3_pollinators.analysis.stats import mean_se
from chapter3_pollinators.analysis.tables import paired_table_tex, results_table_tex
from chapter3_pollinators.experiment.recorder import Recorder
from chapter3_pollinators.experiment.run import branch_paths, no_regulator_paths, warmup_dir, write_json
from chapter3_pollinators.experiment.settings import government_targets, group_by_name

SEEDS = (0, 1, 2)


def _records(rng, steps, start_step, start_counter, condition, shift):
    recorder = Recorder()
    for k in range(steps):
        counter = max(0, start_counter + k)
        own = rng.uniform(0, 1, size=4)
        weights = None
        shaped = None
        transfer = None
        incentive = [0.0] * 4
        if condition in ("reward_shape", "oraa_mb", "oraa_omni", "oraa_mb_b3", "oraa_omni_b3"):
            incentive = list(rng.uniform(-0.5, 1.0, size=4) + shift)
        if condition == "ltos":
            weights = np.eye(4) * 0.8 + 0.05
            weights[0, 2] = weights[2, 0] = 0
            weights = weights / weights.sum(axis=1, keepdims=True)
            shaped = list(weights.T @ own)
            transfer = float(((1 - np.diag(weights)) * own).sum())
        recorder.append(
            step=start_step + k, counter=counter, epsilon=0.4, greedy=bool(k % 3), phase=0,
            done=False, incentive=incentive, reward=list(own), reward_plain=list(own - 0.2 + shift),
            shaped_reward=shaped, mean_action=list(rng.uniform(size=4)),
            mean_declared=list(np.clip(rng.uniform(size=4) * 0.5 + shift, 0, 1)),
            global_reward=0.1, sharing_weights=weights, transfer_volume=transfer,
        )
    return recorder


def _metrics(recorder, community, government, condition):
    from chapter3_pollinators.experiment.metrics import window_metrics

    records = recorder.arrays()
    return window_metrics(records, government_targets(community, government), condition)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    rng = np.random.default_rng(0)
    group = group_by_name("HT+HTC")
    shifts = {"no_regulator": 0.0, "reward_shape": 0.05, "oraa_mb": 0.1, "oraa_omni": 0.12, "ltos": 0.02,
              "oraa_mb_b3": 0.2, "oraa_omni_b3": 0.25}
    conditions = group.conditions + ("oraa_mb_b3", "oraa_omni_b3")
    for community in group.communities[:2]:
        for seed in SEEDS:
            warm = _records(rng, 60, 0, -40, "warmup", 0.0)
            folder = warmup_dir(root, community, seed)
            warm.save(folder / "steps.npz")
            write_json(folder / "warmup.json", {"precurriculum_step": 10, "branch_step": 60})
            for condition in conditions:
                if condition == "no_regulator":
                    rec = _records(rng, 1250, 10, 970, condition, 0.0)
                    npz, js = no_regulator_paths(root, community, seed)
                    rec.save(npz)
                    by_government = {
                        g: {"targets": list(government_targets(community, g)),
                            "metrics": _metrics(rec, community, g, condition), "metrics_greedy": None}
                        for g in ("HMC", "HTC")
                    }
                    write_json(js, {"kind": "no_regulator", "community": community, "seed": seed,
                                    "condition": condition, "by_government": by_government,
                                    "start_step": 10})
                else:
                    rec = _records(rng, 1200, 60, 1000, condition, shifts[condition])
                    npz, js = branch_paths(root, community, "HTC", condition, seed)
                    rec.save(npz)
                    write_json(js, {"kind": "branch", "community": community, "government": "HTC",
                                    "condition": condition, "seed": seed,
                                    "metrics": _metrics(rec, community, "HTC", condition),
                                    "metrics_greedy": None})
    return root


def test_mean_se_matches_numpy() -> None:
    values = [0.1, 0.4, 0.35, 0.2]
    mean, se, n = mean_se(values)
    assert mean == pytest.approx(np.mean(values))
    assert se == pytest.approx(np.std(values, ddof=1) / np.sqrt(4))
    assert n == 4
    single_mean, single_se, single_n = mean_se([0.3])
    assert single_mean == 0.3 and np.isnan(single_se) and single_n == 1


def test_paired_test_matches_scipy() -> None:
    a = {0: 0.5, 1: 0.7, 2: 0.4, 3: 0.9}
    b = {0: 0.45, 1: 0.6, 2: 0.42, 3: 0.7, 4: 1.0}
    result = paired_test(a, b)
    reference = scipy_stats.ttest_rel([0.5, 0.7, 0.4, 0.9], [0.45, 0.6, 0.42, 0.7])
    assert result["n"] == 4
    assert result["t"] == pytest.approx(reference.statistic)
    assert result["p"] == pytest.approx(reference.pvalue)
    diffs = np.array([0.05, 0.1, -0.02, 0.2])
    assert result["mean_diff"] == pytest.approx(diffs.mean())
    assert result["se_diff"] == pytest.approx(diffs.std(ddof=1) / 2)


def test_paired_test_with_identical_values_has_no_t() -> None:
    result = paired_test({0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0})
    assert result["mean_diff"] == 0 and np.isnan(result["t"]) and np.isnan(result["p"])


def test_paired_test_treats_rounding_noise_as_no_spread() -> None:
    # e.g. LToS's net budget: zero by construction, tiny rounding residues per seed
    noise = paired_test({0: 1e-17, 1: -2e-17, 2: 3e-18}, {0: 0.0, 1: 0.0, 2: 0.0})
    assert np.isnan(noise["t"]) and np.isnan(noise["p"]) and abs(noise["mean_diff"]) < 1e-16
    small = paired_test({0: 1e-4, 1: 2e-4, 2: 4e-4}, {0: 0.0, 1: 0.0, 2: 0.0})
    assert not np.isnan(small["t"]) and small["t"] == pytest.approx(
        scipy_stats.ttest_rel([1e-4, 2e-4, 4e-4], [0.0, 0.0, 0.0]).statistic)


def test_load_results_expands_no_regulator_per_government(campaign) -> None:
    results = load_results(campaign)
    no_reg = [r for r in results if r.condition == "no_regulator"]
    assert {r.government for r in no_reg} == {"HMC", "HTC"}
    assert len([r for r in results if r.government == "HTC"]) == 2 * 3 * 7


def test_group_summary_has_every_cell_and_test(campaign) -> None:
    results = load_results(campaign)
    summary = group_summary(results, group_by_name("HT+HTC"))
    cells = summary["cells"]
    assert {(c["community"], c["condition"]) for c in cells} == {
        (community, condition)
        for community in ("HT1", "HT2")
        for condition in ("no_regulator", "reward_shape", "oraa_mb", "oraa_omni", "ltos")
    }
    assert all(c["n"] == 3 for c in cells)
    ltos = next(c for c in cells if c["condition"] == "ltos")
    assert ltos["budget_net"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert "transfer_volume" in ltos
    tests = summary["paired_tests"]
    assert len(tests) == 2 * 10 * len(summary["metrics"])
    one = next(t for t in tests if t["a"] == "oraa_mb" and t["b"] == "no_regulator"
               and t["metric"] == "reward_per_agent" and t["community"] == "HT1")
    a = {r.seed: r.metrics["reward_per_agent"] for r in results
         if r.community == "HT1" and r.government == "HTC" and r.condition == "oraa_mb"}
    b = {r.seed: r.metrics["reward_per_agent"] for r in results
         if r.community == "HT1" and r.government == "HTC" and r.condition == "no_regulator"}
    assert one["mean_diff"] == pytest.approx(paired_test(a, b)["mean_diff"])
    s = summarise([r for r in results if r.government == "HTC"], "reward_per_agent")
    assert s[("HT1", "oraa_mb")]["n"] == 3


def test_tables_are_latex_with_three_decimals(campaign) -> None:
    summary = group_summary(load_results(campaign), group_by_name("HT+HTC"))
    tex = results_table_tex(summary)
    assert "\\begin{tabular}" in tex and "\\toprule" in tex and "\\bottomrule" in tex
    assert "HT1" in tex and "LToS" in tex and "$\\pm$" in tex
    numbers = [token for token in tex.replace("$", " ").split() if token.replace(".", "", 1).replace("-", "", 1).isdigit() and "." in token]
    assert numbers and all(len(token.split(".")[1]) == 3 for token in numbers)
    paired = paired_table_tex(summary)
    assert "ORAA (Model-based)" in paired and "No regulator" in paired


def test_series_and_figures_are_written(campaign, tmp_path) -> None:
    series = load_series(campaign, "HT1", "HTC", "oraa_mb", 0)
    assert series["step"][0] == 0 and series["step"][-1] == 60 + 1199
    no_reg = load_series(campaign, "HT1", "HTC", "no_regulator", 0)
    assert no_reg["step"][0] == 0 and np.all(np.diff(no_reg["step"]) == 1)
    written = write_group_figures(campaign, load_results(campaign), group_by_name("HT+HTC"), tmp_path)
    assert written
    for path in written:
        assert path.is_file() and path.stat().st_size > 0
    suffixes = {p.suffix for p in written}
    assert {".pdf", ".png", ".csv"} <= suffixes
    index = json.loads((tmp_path / "figures_index.json").read_text())
    assert index


def test_build_writes_every_output(campaign, tmp_path) -> None:
    from chapter3_pollinators.analysis.__main__ import build

    written = build(campaign, tmp_path, groups=["HT+HTC"], figures=False)
    names = {p.name for p in written}
    for expected in ("summary_HT_HTC.json", "cells_HT_HTC.csv", "per_seed_HT_HTC.csv",
                     "paired_tests_HT_HTC.csv", "summary_HT_HTC_greedy.json",
                     "table_HT_HTC.tex", "paired_HT_HTC.tex", "model_error_HT_HTC.tex"):
        assert expected in names, expected
    rows = (tmp_path / "per_seed_HT_HTC.csv").read_text().splitlines()
    assert rows[0] == "group,community,condition,seed,metric,value"
    assert len(rows) > 2 * 5 * 3 * 6
    tests = (tmp_path / "paired_tests_HT_HTC.csv").read_text().splitlines()
    assert any(line.split(",")[1] == "pooled" for line in tests[1:])
    summary = json.loads((tmp_path / "summary_HT_HTC.json").read_text())
    assert summary["pooled_tests"] and summary["pooled"]


def test_model_error_table_is_clean_latex(campaign) -> None:
    from chapter3_pollinators.analysis.tables import model_table_tex

    summary = group_summary(load_results(campaign), group_by_name("HT+HTC"))
    tex = model_table_tex(summary)
    assert tex is not None
    assert "\\begin{tabular}" in tex and "\\toprule" in tex and "\\bottomrule" in tex
    assert chr(8) not in tex and chr(9) not in tex
    assert "-0.000" not in tex


def test_ltos_incentive_falls_back_to_the_paid_incentive_outside_sharing() -> None:
    from chapter3_pollinators.experiment.metrics import incentive_series

    records = {
        "shaped_reward": np.array([[np.nan] * 4, [0.5, 0.1, 0.2, 0.2]]),
        "reward": np.array([[0.1] * 4, [0.25] * 4]),
        "incentive": np.array([[-0.25, -0.25, 0.1, 0.6], [0.0] * 4]),
    }
    out = incentive_series(records, "ltos")
    assert out[0].tolist() == [-0.25, -0.25, 0.1, 0.6]
    assert np.allclose(out[1], [0.25, -0.15, -0.05, -0.05])


def test_the_bound_comparison_has_its_table_tests_and_bar_figure(campaign, tmp_path) -> None:
    from chapter3_pollinators.analysis.__main__ import build
    from chapter3_pollinators.analysis.tables import model_table_tex

    summary = group_summary(load_results(campaign), group_by_name("HT+HTC bounds"))
    assert summary["conditions"] == ["no_regulator", "reward_shape", "oraa_mb", "oraa_omni",
                                     "oraa_mb_b3", "oraa_omni_b3"]
    pairs = {(t["a"], t["b"]) for t in summary["paired_tests"]}
    assert {("oraa_mb_b3", "oraa_mb"), ("oraa_omni_b3", "oraa_omni"),
            ("oraa_mb_b3", "no_regulator"), ("oraa_omni_b3", "no_regulator")} <= pairs
    model = model_table_tex(summary)
    assert "ORAA (Model-based, bound 3)" in model and "ORAA (Omniscient, bound 3)" in model
    written = {p.name for p in build(campaign, tmp_path, groups=["HT+HTC bounds"], figures=True)}
    for expected in ("table_HT_HTC_bounds.tex", "paired_HT_HTC_bounds.tex", "paired_budget_HT_HTC_bounds.tex",
                     "model_error_HT_HTC_bounds.tex", "summary_HT_HTC_bounds.json",
                     "bars_HT_HTC_bounds.pdf", "bars_HT_HTC_bounds.png", "bars_HT_HTC_bounds.csv"):
        assert expected in written, expected
    assert not any(name.startswith(("timeseries_", "budget_cumulative_")) for name in written)
    paired = (tmp_path / "paired_HT_HTC_bounds.tex").read_text()
    assert "ORAA (Model-based, bound 3) $-$ ORAA (Model-based)" in paired
    assert "ORAA (Omniscient, bound 3) $-$ No regulator" in paired
    budget = (tmp_path / "paired_budget_HT_HTC_bounds.tex").read_text()
    assert "Gross" in budget and "Penalties" in budget
    bars = (tmp_path / "bars_HT_HTC_bounds.csv").read_text()
    assert "budget_penalties" in bars  # the synthetic runs carry no ORAA objective
