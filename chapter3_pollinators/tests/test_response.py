"""Response of the actors to ORAA's incentive."""

from __future__ import annotations

import numpy as np
import pytest
import torch as th

from chapter3_pollinators.analysis.response import response_to_incentive
from chapter3_pollinators.experiment.run import new_run
from chapter3_pollinators.regulators.oraa import ORAA, get_agents_land_positions

pytestmark = pytest.mark.usefixtures("cuda_required")


def test_response_matches_oraas_objective() -> None:
    run = new_run(seed=4, alphas=(0.4, 1.0, 0.5, 0.6), with_model=True)
    obs, agents = run.state.obs, run.game.agents
    xs = np.array([-1.0, -0.25, 0.0, 0.6, 1.0])
    curves = response_to_incentive(obs, agents, run.real.actors, xs)
    assert curves.shape == (4, len(xs))
    oraa = ORAA([0.0, 0.0, 0.0, 0.0])
    for i, agent in enumerate(agents):
        oraa.counter_networks = i
        oraa.agent_networks = run.real.actors
        oraa.interpreted_obs = np.array(obs)[i][0][0]
        oraa.interpreted_agent = agent
        oraa._prepare_device_inputs(agent)
        for k, x in enumerate(xs):
            trial = type("FixedTrial", (), {"suggest_float": lambda self, name, low, high, _x=x: float(_x)})()
            assert oraa.objective(trial) == pytest.approx(abs(curves[i, k]), abs=1e-7)
    habitat = np.array(obs)[1][0][0]
    layer = get_agents_land_positions(agents[1], habitat, 0.6)
    state = th.cat([th.from_numpy(habitat.copy()).float().unsqueeze(0),
                    th.from_numpy(layer.copy()).float().unsqueeze(0)]).unsqueeze(0).cuda()
    assert curves[1, 3] == pytest.approx(run.real.actors[1](state).mean().item(), abs=1e-7)


def test_response_figure_writes_files_and_index(tmp_path) -> None:
    import csv
    import json

    from chapter3_pollinators.analysis.response import GRID, response_figure
    from chapter3_pollinators.experiment.checkpoint import save_checkpoint
    from chapter3_pollinators.experiment.run import warmup_dir
    from chapter3_pollinators.experiment.settings import GROUPS, community_alphas

    run = new_run(seed=4, alphas=community_alphas("HT1"), with_model=True)
    branch = warmup_dir(tmp_path, "HT1", 0) / "branch.pt"
    branch.parent.mkdir(parents=True)
    save_checkpoint(branch, run, include_model=True)
    group = next(g for g in GROUPS if g.name == "HT+HTC")
    out = tmp_path / "out"
    paths = response_figure(tmp_path, group, out, seeds=[0, 1])
    assert sorted(p.name for p in paths) == [
        "oraa_response_HT_HTC.csv", "oraa_response_HT_HTC.pdf", "oraa_response_HT_HTC.png"]
    assert all(p.is_file() for p in paths)
    index = json.loads((out / "figures_index.json").read_text(encoding="utf-8"))
    assert sorted(index["oraa_response_HT_HTC"]["files"]) == sorted(p.name for p in paths)
    with (out / "oraa_response_HT_HTC.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 2 * len(GRID)
    assert {r["setting"] for r in rows} == {"HT1"} and {r["n"] for r in rows} == {"1"}
    assert response_figure(tmp_path, next(g for g in GROUPS if g.name == "HM+HMC"), out, seeds=[0]) == []


def test_the_bound_comparison_draws_response_curves_over_the_wide_bound(tmp_path) -> None:
    import csv
    import json

    from chapter3_pollinators.analysis.response import reachability_table, response_figure, wide_grid
    from chapter3_pollinators.experiment.checkpoint import save_checkpoint
    from chapter3_pollinators.experiment.run import warmup_dir
    from chapter3_pollinators.experiment.settings import community_alphas, group_by_name

    xs = wide_grid(3.0)
    assert xs[0] == -3.0 and xs[-1] == 3.0 and -1.0 in xs and 1.0 in xs and 0.0 in xs
    run = new_run(seed=4, alphas=community_alphas("HT2"), with_model=True)
    branch = warmup_dir(tmp_path, "HT2", 0) / "branch.pt"
    branch.parent.mkdir(parents=True)
    save_checkpoint(branch, run, include_model=True)
    group = group_by_name("HT+HTC bounds")
    out = tmp_path / "out"
    paths = response_figure(tmp_path, group, out, seeds=[0])
    assert {p.name for p in paths} == {"oraa_response_HT_HTC_bounds.pdf", "oraa_response_HT_HTC_bounds.png",
                                       "oraa_response_HT_HTC_bounds.csv"}
    with (out / "oraa_response_HT_HTC_bounds.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {float(r["x"]) for r in rows} == {float(x) for x in xs}
    paths = reachability_table(tmp_path, group, out, seeds=[0])
    assert {p.name for p in paths} == {"reachability_HT_HTC_bounds.csv", "reachability_HT_HTC_bounds.tex"}
    with (out / "reachability_HT_HTC_bounds.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 2 * 2  # landowners x actor sets x bounds
    by_key = {(r["landowner"], r["actors"], r["bound"]): r for r in rows}
    for landowner in "0123":
        for actors in ("landowners", "model"):
            narrow, wide = by_key[(landowner, actors, "1.0")], by_key[(landowner, actors, "3.0")]
            assert float(wide["gap_mean"]) <= float(narrow["gap_mean"]) + 1e-12
    tex = (out / "reachability_HT_HTC_bounds.tex").read_text()
    assert "\\begin{tabular}" in tex and "HT2" in tex
    index = json.loads((out / "figures_index.json").read_text())
    assert "oraa_response_HT_HTC_bounds" in index
