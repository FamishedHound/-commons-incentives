"""The run pipeline on a compressed horizon: warm-up, every branch, and the run without a regulator.

The warm-up is started at epsilon 0.4 and counter 975 (so it reaches counted
step 999 in 24 steps) and the branches stop at counter 1,012; everything else is
the production code path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from chapter3_pollinators.experiment.recorder import load_records
from chapter3_pollinators.experiment.run import (
    branch_paths,
    no_regulator_paths,
    run_branch,
    run_no_regulator,
    run_warmup,
    warmup_dir,
)
from chapter3_pollinators.experiment.schedule import PHASE_CODES
from chapter3_pollinators.experiment.settings import government_targets

pytestmark = pytest.mark.usefixtures("cuda_required")

COMMUNITY, SEED, LAST = "HT2", 5, 1012


def _fast_forward(run) -> None:
    run.state.epsilon = 0.4
    run.state.counters = 975


@pytest.fixture(scope="module")
def campaign_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("campaign")
    run_warmup(COMMUNITY, SEED, root, prepare=_fast_forward)
    return root


def test_warmup_writes_both_checkpoints_and_records(campaign_root) -> None:
    folder = warmup_dir(campaign_root, COMMUNITY, SEED)
    for name in ("precurriculum.pt", "branch.pt", "steps.npz", "warmup.json"):
        assert (folder / name).is_file(), name
    record = json.loads((folder / "warmup.json").read_text())
    assert record["precurriculum_step"] == 0
    assert record["branch_step"] == record["steps"] == 24
    records = load_records(folder / "steps.npz")
    assert records["counter"].tolist() == list(range(976, 1000))
    assert set(records["phase"].tolist()) == {PHASE_CODES["rewarding"]}


@pytest.mark.parametrize("condition", ["reward_shape", "oraa_mb", "oraa_omni", "ltos", "oraa_mb_b3", "oraa_omni_b3"])
def test_each_branch_runs_the_evaluation_window(campaign_root, condition: str) -> None:
    government = "HTC"
    record = run_branch(
        COMMUNITY, government, condition, SEED, campaign_root, oraa_trials=6, last_counter=LAST
    )
    npz_path, json_path = branch_paths(campaign_root, COMMUNITY, government, condition, SEED)
    assert json.loads(json_path.read_text())["condition"] == condition
    records = load_records(npz_path)
    assert records["counter"].tolist() == list(range(1000, LAST + 1))
    assert record["metrics"]["steps"] == LAST - 999
    targets = np.asarray(government_targets(COMMUNITY, government))
    if condition == "reward_shape":
        expected = 1 - np.abs(targets[None, :] - records["mean_action"])
        assert np.allclose(records["incentive"], expected)
        assert set(records["phase"].tolist()) == {PHASE_CODES["reward_shape"]}
    if condition.startswith("oraa"):
        bound = 3.0 if condition.endswith("_b3") else 1.0
        assert record["oraa_bound"] == bound
        assert np.all(np.abs(records["incentive"]) <= bound)
        assert not np.isnan(records["oraa_best"]).any()
        assert set(records["phase"].tolist()) == {PHASE_CODES["oraa"]}
        has_model = not np.isnan(records["model_distance_greedy"]).all()
        assert has_model == condition.startswith("oraa_mb")
    else:
        assert record["oraa_bound"] is None
    if condition == "ltos":
        assert np.all(records["incentive"] == 0)
        assert np.allclose(records["shaped_reward"].sum(axis=1), records["reward"].sum(axis=1))
        w = records["sharing_weights"]
        assert np.allclose(w.sum(axis=2), 1.0) and np.all(w >= 0)
        assert np.all(w[:, 0, 2] == 0) and np.all(w[:, 2, 0] == 0)
        assert np.all(w == w[0])
        assert record["metrics"]["budget_net"] == pytest.approx(0.0, abs=1e-12)
        assert record["ltos"]["neighbourhoods"] == [[0, 1, 3], [0, 1, 2, 3], [1, 2, 3], [0, 1, 2, 3]]


def test_ltos_holds_weights_for_fifteen_steps_and_updates_every_twenty(campaign_root) -> None:
    record = run_branch(COMMUNITY, "HTC", "ltos", SEED, campaign_root, last_counter=1021)
    npz_path, _ = branch_paths(campaign_root, COMMUNITY, "HTC", "ltos", SEED)
    w = load_records(npz_path)["sharing_weights"]
    assert len(w) == 22
    assert np.all(w[:15] == w[0])
    assert not np.array_equal(w[15], w[14])
    assert np.all(w[15:] == w[15])
    assert record["ltos"]["high_level_updates"] == 1
    assert record["ltos"]["mean_abs_exchanged_gradient"] > 0


def test_branches_share_their_starting_state(campaign_root) -> None:
    first_greedy = []
    for condition in ("reward_shape", "oraa_omni", "ltos"):
        run_branch(COMMUNITY, "HMC", condition, SEED, campaign_root, oraa_trials=4, last_counter=1001)
        npz_path, _ = branch_paths(campaign_root, COMMUNITY, "HMC", condition, SEED)
        first_greedy.append(bool(load_records(npz_path)["greedy"][0]))
    assert len(set(first_greedy)) == 1


def test_a_branch_is_reproducible(campaign_root) -> None:
    arrays = []
    for _ in range(2):
        run_branch(COMMUNITY, "HTC", "ltos", SEED, campaign_root, last_counter=1006)
        npz_path, _ = branch_paths(campaign_root, COMMUNITY, "HTC", "ltos", SEED)
        arrays.append(load_records(npz_path))
    for name in arrays[0]:
        assert np.array_equal(arrays[0][name], arrays[1][name], equal_nan=True), name


def test_no_regulator_pays_nothing_and_reports_both_governments(campaign_root) -> None:
    record = run_no_regulator(COMMUNITY, SEED, campaign_root, last_counter=LAST)
    npz_path, _ = no_regulator_paths(campaign_root, COMMUNITY, SEED)
    records = load_records(npz_path)
    assert records["step"][0] == 0
    assert records["counter"][-1] == LAST
    assert np.all(records["incentive"] == 0)
    assert set(records["phase"].tolist()) == {PHASE_CODES["none"]}
    assert set(record["by_government"]) == {"HMC", "HTC"}
    for government, block in record["by_government"].items():
        assert block["metrics"]["budget_net"] == 0.0
        assert block["targets"] == list(government_targets(COMMUNITY, government))
