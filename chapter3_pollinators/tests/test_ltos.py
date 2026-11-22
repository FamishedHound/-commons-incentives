"""LToS: the property graph, sharing weights, shaped rewards, transfers and the bi-level update."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch as th

from chapter3_pollinators.env.game import PollinatorsGame
from chapter3_pollinators.env.properties import PROPERTY_CELLS
from chapter3_pollinators.experiment.metrics import window_metrics
from chapter3_pollinators.ltos.batched import actor_batch, critic_batch
from chapter3_pollinators.ltos.config import LToSConfig
from chapter3_pollinators.ltos.graph import neighbourhoods, property_adjacency
from chapter3_pollinators.ltos.high_level import OrnsteinUhlenbeckNoise, SharingActor, selfishness_logit
from chapter3_pollinators.ltos.sharing import (
    exchange_gradients,
    incoming_layer,
    net_received,
    shaped_rewards,
    transfer_volume,
    weight_matrix,
)

NEIGHBOURHOODS = neighbourhoods(property_adjacency(PROPERTY_CELLS))


def _random_weight_matrix(rng: np.random.Generator) -> np.ndarray:
    outs = [rng.dirichlet(np.ones(len(n))) for n in NEIGHBOURHOODS]
    return weight_matrix(outs, NEIGHBOURHOODS)


# --- graph ------------------------------------------------------------------
def test_properties_are_neighbours_when_they_share_a_cell_edge() -> None:
    adjacency = property_adjacency(PROPERTY_CELLS)
    expected = {0: {1, 3}, 1: {0, 2, 3}, 2: {1, 3}, 3: {0, 1, 2}}
    for i in range(4):
        assert {j for j in range(4) if adjacency[i, j]} == expected[i]
    assert np.array_equal(adjacency, adjacency.T)
    assert not adjacency.diagonal().any()
    assert NEIGHBOURHOODS == ((0, 1, 3), (0, 1, 2, 3), (1, 2, 3), (0, 1, 2, 3))


def test_diagonal_contact_alone_does_not_make_neighbours() -> None:
    cells = (((0, 0),), ((1, 1),))
    assert not property_adjacency(cells).any()


# --- weights -----------------------------------------------------------------
@pytest.mark.parametrize("agent", range(4))
def test_sharing_actor_outputs_lie_on_the_simplex(agent: int) -> None:
    generator = th.Generator().manual_seed(agent)
    actor = SharingActor(len(NEIGHBOURHOODS[agent]), NEIGHBOURHOODS[agent].index(agent), 0.8, generator)
    inputs = th.rand((64, 16), generator=generator)
    weights = th.softmax(actor(inputs), dim=-1)
    assert weights.shape == (64, len(NEIGHBOURHOODS[agent]))
    assert th.all(weights >= 0)
    assert th.allclose(weights.sum(dim=-1), th.ones(64), atol=1e-6)


@pytest.mark.parametrize("agent", range(4))
def test_selfishness_initialiser_keeps_the_initial_share(agent: int) -> None:
    size = len(NEIGHBOURHOODS[agent])
    self_slot = NEIGHBOURHOODS[agent].index(agent)
    actor = SharingActor(size, self_slot, 0.8, th.Generator().manual_seed(3))
    weights = th.softmax(actor(th.zeros((1, 16))), dim=-1)[0]
    assert weights[self_slot].item() == pytest.approx(0.8, abs=1e-6)
    others = [w.item() for k, w in enumerate(weights) if k != self_slot]
    assert others == pytest.approx([0.2 / (size - 1)] * (size - 1), abs=1e-6)
    assert selfishness_logit(0.8, size) == pytest.approx(np.log(0.8 / 0.2 * (size - 1)))


def test_weight_matrix_is_zero_outside_neighbourhoods_and_rows_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = _random_weight_matrix(rng)
        assert w[0, 2] == 0 and w[2, 0] == 0
        assert np.allclose(w.sum(axis=1), 1.0)
        assert np.all(w >= 0)


# --- shaped reward and transfers -----------------------------------------------
def test_shaped_reward_is_the_sum_of_received_shares() -> None:
    w = np.zeros((4, 4))
    w[0, [0, 1, 3]] = (0.5, 0.25, 0.25)
    w[1, [0, 1, 2, 3]] = (0.1, 0.6, 0.2, 0.1)
    w[2, [1, 2, 3]] = (0.3, 0.4, 0.3)
    w[3, [0, 1, 2, 3]] = (0.25, 0.25, 0.25, 0.25)
    own = [1.0, 2.0, 0.0, 4.0]
    shaped = shaped_rewards(own, w)
    assert shaped == pytest.approx([0.5 + 0.2 + 1.0, 0.25 + 1.2 + 1.0, 0.4 + 1.0, 0.25 + 0.2 + 1.0])


def test_sharing_conserves_the_total_reward() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = _random_weight_matrix(rng)
        own = rng.uniform(0, 1, size=4)
        shaped = shaped_rewards(own, w)
        assert sum(shaped) == pytest.approx(float(own.sum()), abs=1e-12)
        assert sum(net_received(own, w)) == pytest.approx(0.0, abs=1e-12)


def test_transfer_volume_counts_every_share_given_to_another_landowner() -> None:
    rng = np.random.default_rng(2)
    for _ in range(100):
        w = _random_weight_matrix(rng)
        own = rng.uniform(0, 1, size=4)
        expected = sum(w[i, j] * own[i] for i in range(4) for j in range(4) if j != i)
        volume = transfer_volume(own, w)
        assert volume == pytest.approx(expected, abs=1e-12)
        assert volume == pytest.approx(float(((1 - np.diag(w)) * own).sum()), abs=1e-12)
        assert volume >= sum(max(x, 0.0) for x in net_received(own, w)) - 1e-12
    identity = np.eye(4)
    assert transfer_volume([1, 2, 3, 4], identity) == 0.0
    assert shaped_rewards([1, 2, 3, 4], identity) == [1, 2, 3, 4]


def test_window_metrics_report_transfers_with_zero_net_budget() -> None:
    rng = np.random.default_rng(3)
    steps = 1300
    records = {name: np.full((steps, 4), np.nan) for name in ("real_greedy_mean", "model_mean", "oraa_best")}
    weights = np.stack([_random_weight_matrix(rng) for _ in range(steps)])
    own = rng.uniform(0, 1, size=(steps, 4))
    shaped = np.einsum("tij,ti->tj", weights, own)
    records.update(
        counter=np.arange(900, 900 + steps),
        greedy=rng.uniform(size=steps) > 0.4,
        incentive=np.zeros((steps, 4)),
        reward=own,
        shaped_reward=shaped,
        reward_plain=own - 0.1,
        mean_declared=rng.uniform(size=(steps, 4)),
        mean_action=rng.uniform(size=(steps, 4)),
        global_reward=rng.uniform(size=steps),
        sharing_weights=weights,
        transfer_volume=np.array([transfer_volume(o, w) for o, w in zip(own, weights)]),
        model_distance_greedy=np.full(steps, np.nan),
        model_distance_taken=np.full(steps, np.nan),
    )
    metrics = window_metrics(records, (0.2, 0.1, 0.8, 0.9), "ltos")
    assert metrics["steps"] == 1200
    assert metrics["budget_net"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["budget_gross"] == pytest.approx(-metrics["budget_penalties"], abs=1e-12)
    assert metrics["transfer_volume"] >= metrics["budget_gross"] - 1e-12
    assert 0 < metrics["selfishness"] < 1


# --- the incoming-weight layer --------------------------------------------------
def test_incoming_layer_matches_the_games_incentive_layer_orientation() -> None:
    random.seed(0)
    game = PollinatorsGame()
    incentive = [0.11, 0.22, 0.33, 0.44]
    local, global_state = game.create_observation(incentive)
    diagonal = np.diag(incentive)
    for j in range(4):
        layer = incoming_layer(diagonal, j, PROPERTY_CELLS)
        assert np.array_equal(layer, local[j][0][1])
    full = np.zeros((4, 4))
    for i in range(4):
        full[i, i] = incentive[i]
    combined = sum(incoming_layer(full, j, PROPERTY_CELLS) for j in range(4))
    assert np.allclose(combined, global_state[1])


def test_incoming_layer_puts_each_share_on_the_givers_cells_only() -> None:
    w = _random_weight_matrix(np.random.default_rng(4))
    for j in range(4):
        layer = np.rot90(incoming_layer(w, j, PROPERTY_CELLS), k=-1)
        for i, cells in enumerate(PROPERTY_CELLS):
            for x, y in cells:
                assert layer[x, y] == w[i, j]
                if i not in NEIGHBOURHOODS[j]:
                    assert layer[x, y] == 0


# --- the high level -----------------------------------------------------------
def test_gradient_exchange_equals_the_joint_gradient() -> None:
    """Sending g_j^in back to the givers gives d(sum_j v_j)/d w_i^out."""
    th.manual_seed(0)
    batch = 5
    logits = [th.randn(batch, len(n), requires_grad=True) for n in NEIGHBOURHOODS]
    coefficients = th.randn(4, 4, batch)

    def values(w_in_list):
        return sum(
            (coefficients[j, : w.shape[1]].T * w * w).sum() for j, w in enumerate(w_in_list)
        )

    w_out = [th.softmax(x, dim=-1) for x in logits]
    w = th.zeros(batch, 4, 4)
    for i, n in enumerate(NEIGHBOURHOODS):
        for slot, j in enumerate(n):
            w[:, i, j] = w_out[i][:, slot]
    w_in = [w[:, list(n), j] for j, n in enumerate(NEIGHBOURHOODS)]
    joint = th.autograd.grad(values(w_in), w_out, retain_graph=True)

    leaves = [x.detach().clone().requires_grad_(True) for x in w_in]
    g_in = th.autograd.grad(values(leaves), leaves)
    g_out = exchange_gradients([g.numpy() for g in g_in], NEIGHBOURHOODS)
    for i in range(4):
        assert np.allclose(g_out[i], joint[i].numpy(), atol=1e-6)


def test_ou_noise_is_seeded_and_scaled() -> None:
    a = OrnsteinUhlenbeckNoise(3, sigma=0.25, rng=np.random.default_rng(5))
    b = OrnsteinUhlenbeckNoise(3, sigma=0.25, rng=np.random.default_rng(5))
    assert np.array_equal(a.sample(), b.sample())
    assert np.array_equal(a.mu, b.mu)
    quiet = OrnsteinUhlenbeckNoise(3, sigma=0.0, rng=np.random.default_rng(5))
    assert np.array_equal(quiet.sample(), np.zeros(3))


def test_config_carries_the_published_traffic_values() -> None:
    config = LToSConfig()
    assert config.selfishness == 0.8
    assert config.action_interval == 15
    assert config.update_interval == 20
    assert config.sample_size == 1000 and config.batch_size == 20
    assert config.learning_rate == 1e-3 and config.optimiser == "SGD"
    assert config.ou_sigma == 0.25 and config.tau == 0.1
    assert config.hidden_units == (32, 32)


@pytest.mark.usefixtures("cuda_required")
def test_batched_networks_match_the_per_sample_networks() -> None:
    from chapter3_pollinators.learning.networks import Actor, Critic

    th.manual_seed(0)
    actor = Actor(5, 1).cuda()
    critic = Critic(5, 5, 16).cuda()
    states = th.rand(7, 2, 4, 4).cuda()
    actions = th.rand(7, 5).cuda()
    per_sample_actions = th.stack([actor(s) for s in states])
    per_sample_q = th.stack([critic(s, a) for s, a in zip(states, actions)]).squeeze(-1)
    assert th.allclose(actor_batch(actor, states), per_sample_actions, atol=1e-6)
    assert th.allclose(critic_batch(critic, states, actions), per_sample_q, atol=1e-5)
