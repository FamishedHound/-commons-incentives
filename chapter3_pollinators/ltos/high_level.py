"""The high-level sharing policy of one landowner and its exploration noise."""

from __future__ import annotations

import math

import numpy as np
import torch as th
import torch.nn as nn

#: Input of the high level: the habitat of the 16 cells, zero outside the neighbourhood.
HIGH_INPUT = 16


def selfishness_logit(selfishness: float, size: int) -> float:
    """Self logit that makes a softmax keep ``selfishness`` when the other logits are 0."""
    return math.log(selfishness / (1.0 - selfishness) * (size - 1))


class SharingActor(nn.Module):
    """``phi_i``: neighbourhood habitat -> logits over ``N_i`` (the softmax gives ``w_i^out``).

    Weights are drawn from ``N(0, init_std^2)`` and biases are zero, except the
    last layer's bias on the landowner's own slot, which is set by
    :func:`selfishness_logit` so that the landowner initially keeps
    ``selfishness`` of its reward and splits the rest evenly.
    """

    def __init__(
        self,
        size: int,
        self_slot: int,
        selfishness: float,
        generator: th.Generator,
        hidden_units: tuple[int, ...] = (32, 32),
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = HIGH_INPUT
        for units in hidden_units:
            layers += [nn.Linear(width, units), nn.ReLU()]
            width = units
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, size)
        with th.no_grad():
            for module in list(self.body) + [self.head]:
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0.0, init_std, generator=generator)
                    module.bias.zero_()
            self.head.bias[self_slot] = selfishness_logit(selfishness, size)
        self.size = size
        self.self_slot = self_slot

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.head(self.body(x))


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck exploration noise on the logits.

    ``x <- x + theta (mu - x) dt + sigma sqrt(dt) N(0, 1)``, started at
    ``x0 = mu = sigma N(0, 1)`` (one draw per run), as in the LToS release.
    """

    def __init__(
        self,
        size: int,
        sigma: float,
        rng: np.random.Generator,
        theta: float = 0.15,
        dt: float = 1e-2,
    ) -> None:
        self.rng = rng
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.mu = sigma * rng.standard_normal(size)
        self.x_prev = self.mu.copy()

    def sample(self) -> np.ndarray:
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.mu.shape)
        )
        self.x_prev = x
        return x


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with th.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1.0 - tau).add_(s, alpha=tau)
