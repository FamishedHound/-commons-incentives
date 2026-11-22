"""Hyper-parameters of the LToS high level.

Values are those of the *Traffic 6 x 6* column of Yi et al., Tables 4 and 5
(the published scenario with a fixed spatial graph of agents), read as follows
for a run that is one continuous sequence of steps:

==========================  =============  ===========================================
parameter                   value          source
==========================  =============  ===========================================
initial selfishness         0.8            Table 5 (selfishness initialiser, App. C)
action interval M           15 steps       Table 5 (weights held for M steps)
update frequency            every 20 steps Table 5 gives 20 *episodes*; a run here has
                                           no episodes, so it is read as 20 steps
sample size / batch size    1,000 / 20     Table 5; one pass over the sample per update
optimiser, learning rate    SGD, 1e-3      Table 5
exploration noise           OU, 0.25 eps   Table 5 (eps = the landowners' epsilon)
last-layer activation       softmax        Table 5
MLP units / activation      32 & 32, ReLU  Table 4 (the paper's MLP size)
initialiser                 random normal  Table 4
target network              soft, tau 0.1  Appendix D and Table 4
replay capacity             10,000         Table 4
==========================  =============  ===========================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LToSConfig:
    selfishness: float = 0.8
    action_interval: int = 15
    update_interval: int = 20
    sample_size: int = 1000
    batch_size: int = 20
    optimiser: str = "SGD"
    learning_rate: float = 1e-3
    ou_sigma: float = 0.25
    ou_theta: float = 0.15
    ou_dt: float = 1e-2
    tau: float = 0.1
    hidden_units: tuple[int, ...] = (32, 32)
    init_std: float = 0.05
    capacity: int = 10_000

    def __post_init__(self) -> None:
        if not 0.0 < self.selfishness < 1.0:
            raise ValueError("selfishness must lie strictly between 0 and 1")
        if self.action_interval < 1 or self.update_interval < 1:
            raise ValueError("intervals must be positive")
        if self.batch_size < 1 or self.sample_size < self.batch_size:
            raise ValueError("need sample_size >= batch_size >= 1")
        if self.optimiser != "SGD":
            raise ValueError("the high level is trained with SGD")
