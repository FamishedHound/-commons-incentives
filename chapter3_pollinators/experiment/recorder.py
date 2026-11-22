"""Per-step records of a run.

Every environment step stores one row; fields that do not apply to a run (the
model error without a model, ORAA's best values outside ORAA, the sharing
weights outside LToS) are ``NaN``.

======================  =======  ===================================================
field                   shape    meaning
======================  =======  ===================================================
step                    ()       environment step index from the start of the run
counter                 ()       counted step (0 until epsilon reaches 0.4)
epsilon                 ()       exploration rate used at this step
greedy                  ()       True when the landowners acted greedily
phase                   ()       regulator code, see ``schedule.PHASE_CODES``
done                    ()       True when the pollinators died out at this step
incentive               (4,)     external incentive paid to each landowner
reward                  (4,)     reward each landowner received (incentive incl.,
                                 clipped at 0); under LToS the reward before sharing
reward_plain            (4,)     reward without incentive and before the clip
shaped_reward           (4,)     LToS: reward after sharing (what the landowner
                                 trained on)
mean_action             (4,)     mean continuous action over the landowner's cells
mean_declared           (4,)     mean declared habitat share (after the 0.1 snap)
global_reward           ()       G, the global pollination term
model_distance_taken    ()       sum |round(a,1) - round(model,1)|, taken action
model_distance_greedy   ()       the same with the landowners' greedy action
real_greedy_mean        (4,)     mean greedy action of the landowners' actors
model_mean              (4,)     mean action of the model's actors on the same input
oraa_best               (4,)     ORAA: best objective value |target - mean(actor)|
sharing_weights         (4, 4)   LToS: w[i, j] = share of i's reward given to j
transfer_volume         ()       LToS: sum_i sum_{j != i} w[i, j] * reward_i
======================  =======  ===================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chapter3_pollinators.experiment import atomic

FIELDS: dict[str, tuple[tuple[int, ...], type]] = {
    "step": ((), np.int32),
    "counter": ((), np.int32),
    "epsilon": ((), np.float64),
    "greedy": ((), np.bool_),
    "phase": ((), np.int8),
    "done": ((), np.bool_),
    "incentive": ((4,), np.float64),
    "reward": ((4,), np.float64),
    "reward_plain": ((4,), np.float64),
    "shaped_reward": ((4,), np.float64),
    "mean_action": ((4,), np.float64),
    "mean_declared": ((4,), np.float64),
    "global_reward": ((), np.float64),
    "model_distance_taken": ((), np.float64),
    "model_distance_greedy": ((), np.float64),
    "real_greedy_mean": ((4,), np.float64),
    "model_mean": ((4,), np.float64),
    "oraa_best": ((4,), np.float64),
    "sharing_weights": ((4, 4), np.float64),
    "transfer_volume": ((), np.float64),
}


class Recorder:
    """Accumulates per-step rows and writes them as one compressed ``.npz``."""

    def __init__(self) -> None:
        self._rows: dict[str, list] = {name: [] for name in FIELDS}

    def __len__(self) -> int:
        return len(self._rows["step"])

    def append(self, **values) -> None:
        unknown = set(values) - set(FIELDS)
        if unknown:
            raise KeyError(f"unknown record fields: {sorted(unknown)}")
        for name, (shape, _dtype) in FIELDS.items():
            value = values.get(name)
            if value is None:
                value = np.full(shape, np.nan) if shape else np.nan
            self._rows[name].append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for name, (shape, dtype) in FIELDS.items():
            rows = self._rows[name]
            if not rows:
                out[name] = np.zeros((0, *shape), dtype=dtype)
                continue
            out[name] = np.asarray(rows, dtype=dtype)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **self.arrays())
        atomic.replace_file(tmp, path)


def load_records(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}
