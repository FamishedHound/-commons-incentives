"""Read finished runs back from a campaign folder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chapter3_pollinators.experiment.recorder import load_records
from chapter3_pollinators.experiment.run import branch_paths, no_regulator_paths, warmup_dir


@dataclass(frozen=True)
class RunResult:
    """The window metrics of one run under one government."""

    community: str
    government: str
    condition: str
    seed: int
    metrics: dict
    metrics_greedy: dict | None
    records_path: Path


def load_results(root: Path) -> list[RunResult]:
    """Every finished run; a run without a regulator appears once per government."""
    results = []
    for path in sorted((Path(root) / "runs").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        npz = path.with_suffix(".npz")
        if payload.get("kind") == "branch" and payload.get("metrics"):
            results.append(
                RunResult(payload["community"], payload["government"], payload["condition"],
                          int(payload["seed"]), payload["metrics"], payload.get("metrics_greedy"), npz)
            )
        elif payload.get("kind") == "no_regulator":
            for government, block in payload.get("by_government", {}).items():
                results.append(
                    RunResult(payload["community"], government, "no_regulator", int(payload["seed"]),
                              block["metrics"], block.get("metrics_greedy"), npz)
                )
    return results


def _concatenate(head: dict, tail: dict) -> dict:
    return {name: np.concatenate([head[name], tail[name]]) for name in tail}


def load_series(root: Path, community: str, government: str, condition: str, seed: int) -> dict:
    """The whole run, step 0 to the end of the window, as one set of per-step arrays.

    A regulated run is its warm-up followed by its branch; the run without a
    regulator is the warm-up up to the restore point followed by its own steps.
    """
    folder = warmup_dir(root, community, seed)
    warm = load_records(folder / "steps.npz")
    if condition == "no_regulator":
        tail = load_records(no_regulator_paths(root, community, seed)[0])
        start = int(tail["step"][0])
        head = {name: values[warm["step"] < start] for name, values in warm.items()}
    else:
        tail = load_records(branch_paths(root, community, government, condition, seed)[0])
        head = warm
    return _concatenate(head, tail)
