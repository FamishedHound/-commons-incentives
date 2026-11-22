"""Metrics of one run over the evaluation window (counted steps 1,000-2,199).

All metrics are means over the steps of the window (every step, greedy and
exploratory, unless ``greedy_only``):

* ``reward_per_agent``: the landowners' reward without incentive and before the
  clip, averaged over landowners (the intrinsic reward; transfers excluded);
* ``received_reward_per_agent``: the reward the landowners learnt from
  (incentive included and clipped at 0; under LToS the reward after sharing);
* ``pollination_difference``: ``mean_i |target_i - mean declared habitat_i|``
  as a fraction (``_pp``: in percentage points);
* ``budget_net`` / ``budget_gross`` / ``budget_penalties``: per-step sum over
  landowners of the incentive, of its positive part and of its negative part;
  ``cumulative_net_budget`` is the window's total net spend.  Under LToS the
  incentive of a landowner is the reward it receives from others net of what it
  gives (shaped minus own reward), so the net budget is zero and
  ``transfer_volume`` reports the reward moved between landowners per step.
"""

from __future__ import annotations

import numpy as np

from chapter3_pollinators.experiment.schedule import BRANCH_COUNTER, LAST_COUNTER


def window_mask(records: dict, first: int = BRANCH_COUNTER, last: int = LAST_COUNTER) -> np.ndarray:
    counter = records["counter"]
    return (counter >= first) & (counter <= last)


def incentive_series(records: dict, condition: str) -> np.ndarray:
    """Per-step, per-landowner incentive (LToS: net reward received from others).

    Under LToS the incentive of a step with sharing is the shaped minus the own
    reward; steps without sharing (the warm-up part of a whole-run series) keep
    the incentive that was paid.
    """
    if condition == "ltos":
        shaped = records["shaped_reward"]
        return np.where(np.isnan(shaped), records["incentive"], shaped - records["reward"])
    return records["incentive"]


def pollination_gap(mean_habitat: np.ndarray, targets) -> np.ndarray:
    """``|target_i - mean habitat_i|`` per step and landowner."""
    return np.abs(np.asarray(targets, dtype=np.float64)[None, :] - mean_habitat)


def window_metrics(records: dict, targets, condition: str, greedy_only: bool = False) -> dict:
    mask = window_mask(records)
    if greedy_only:
        mask = mask & records["greedy"].astype(bool)
    steps = int(mask.sum())
    if steps == 0:
        raise ValueError("no steps of the evaluation window in these records")
    incentive = incentive_series(records, condition)[mask]
    per_step = incentive.sum(axis=1)
    declared_gap = pollination_gap(records["mean_declared"][mask], targets)
    action_gap = pollination_gap(records["mean_action"][mask], targets)
    received = records["shaped_reward"] if condition == "ltos" else records["reward"]
    metrics = {
        "steps": steps,
        "reward_per_agent": float(records["reward_plain"][mask].mean()),
        "received_reward_per_agent": float(received[mask].mean()),
        "pollination_difference": float(declared_gap.mean()),
        "pollination_difference_pp": float(100.0 * declared_gap.mean()),
        "pollination_difference_action": float(action_gap.mean()),
        "signed_target_gap": float(
            (np.asarray(targets)[None, :] - records["mean_declared"][mask]).mean()
        ),
        "mean_declared_habitat": float(records["mean_declared"][mask].mean()),
        "global_pollination_reward": float(records["global_reward"][mask].mean()),
        "budget_net": float(per_step.mean()),
        "budget_gross": float(np.clip(incentive, 0, None).sum(axis=1).mean()),
        "budget_penalties": float(np.clip(incentive, None, 0).sum(axis=1).mean()),
        "cumulative_net_budget": float(per_step.sum()),
        "greedy_fraction": float(records["greedy"][mask].mean()),
    }
    if condition == "ltos":
        metrics["transfer_volume"] = float(records["transfer_volume"][mask].mean())
        weights = records["sharing_weights"][mask]
        metrics["selfishness"] = float(np.einsum("tii->ti", weights).mean())
    if not np.all(np.isnan(records["model_distance_greedy"][mask])):
        metrics["model_distance_greedy"] = float(np.nanmean(records["model_distance_greedy"][mask]))
        metrics["model_distance_taken"] = float(np.nanmean(records["model_distance_taken"][mask]))
        metrics["model_mean_action_error"] = float(
            np.nanmean(np.abs(records["real_greedy_mean"][mask] - records["model_mean"][mask]))
        )
    if not np.all(np.isnan(records["oraa_best"][mask])):
        metrics["oraa_best_objective"] = float(np.nanmean(records["oraa_best"][mask]))
    return metrics
