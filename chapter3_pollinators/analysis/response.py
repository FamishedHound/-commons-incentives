"""How far ORAA's incentive can move the landowners' actions.

At the branch point (the warm-up's ``branch.pt``) each landowner's mean action
is evaluated with ORAA's candidate observation for a grid of incentives
``x in [-b, b]`` - the habitat layer of the landowner's current observation and
``x`` on its cells - once with the landowners' own actors (what Omniscient ORAA
consults) and once with the model learner's actors (Model-based ORAA).  The
bound ``b`` is the widest bound of ORAA in the group (1, or 3 for the bound
comparison, whose figure shades the default range [-1, 1]).  The figure
``oraa_response_<group>`` shows the mean over seeds with one standard error,
with the government's targets marked.

``reachability_<group>`` gives, per landowner and bound, the gap between the
target and the closest mean action any admissible incentive produces,
``min over |x| <= b of |target_i - mean action_i(x)|`` (mean +- SE over seeds),
and the share of seeds where that gap is at most ``REACH_TOLERANCE``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

from chapter3_pollinators.experiment.checkpoint import load_checkpoint  # noqa: E402
from chapter3_pollinators.experiment.run import warmup_dir  # noqa: E402
from chapter3_pollinators.experiment.settings import Group, government_targets, oraa_bound  # noqa: E402
from chapter3_pollinators.regulators.oraa import get_agents_land_positions  # noqa: E402

GRID = np.linspace(-1.0, 1.0, 21)
#: A target counts as reachable when some admissible incentive brings the mean action this close.
REACH_TOLERANCE = 0.05
ACTOR_SETS = (("landowners", "landowners' actors"), ("model", "model's actors"))


def wide_grid(bound: float) -> np.ndarray:
    """Incentives from ``-bound`` to ``bound`` in steps of 0.1 (``-1``, ``0`` and ``1`` included)."""
    steps = int(round(20 * bound))
    return np.round(np.linspace(-bound, bound, steps + 1), 10)


def group_bound(group: Group) -> float:
    """The widest incentive bound of ORAA among the group's conditions (1 without ORAA)."""
    bounds = [oraa_bound(c) for c in group.conditions if oraa_bound(c) is not None]
    return max(bounds, default=1.0)


def response_to_incentive(obs, agents, actors, xs=GRID) -> np.ndarray:
    """``(n_agents, len(xs))`` mean action of each actor under ORAA's candidate observation."""
    out = np.zeros((len(agents), len(xs)))
    with th.no_grad():
        for i, agent in enumerate(agents):
            habitat = np.array(obs)[i][0][0]
            for k, x in enumerate(xs):
                layer = get_agents_land_positions(agent, habitat, float(x))
                state = th.cat(
                    [th.from_numpy(habitat.copy()).float().unsqueeze(0),
                     th.from_numpy(layer.copy()).float().unsqueeze(0)]
                ).unsqueeze(0)
                device = next(actors[i].parameters()).device
                out[i, k] = actors[i](state.to(device)).mean().item()
    return out


_CURVES: dict[tuple, dict[str, np.ndarray]] = {}


def branch_point_curves(root: Path, community: str, seeds, xs=GRID) -> dict[str, np.ndarray]:
    """Response curves ``(seeds, landowners, len(xs))`` for every seed with a ``branch.pt``."""
    key = (str(Path(root).resolve()), community, tuple(seeds), tuple(float(x) for x in xs))
    if key in _CURVES:
        return _CURVES[key]
    curves: dict[str, list] = {"landowners": [], "model": [], "seeds": []}
    for seed in seeds:
        path = warmup_dir(root, community, seed) / "branch.pt"
        if not path.is_file():
            continue
        run = load_checkpoint(path)
        agents = run.game.agent_processor.all_agents
        curves["landowners"].append(response_to_incentive(run.state.obs, agents, run.real.actors, xs))
        curves["model"].append(response_to_incentive(run.state.obs, agents, run.model.actors, xs))
        curves["seeds"].append(seed)
        del run
    result = {k: np.stack(v) for k, v in curves.items() if v}
    _CURVES[key] = result
    return result


def _mean_se(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(len(stack)) if len(stack) > 1 else np.zeros_like(mean)
    return mean, se


def response_figure(root: Path, group: Group, out_dir: Path, seeds) -> list[Path]:
    from chapter3_pollinators.analysis.figures import COLOURS, STYLE, _index, _save

    bound = group_bound(group)
    xs = GRID if bound <= 1.0 else wide_grid(bound)
    data = {c: branch_point_curves(root, c, seeds, xs) for c in group.communities}
    data = {c: d for c, d in data.items() if d}
    if not data:
        return []
    rows = []
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(len(data), 4, figsize=(7.0, 1.75 * len(data) + 0.6),
                                    sharex=True, squeeze=False)
        for r, (community, curves) in enumerate(data.items()):
            targets = government_targets(community, group.government)
            for i in range(4):
                axis = axes[r][i]
                if bound > 1.0:
                    axis.axvspan(-1.0, 1.0, color="black", alpha=0.06, linewidth=0, label="bound 1")
                for key, colour, style, label in (
                    ("landowners", COLOURS["oraa_omni"], "-", "landowners' actors"),
                    ("model", COLOURS["oraa_mb"], "-.", "model's actors"),
                ):
                    stack = curves[key][:, i, :]
                    mean, se = _mean_se(stack)
                    axis.plot(xs, mean, color=colour, linestyle=style, linewidth=1.0,
                              label=f"{label}, n = {len(stack)}")
                    axis.fill_between(xs, mean - se, mean + se, color=colour, alpha=0.25, linewidth=0)
                    rows += [[group.name, community, i, key, float(x), float(m), float(e), len(stack)]
                             for x, m, e in zip(xs, mean, se)]
                axis.axhline(targets[i], color="black", linestyle="--", linewidth=0.7, label="target")
                axis.set_ylim(-0.02, 1.02)
                axis.set_title(f"{community}, landowner {i}", fontsize=7)
                if i == 0:
                    axis.set_ylabel("Mean action")
                if r == len(data) - 1:
                    axis.set_xlabel("Incentive x")
                axis.legend(loc="best", fontsize=5, handlelength=2.4, borderaxespad=0.2,
                            frameon=True, framealpha=0.85, edgecolor="none")
        figure.suptitle(f"{group.name}: response to ORAA's incentive at the branch point "
                        "(mean $\\pm$ SE over seeds)", fontsize=9)
        figure.tight_layout()
    name = f"oraa_response_{group.file_stem}"
    paths = _save(figure, out_dir, name, rows,
                  ["group", "setting", "landowner", "actors", "x", "mean_action", "se", "n"])
    caption = (f"{group.name}: each landowner's mean action at the branch point against ORAA's incentive, "
               "with the landowners' and with the model's actors.")
    if bound > 1.0:
        caption += f" Incentives up to {bound:g} in magnitude; the shaded band is the default range [-1, 1]."
    _index(out_dir, name, caption, paths)
    return paths


def _smallest_gap(curve: np.ndarray, xs: np.ndarray, target: float, bound: float) -> float:
    """Smallest ``|target - mean action|`` over incentives with ``|x| <= bound``."""
    admissible = np.abs(xs) <= bound + 1e-9
    return float(np.min(np.abs(target - curve[admissible])))


def reachability_table(root: Path, group: Group, out_dir: Path, seeds) -> list[Path]:
    """Per landowner and bound: how close any admissible incentive brings the mean action to the target."""
    bound = group_bound(group)
    bounds = sorted({1.0, bound})
    xs = GRID if bound <= 1.0 else wide_grid(bound)
    rows = []
    for community in group.communities:
        curves = branch_point_curves(root, community, seeds, xs)
        if not curves:
            continue
        targets = government_targets(community, group.government)
        for i in range(4):
            for key, _label in ACTOR_SETS:
                for b in bounds:
                    gaps = np.array([_smallest_gap(curve[i], xs, targets[i], b) for curve in curves[key]])
                    se = float(gaps.std(ddof=1) / math.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")
                    rows.append({
                        "group": group.name, "setting": community, "landowner": i, "target": targets[i],
                        "actors": key, "bound": b, "gap_mean": float(gaps.mean()), "gap_se": se,
                        "reachable_share": float(np.mean(gaps <= REACH_TOLERANCE)), "n": len(gaps),
                    })
    if not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"reachability_{group.file_stem}"
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(_reachability_tex(rows, bounds, group), encoding="utf-8")
    return [csv_path, tex_path]


def _reachability_tex(rows: list[dict], bounds: list[float], group: Group) -> str:
    def cell(row: dict) -> str:
        se = "" if math.isnan(row["gap_se"]) else f" $\\pm$ {row['gap_se']:.3f}"
        share = f" ({100 * row['reachable_share']:.0f}\\%)"
        return f"{row['gap_mean']:.3f}{se}{share}"

    index = {(r["setting"], r["landowner"], r["actors"], r["bound"]): r for r in rows}
    columns = [(key, b) for key, _ in ACTOR_SETS for b in bounds]
    header = ["Setting", "Landowner", "Target"] + [
        f"{'Landowners' if key == 'landowners' else 'Model'}, $|x|\\le{b:g}$" for key, b in columns
    ]
    lines = [
        f"% {group.name}: smallest |target - mean action| any admissible incentive gives at the branch point,"
        f" mean $\\pm$ SE over seeds (in brackets: share of seeds within {REACH_TOLERANCE:g})",
        "\\begin{tabular}{llr" + "r" * len(columns) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    previous = None
    for setting in group.communities:
        for i in range(4):
            if (setting, i, "landowners", bounds[0]) not in index:
                continue
            if previous is not None and setting != previous and i == 0:
                lines.append("\\addlinespace")
            first = index[(setting, i, "landowners", bounds[0])]
            row = [setting if i == 0 else "", str(i), f"{first['target']:.2f}"]
            row += [cell(index[(setting, i, key, b)]) for key, b in columns]
            lines.append(" & ".join(row) + " \\\\")
            previous = setting
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"
