"""Figures for the thesis (matplotlib), each with a CSV of the numbers it draws.

* ``bars_<group>``: per setting and condition, the average reward per agent,
  the pollination difference (percentage points), the net budget and the gross
  payments per step; bars are means over seeds, error bars one standard error.
* ``budget_cumulative_<group>`` (groups with both ORAA forms): the cumulative
  net budget over the evaluation window, Omniscient against Model-based, mean
  +- SE over seeds after averaging the group's settings (and one panel per
  setting in ``..._by_setting``).
* ``ltos_comparison_<group>`` (HT+HTC): LToS against Model-based ORAA and the
  run without a regulator; the third panel sets the government's gross
  payments beside the reward LToS moves between landowners.
* ``timeseries_<group>``: the whole run (moving average over 100 steps) of the
  regulated conditions and of the run without a regulator, with the start of
  the curriculum, the start of the step counter and the branch point marked.

Every figure is written as ``.pdf``, ``.png`` and ``.csv`` under the same name.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

from chapter3_pollinators.analysis.aggregate import CONDITION_ORDER, group_summary  # noqa: E402
from chapter3_pollinators.analysis.load import RunResult, load_series  # noqa: E402
from chapter3_pollinators.experiment.metrics import incentive_series, window_mask  # noqa: E402
from chapter3_pollinators.experiment.schedule import BRANCH_COUNTER  # noqa: E402
from chapter3_pollinators.experiment.settings import (  # noqa: E402
    BOUND_COMPARISON,
    CONDITION_LABELS,
    Group,
    government_targets,
)

#: Okabe-Ito colours, one per condition.
COLOURS = {
    "no_regulator": "#7f7f7f",
    "reward_shape": "#E69F00",
    "oraa_mb": "#0072B2",
    "oraa_omni": "#56B4E9",
    "ltos": "#D55E00",
    "oraa_mb_b3": "#0072B2",
    "oraa_omni_b3": "#56B4E9",
}
#: LToS hatched; ORAA with the wider bound cross-hatched in its form's colour.
HATCHES = {"ltos": "//", "oraa_mb_b3": "xx", "oraa_omni_b3": "xx"}
PNG_DPI = 200
#: Whole percentage points on pollination-difference axes (0, 10, 20, ... or 0, 2, 4, ...).
PP_TICKS = lambda: MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True)  # noqa: E731
MOVING_AVERAGE = 100

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
}


def _save(figure, out_dir: Path, name: str, rows: list[list], header: list[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{name}{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=PNG_DPI if suffix == ".png" else None)
        paths.append(path)
    plt.close(figure)
    path = out_dir / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    paths.append(path)
    return paths


def _index(out_dir: Path, name: str, caption: str, paths: list[Path]) -> None:
    path = out_dir / "figures_index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    index[name] = {"caption": caption, "files": [p.name for p in paths]}
    path.write_text(json.dumps(index, indent=1), encoding="utf-8")


def _safe(group: Group) -> str:
    return group.file_stem


def _legend_above(axis, ncol: int, fontsize=None) -> None:
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, borderaxespad=0.3,
                handlelength=1.4, columnspacing=0.8, fontsize=fontsize)


def _settings_present(results: list[RunResult], group: Group) -> list[str]:
    present = {r.community for r in results if r.government == group.government}
    return [c for c in group.communities if c in present]


# ----------------------------------------------------------------------
# bar charts
# ----------------------------------------------------------------------
BAR_PANELS = (
    ("reward_per_agent", "Average reward per agent"),
    ("pollination_difference_pp", "Pollination difference (pp)"),
    ("budget_net", "Net budget per step"),
    ("budget_gross", "Gross payments per step"),
)
#: The bound comparison adds the penalties and ORAA's own objective.
BOUND_PANELS = BAR_PANELS + (
    ("budget_penalties", "Penalties per step"),
    ("oraa_best_objective", "ORAA's best |target - mean action|"),
)
#: Short legend labels for the bound comparison.
BOUND_LABELS = {
    "no_regulator": "No regulator",
    "reward_shape": "Reward-shape",
    "oraa_mb": "ORAA-MB, bound 1",
    "oraa_omni": "ORAA-Omni, bound 1",
    "oraa_mb_b3": "ORAA-MB, bound 3",
    "oraa_omni_b3": "ORAA-Omni, bound 3",
}


def bar_figure(summary: dict, out_dir: Path, name: str, panels=BAR_PANELS, labels=None,
               legend_fontsize=None) -> list[Path]:
    conditions = summary["conditions"]
    settings = [c for c in summary["communities"]
                if any(cell["community"] == c for cell in summary["cells"])]
    cells = {(c["community"], c["condition"]): c for c in summary["cells"]}
    rows = []
    n_rows = (len(panels) + 1) // 2
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(n_rows, 2, figsize=(7.0, 2.4 * n_rows))
        for axis, (metric, label) in zip(axes.flat, panels):
            # the conditions that have this metric share the width of a setting
            present = [c for c in conditions if any(metric in cells.get((s, c), {}) for s in settings)]
            width = 0.8 / max(len(present), 1)
            for k, condition in enumerate(present):
                means, errors, xs = [], [], []
                for s, setting in enumerate(settings):
                    cell = cells.get((setting, condition))
                    if cell is None or metric not in cell:
                        continue
                    xs.append(s + (k - (len(present) - 1) / 2) * width)
                    means.append(cell[metric]["mean"])
                    errors.append(0.0 if np.isnan(cell[metric]["se"]) else cell[metric]["se"])
                    rows.append([summary["group"], setting, condition, metric, cell[metric]["mean"],
                                 cell[metric]["se"], cell["n"]])
                if not xs:
                    continue
                legend_label = (labels or CONDITION_LABELS)[condition]
                if condition == "ltos" and metric.startswith("budget"):
                    legend_label += " (transfers)"
                axis.bar(xs, means, width=width, yerr=errors, capsize=1.5, color=COLOURS[condition],
                         hatch=HATCHES.get(condition), edgecolor="white" if condition != "ltos" else "black",
                         linewidth=0.3, error_kw={"elinewidth": 0.6}, label=legend_label)
            axis.set_xticks(range(len(settings)))
            axis.set_xticklabels(settings)
            axis.set_ylabel(label)
            if metric == "pollination_difference_pp":
                axis.yaxis.set_major_locator(PP_TICKS())
            axis.axhline(0.0, color="black", linewidth=0.5)
            if axis.get_legend_handles_labels()[0]:
                _legend_above(axis, ncol=3, fontsize=legend_fontsize)
            else:  # no condition has this metric
                axis.set_visible(False)
        figure.suptitle(f"{summary['group']}: mean $\\pm$ SE over seeds", fontsize=9)
        figure.tight_layout()
    return _save(figure, out_dir, name, rows,
                 ["group", "setting", "condition", "metric", "mean", "se", "n"])


# ----------------------------------------------------------------------
# cumulative budget
# ----------------------------------------------------------------------
def _cumulative_budget(result: RunResult) -> np.ndarray:
    with np.load(result.records_path) as data:
        records = {k: data[k] for k in ("counter", "incentive", "shaped_reward", "reward")}
    mask = window_mask(records)
    incentive = incentive_series(records, result.condition)[mask]
    return np.cumsum(incentive.sum(axis=1))


def _band(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(c) for c in curves)
    stack = np.stack([c[:length] for c in curves])
    mean = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros(length)
    return mean, se


def cumulative_budget_figures(results: list[RunResult], group: Group, out_dir: Path) -> list[Path]:
    chosen = [r for r in results if r.government == group.government and r.community in group.communities]
    settings = _settings_present(chosen, group)
    conditions = [c for c in ("oraa_omni", "oraa_mb") if c in group.conditions]
    curves = {(r.community, r.condition, r.seed): _cumulative_budget(r) for r in chosen if r.condition in conditions}
    if not curves:
        return []
    paths: list[Path] = []
    rows = []
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(3.4, 2.6))
        for condition in conditions:
            seeds = sorted({s for (c, cond, s) in curves if cond == condition})
            pooled = []
            for seed in seeds:
                per_setting = [curves[(c, condition, seed)] for c in settings if (c, condition, seed) in curves]
                if per_setting and len(per_setting) == len(settings):
                    pooled.append(_band(per_setting)[0] * 1.0)
            if not pooled:
                continue
            mean, se = _band(pooled)
            steps = np.arange(1, len(mean) + 1)
            axis.plot(steps, mean, color=COLOURS[condition], linewidth=1.0,
                      label=f"{CONDITION_LABELS[condition]} (n = {len(pooled)})")
            axis.fill_between(steps, mean - se, mean + se, color=COLOURS[condition], alpha=0.25, linewidth=0)
            rows += [[group.name, "pooled", condition, int(k), float(m), float(e), len(pooled)]
                     for k, m, e in zip(steps, mean, se)]
        axis.set_xlabel("Step of the evaluation window")
        axis.set_ylabel("Cumulative net budget")
        axis.legend(loc="upper left")
        axis.set_title(
            f"{group.name} ({', '.join(settings)}): mean over settings\nband: $\\pm$ SE over seeds"
        )
    name = f"budget_cumulative_{_safe(group)}"
    paths += _save(figure, out_dir, name, rows, ["group", "setting", "condition", "window_step", "mean", "se", "n"])
    _index(out_dir, name, f"Cumulative net budget of Omniscient and Model-based ORAA in {group.name}.", paths[-3:])

    rows = []
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
        for axis in axes.flat[len(settings):]:
            axis.set_visible(False)
        for axis, community in zip(axes.flat, settings):
            for condition in conditions:
                runs = [curves[k] for k in sorted(curves) if k[0] == community and k[1] == condition]
                if not runs:
                    continue
                mean, se = _band(runs)
                steps = np.arange(1, len(mean) + 1)
                axis.plot(steps, mean, color=COLOURS[condition], linewidth=1.0,
                          label=f"{CONDITION_LABELS[condition]} (n = {len(runs)})")
                axis.fill_between(steps, mean - se, mean + se, color=COLOURS[condition], alpha=0.25, linewidth=0)
                rows += [[group.name, community, condition, int(k), float(m), float(e), len(runs)]
                         for k, m, e in zip(steps, mean, se)]
            axis.set_title(community, pad=14)
            axis.set_ylabel("Cumulative net budget")
            axis.legend(loc="upper left")
        for axis in axes[-1]:
            axis.set_xlabel("Step of the evaluation window")
        figure.tight_layout()
    name = f"budget_cumulative_{_safe(group)}_by_setting"
    new = _save(figure, out_dir, name, rows, ["group", "setting", "condition", "window_step", "mean", "se", "n"])
    _index(out_dir, name, f"Cumulative net budget per setting in {group.name}.", new)
    return paths + new


# ----------------------------------------------------------------------
# LToS comparison
# ----------------------------------------------------------------------
def ltos_figure(summary: dict, out_dir: Path, name: str) -> list[Path]:
    conditions = [c for c in ("no_regulator", "oraa_mb", "ltos") if c in summary["conditions"]]
    cells = {(c["community"], c["condition"]): c for c in summary["cells"]}
    settings = [c for c in summary["communities"] if (c, "ltos") in cells]
    if not settings:
        return []
    panels = (
        ("reward_per_agent", "Average reward per agent", conditions),
        ("pollination_difference_pp", "Pollination difference (pp)", conditions),
        ("money", "Reward moved per step", [c for c in ("oraa_mb", "ltos") if c in conditions]),
    )
    rows = []
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
        for axis, (metric, label, shown) in zip(axes, panels):
            width = 0.8 / len(shown)
            for k, condition in enumerate(shown):
                key = metric
                legend = CONDITION_LABELS[condition]
                if metric == "money":
                    key = "transfer_volume" if condition == "ltos" else "budget_gross"
                    legend = ("LToS: transfers between landowners" if condition == "ltos"
                              else f"{CONDITION_LABELS[condition]}: government gross payments")
                xs, means, errors = [], [], []
                for s, setting in enumerate(settings):
                    cell = cells.get((setting, condition))
                    if cell is None or key not in cell:
                        continue
                    xs.append(s + (k - (len(shown) - 1) / 2) * width)
                    means.append(cell[key]["mean"])
                    errors.append(0.0 if np.isnan(cell[key]["se"]) else cell[key]["se"])
                    rows.append([summary["group"], setting, condition, key, cell[key]["mean"],
                                 cell[key]["se"], cell["n"]])
                axis.bar(xs, means, width=width, yerr=errors, capsize=1.5, color=COLOURS[condition],
                         hatch=HATCHES.get(condition), edgecolor="black" if condition == "ltos" else "white",
                         linewidth=0.3, error_kw={"elinewidth": 0.6}, label=legend)
            axis.set_xticks(range(len(settings)))
            axis.set_xticklabels(settings)
            axis.set_ylabel(label)
            axis.axhline(0.0, color="black", linewidth=0.5)
            _legend_above(axis, ncol=1)
        figure.suptitle(f"{summary['group']}: LToS comparison, mean $\\pm$ SE over seeds", fontsize=9)
        figure.tight_layout()
    return _save(figure, out_dir, name, rows, ["group", "setting", "condition", "metric", "mean", "se", "n"])


# ----------------------------------------------------------------------
# time series
# ----------------------------------------------------------------------
def _moving_average(x: np.ndarray, window: int = MOVING_AVERAGE) -> np.ndarray:
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def _series_metrics(series: dict, targets, condition: str) -> dict[str, np.ndarray]:
    gap = np.abs(np.asarray(targets)[None, :] - series["mean_declared"]).mean(axis=1) * 100.0
    incentive = incentive_series(series, condition).sum(axis=1)
    out = {
        "reward_per_agent": series["reward_plain"].mean(axis=1),
        "pollination_difference_pp": gap,
        "net_incentive": incentive,
    }
    if condition == "ltos":
        transfer = np.where(np.isnan(series["transfer_volume"]), 0.0, series["transfer_volume"])
        out["transfer_volume"] = transfer
    return out


def timeseries_figure(root: Path, results: list[RunResult], group: Group, out_dir: Path) -> list[Path]:
    chosen = [r for r in results if r.government == group.government and r.community in group.communities]
    if not chosen:
        return []
    settings = _settings_present(chosen, group)
    seeds_by_condition: dict[str, set[int]] = {}
    for r in chosen:
        seeds_by_condition.setdefault(r.condition, set()).add(r.seed)
    curves: dict[tuple[str, str], list[np.ndarray]] = {}
    markers: dict[str, float] = {}
    rows = []
    for condition in [c for c in CONDITION_ORDER if c in seeds_by_condition]:
        for seed in sorted(seeds_by_condition[condition]):
            per_setting = []
            for community in settings:
                try:
                    series = load_series(root, community, group.government, condition, seed)
                except FileNotFoundError:
                    per_setting = []
                    break
                per_setting.append(_series_metrics(series, government_targets(community, group.government), condition))
                if condition != "no_regulator" and "branch" not in markers:
                    counter = series["counter"]
                    markers["branch"] = float(series["step"][np.argmax(counter >= BRANCH_COUNTER)])
                    markers["counting"] = float(series["step"][np.argmax(counter > 0)])
                    markers["curriculum"] = float(series["step"][np.argmax(series["phase"] > 0)])
            if not per_setting:
                continue
            for metric in per_setting[0]:
                pooled = np.mean([m[metric] for m in per_setting], axis=0)
                curves.setdefault((condition, metric), []).append(_moving_average(pooled))
    if not curves:
        return []
    panels = (
        ("reward_per_agent", "Average reward per agent"),
        ("pollination_difference_pp", "Pollination difference (pp)"),
        ("net_incentive", "Net incentive per step"),
    )
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(3, 1, figsize=(7.0, 6.6), sharex=True)
        for axis, (metric, label) in zip(axes, panels):
            for condition in [c for c in CONDITION_ORDER if (c, metric) in curves]:
                mean, se = _band(curves[(condition, metric)])
                steps = np.arange(len(mean)) + MOVING_AVERAGE - 1
                n = len(curves[(condition, metric)])
                axis.plot(steps, mean, color=COLOURS[condition], linewidth=0.8,
                          label=f"{CONDITION_LABELS[condition]} (n = {n})")
                axis.fill_between(steps, mean - se, mean + se, color=COLOURS[condition], alpha=0.2, linewidth=0)
                rows += [[group.name, condition, metric, int(s), float(m), float(e), n]
                         for s, m, e in zip(steps[::10], mean[::10], se[::10])]
            if metric == "net_incentive" and ("ltos", "transfer_volume") in curves:
                mean, se = _band(curves[("ltos", "transfer_volume")])
                steps = np.arange(len(mean)) + MOVING_AVERAGE - 1
                axis.plot(steps, mean, color=COLOURS["ltos"], linewidth=0.8, linestyle="--",
                          label=f"LToS transfers (n = {len(curves[('ltos', 'transfer_volume')])})")
                rows += [[group.name, "ltos", "transfer_volume", int(s), float(m), float(e), len(curves[("ltos", "transfer_volume")])]
                         for s, m, e in zip(steps[::10], mean[::10], se[::10])]
            for key, style, text in (
                ("curriculum", ":", "curriculum starts"),
                ("counting", "-.", "step counter starts"),
                ("branch", "--", "branch point"),
            ):
                if key in markers:
                    axis.axvline(markers[key], color="black", linestyle=style, linewidth=0.7, label=text)
            axis.set_ylabel(label)
            if metric == "pollination_difference_pp":
                axis.yaxis.set_major_locator(PP_TICKS())
            _legend_above(axis, ncol=4)
        axes[-1].set_xlabel(f"Environment step (moving average over {MOVING_AVERAGE} steps; "
                            "branch point = counted step 1,000)")
        figure.suptitle(f"{group.name} ({', '.join(settings)}): mean over settings; band: $\\pm$ SE over seeds", fontsize=9)
        figure.tight_layout()
    name = f"timeseries_{_safe(group)}"
    paths = _save(figure, out_dir, name, rows, ["group", "condition", "metric", "step", "mean", "se", "n"])
    _index(out_dir, name, f"{group.name}: regulated runs and the run without a regulator over the whole run "
                          "(moving average over 100 steps; vertical lines: curriculum start, step counter start, "
                          "branch point at counted step 1,000). LToS: its net incentive is zero by construction; "
                          "the dashed line is the reward moved between landowners.", paths)
    return paths


def write_group_figures(root: Path, results: list[RunResult], group: Group, out_dir: Path) -> list[Path]:
    summary = group_summary(results, group)
    if group.name == BOUND_COMPARISON.name:
        name = f"bars_{_safe(group)}"
        paths = bar_figure(summary, out_dir, name, panels=BOUND_PANELS, labels=BOUND_LABELS, legend_fontsize=6)
        _index(out_dir, name, f"{group.name}: no regulator, reward-shape and both forms of ORAA with the "
                              "incentive bound 1 and 3 (cross-hatched), per setting; ORAA's best objective is "
                              "|target - mean action| the search reached on the actors it consulted.", paths)
        return paths
    paths = []
    name = f"bars_{_safe(group)}"
    new = bar_figure(summary, out_dir, name)
    caption = f"{group.name}: reward, pollination difference and budget per setting."
    if "ltos" in group.conditions:
        caption += (" LToS (hatched) has no external budget: its budget bars are the reward a landowner"
                    " receives from the others net of what it gives (zero by construction) and its positive part.")
    _index(out_dir, name, caption, new)
    paths += new
    if {"oraa_mb", "oraa_omni"} <= set(group.conditions):
        paths += cumulative_budget_figures(results, group, out_dir)
    if "ltos" in group.conditions:
        name = f"ltos_comparison_{_safe(group)}"
        new = ltos_figure(summary, out_dir, name)
        if new:
            _index(out_dir, name, f"{group.name}: LToS against Model-based ORAA.", new)
        paths += new
    paths += timeseries_figure(root, results, group, out_dir)
    return paths
