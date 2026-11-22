"""Build the thesis tables and figures from a campaign folder.

Usage (from the folder that contains the package)::

    python -m chapter3_pollinators.analysis --root chapter3_pollinators/results/campaign_main \\
        --out chapter3_pollinators/results/thesis --response-curves

For every group of the analysis with finished runs (HT+HTC, HT+HMC, HM+HMC,
HM+HTC, and "HT+HTC bounds", the sensitivity analysis on ORAA's incentive
bound) it writes, under ``--out``:

* ``summary_<group>.json``, ``cells_<group>.csv``, ``per_seed_<group>.csv``,
  ``paired_tests_<group>.csv`` - mean +- SE over seeds per setting and
  condition, every seed's value, and every paired t-test (all steps of the
  evaluation window); the same files with suffix ``_greedy`` use greedy steps only;
* ``table_<group>.tex``, ``paired_<group>.tex`` and ``model_error_<group>.tex``
  - the LaTeX tables (results, paired tests, model error of Model-based ORAA);
  for the bound comparison also ``paired_budget_<group>.tex`` (paired tests of
  the gross payments and the penalties);
* the figures listed in :mod:`chapter3_pollinators.analysis.figures`, and with
  ``--response-curves`` ``oraa_response_<group>`` and, for the bound
  comparison, ``reachability_<group>`` (:mod:`chapter3_pollinators.analysis.response`).

``<group>`` in file names is the group's name with ``+`` and spaces replaced
by ``_`` (``HT_HTC``, ``HT_HTC_bounds``).  The analysis can be run at any time:
groups and cells are reported with the seeds that have finished, and every
table states ``n``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chapter3_pollinators.analysis.aggregate import group_summary, write_group_outputs
from chapter3_pollinators.analysis.figures import write_group_figures
from chapter3_pollinators.analysis.load import load_results
from chapter3_pollinators.analysis.tables import (
    BUDGET_PAIRED_METRICS,
    model_table_tex,
    paired_table_tex,
    results_table_tex,
)
from chapter3_pollinators.experiment.settings import ANALYSIS_GROUPS, BOUND_COMPARISON


def build(
    root: Path,
    out: Path,
    groups: list[str] | None = None,
    figures: bool = True,
    response_curves: bool = False,
) -> list[Path]:
    results = load_results(root)
    written: list[Path] = []
    for group in ANALYSIS_GROUPS:
        if groups and group.name not in groups:
            continue
        summary = group_summary(results, group)
        if not summary["cells"]:
            continue
        stem = group.file_stem
        written += write_group_outputs(summary, out)
        written += write_group_outputs(group_summary(results, group, greedy=True), out, suffix="_greedy")
        tables = [(f"table_{stem}.tex", results_table_tex(summary)),
                  (f"paired_{stem}.tex", paired_table_tex(summary)),
                  (f"model_error_{stem}.tex", model_table_tex(summary))]
        if group.name == BOUND_COMPARISON.name:
            tables.append((f"paired_budget_{stem}.tex", paired_table_tex(summary, BUDGET_PAIRED_METRICS)))
        for name, text in tables:
            if text is None:
                continue
            path = out / name
            path.write_text(text, encoding="utf-8")
            written.append(path)
        if figures:
            written += write_group_figures(root, results, group, out)
        if response_curves and any(c.startswith("oraa_mb") for c in group.conditions):
            from chapter3_pollinators.analysis.response import reachability_table, response_figure

            seeds = sorted({r.seed for r in results if r.community in group.communities})
            written += response_figure(root, group, out, seeds)
            if group.name == BOUND_COMPARISON.name:
                written += reachability_table(root, group, out, seeds)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=None,
                        help='group names, e.g. "HT+HTC" "HT+HTC bounds" (default: all)')
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--response-curves", action="store_true",
        help="also draw the actors' response to ORAA's incentive (loads every branch checkpoint)",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in build(args.root, args.out, args.groups, figures=not args.no_figures,
                      response_curves=args.response_curves):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
