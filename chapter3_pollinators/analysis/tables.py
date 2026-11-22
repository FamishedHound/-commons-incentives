"""LaTeX tables (booktabs, three decimals) for the thesis."""

from __future__ import annotations

import math

from chapter3_pollinators.analysis.aggregate import MAIN_COMPARISONS
from chapter3_pollinators.experiment.settings import CONDITION_LABELS

RESULT_COLUMNS = (
    ("reward_per_agent", "Reward / agent"),
    ("pollination_difference_pp", "Poll. diff. (pp)"),
    ("budget_net", "Net budget"),
    ("budget_gross", "Gross pay."),
    ("budget_penalties", "Penalties"),
    ("cumulative_net_budget", "Cum. net budget"),
)
PAIRED_METRICS = (
    ("reward_per_agent", "$\\Delta$ reward / agent"),
    ("pollination_difference_pp", "$\\Delta$ poll. diff. (pp)"),
    ("budget_net", "$\\Delta$ net budget"),
)
#: The budget components, for a second paired table where they matter.
BUDGET_PAIRED_METRICS = (
    ("budget_gross", "$\\Delta$ Gross pay."),
    ("budget_penalties", "$\\Delta$ Penalties"),
)


def _number(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.3f}"


def _p_value(p: float) -> str:
    if p is None or math.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _mean_se(entry: dict | None) -> str:
    if not entry:
        return "--"
    if math.isnan(entry["se"]):
        return _number(entry["mean"])
    return f"{_number(entry['mean'])} $\\pm$ {_number(entry['se'])}"


def results_table_tex(summary: dict) -> str:
    """Mean +- SE over seeds per setting and condition."""
    has_ltos = "ltos" in summary["conditions"]
    columns = list(RESULT_COLUMNS) + ([("transfer_volume", "Transfers")] if has_ltos else [])
    header = " & ".join(["Setting", "Condition", "$n$"] + [label for _, label in columns])
    lines = [
        f"% {summary['group']}: mean $\\pm$ SE over seeds, evaluation window (counted steps 1,000--2,199)",
        "\\begin{tabular}{ll" + "r" * (len(columns) + 1) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    previous = None
    for cell in summary["cells"]:
        setting = cell["community"] if cell["community"] != previous else ""
        if previous is not None and setting:
            lines.append("\\addlinespace")
        previous = cell["community"]
        row = [setting, CONDITION_LABELS[cell["condition"]], str(cell["n"])]
        row += [_mean_se(cell.get(metric)) for metric, _ in columns]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if has_ltos:
        lines.append(
            "% LToS has no external budget: its budget columns are the reward a landowner receives "
            "from the others net of what it gives (net is zero by construction), and Transfers is the "
            "total reward moved between landowners per step."
        )
    return "\n".join(lines) + "\n"


def paired_table_tex(summary: dict, metrics=PAIRED_METRICS) -> str:
    """Paired differences ``a - b`` with their SE, t and p for the main comparisons."""
    comparisons = MAIN_COMPARISONS[summary["group"]]
    tests = {(t["community"], t["metric"], t["a"], t["b"]): t for t in summary["paired_tests"]}
    tests.update({(t["community"], t["metric"], t["a"], t["b"]): t for t in summary["pooled_tests"]})
    header = ["Setting", "Comparison"]
    for _, label in metrics:
        header += [label, "$t$", "$p$"]
    lines = [
        f"% {summary['group']}: paired t-tests over seeds (a $-$ b); p-values uncorrected",
        "\\begin{tabular}{ll" + "rrr" * len(metrics) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    pooled_label = "pooled"
    if summary.get("pooled_settings") and summary["pooled_settings"] != summary["communities"]:
        pooled_label = "pooled (" + ", ".join(summary["pooled_settings"]) + ")"
    for community in list(summary["communities"]) + ["pooled"]:
        first = True
        for a, b in comparisons:
            shown = pooled_label if community == "pooled" else community
            row = [shown if first else "",
                   f"{CONDITION_LABELS[a]} $-$ {CONDITION_LABELS[b]}"]
            found = False
            for metric, _ in metrics:
                test = tests.get((community, metric, a, b))
                if test is None:
                    row += ["--", "--", "--"]
                    continue
                found = True
                row += [_mean_se({"mean": test["mean_diff"], "se": test["se_diff"]}),
                        _number(test["t"]), _p_value(test["p"])]
            if found:
                if first and community != summary["communities"][0]:
                    lines.append("\\addlinespace")
                lines.append(" & ".join(row) + " \\\\")
                first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


MODEL_COLUMNS = (
    ("model_distance_greedy", "Action distance"),
    ("model_mean_action_error", "Mean-action gap"),
    ("oraa_best_objective", "ORAA objective"),
)


def model_table_tex(summary: dict) -> str | None:
    """Model error of Model-based ORAA, and ORAA's best objective under both forms.

    ``Action distance`` is ``sum over landowners and cells of |round(a, 1) -
    round(model, 1)|`` between the landowners' greedy actions and the model's
    actions on the same observation (16 cells); ``Mean-action gap`` is
    ``|mean a_i - mean model_i|`` averaged over landowners; ``ORAA objective`` is
    the best ``|target_i - mean(actor_i)|`` the search reached on the actors it
    consulted.
    """
    if not any(c.startswith("oraa_mb") for c in summary["conditions"]):
        return None
    rows = [c for c in summary["cells"] if c["condition"].startswith("oraa")]
    header = " & ".join(["Setting", "Condition", "$n$"] + [label for _, label in MODEL_COLUMNS])
    lines = [
        f"% {summary['group']}: model error of Model-based ORAA (mean $\\pm$ SE over seeds, all steps)",
        "\\begin{tabular}{llr" + "r" * len(MODEL_COLUMNS) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for cell in rows:
        values = [_mean_se(cell.get(metric)) for metric, _ in MODEL_COLUMNS]
        lines.append(" & ".join([cell["community"], CONDITION_LABELS[cell["condition"]], str(cell["n"])]
                                + values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"
