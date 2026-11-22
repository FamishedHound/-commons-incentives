"""Summary statistics over seeds and the paired t-test between conditions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
from scipy import stats

#: Spread of the differences below this fraction of the values' scale (at least 1)
#: is rounding, e.g. LToS's net budget, zero by construction.
ROUNDING_TOLERANCE = 1e-12


def mean_se(values: Iterable[float]) -> tuple[float, float, int]:
    """Mean, standard error (sample SD / sqrt(n)) and n; SE is NaN for n < 2."""
    array = np.asarray(list(values), dtype=np.float64)
    n = int(array.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return mean, se, n


def paired_test(a: Mapping[int, float], b: Mapping[int, float]) -> dict:
    """Paired t-test of ``a - b`` over the seeds both conditions have.

    Returns n, the mean difference and its standard error, t, the degrees of
    freedom and the two-sided p-value (no correction for multiple comparisons).
    ``t`` and ``p`` are NaN when n < 2 or the differences have no spread beyond
    rounding (:data:`ROUNDING_TOLERANCE`).
    """
    seeds = sorted(set(a) & set(b))
    diffs = np.asarray([a[s] - b[s] for s in seeds], dtype=np.float64)
    n = int(diffs.size)
    mean_diff, se_diff, _ = mean_se(diffs)
    t = p = float("nan")
    scale = max([1.0] + [abs(a[s]) for s in seeds] + [abs(b[s]) for s in seeds])
    if n >= 2 and np.std(diffs, ddof=1) > ROUNDING_TOLERANCE * scale:
        result = stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds])
        t, p = float(result.statistic), float(result.pvalue)
    return {"n": n, "seeds": seeds, "mean_diff": mean_diff, "se_diff": se_diff,
            "t": t, "df": n - 1, "p": p}
