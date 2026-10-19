"""Paired bootstrap intervals, Wilcoxon tests, and Benjamini–Hochberg."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values, NaN preserved."""
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return adjusted
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    running = 1.0
    tmp = np.empty(m)
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        tmp[i] = running
    out = np.empty(m)
    out[order] = np.clip(tmp, 0.0, 1.0)
    adjusted[np.where(valid)[0]] = out
    return adjusted


def paired_bootstrap_delta(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the paired difference metric(a) - metric(b) over rows."""
    y = np.asarray(y, dtype=np.float64)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    observed = metric(y, pred_a) - metric(y, pred_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = metric(y[idx], pred_a[idx]) - metric(y[idx], pred_b[idx])
    finite = deltas[np.isfinite(deltas)]
    if len(finite) == 0:
        lo = hi = p = float("nan")
    else:
        lo, hi = np.quantile(finite, [0.025, 0.975])
        # Two-sided bootstrap p: twice the smaller tail beyond 0.
        p = float(min(1.0, 2.0 * min(np.mean(finite <= 0), np.mean(finite >= 0))))
    return {
        "delta": float(observed) if np.isfinite(observed) else float("nan"),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_bootstrap": float(p),
        "n": int(n),
        "n_boot_finite": int(len(finite)),
    }


def bootstrap_statistic(values: np.ndarray, statistic, *, n_boot: int, seed: int) -> dict[str, float]:
    """Bootstrap a statistic of a 1-d sample (for example the mean of endpoint deltas)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    estimate = float(statistic(values))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        boots[i] = statistic(sample)
    lo, hi = np.quantile(boots[np.isfinite(boots)], [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(len(values)),
    }


def wilcoxon_signed_rank(deltas: np.ndarray) -> dict[str, float | str]:
    """Two-sided Wilcoxon signed-rank test.

    Endpoints share compounds, so the independence assumption is false. The
    p-value is descriptive. The paired bootstrap interval is the primary
    uncertainty statement.
    """
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    note = (
        "Wilcoxon signed-rank assumes a symmetric null distribution of paired "
        "deltas and independent pairs. Endpoints share compounds, so "
        "independence fails; treat p as descriptive."
    )
    if len(d) < 5 or np.allclose(d, 0):
        return {"statistic": float("nan"), "p": float("nan"), "n": int(len(d)), "note": note}
    result = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
    return {
        "statistic": float(result.statistic),
        "p": float(result.pvalue),
        "n": int(len(d)),
        "note": note,
    }
