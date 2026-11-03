"""Train-only imputation, clipping, variance filtering, and scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrepState:
    """Statistics fit on a training matrix only."""

    median: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    keep: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    continuous: np.ndarray


def _as_float(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional.")
    return np.where(np.isfinite(matrix), matrix, np.nan)


def fit_preprocess(
    X: np.ndarray,
    continuous_mask: np.ndarray,
    *,
    clip: bool = True,
    scale: bool = True,
) -> PrepState:
    """Fit preprocessing on training rows.

    Test-matrix values are not an input. Quantiles, medians, and scales are
    training statistics.
    """
    matrix = _as_float(X)
    n_rows, n_cols = matrix.shape
    if n_rows < 1:
        raise ValueError("Cannot fit preprocessing on an empty training matrix.")
    continuous = np.asarray(continuous_mask, dtype=bool)
    if continuous.shape != (n_cols,):
        raise ValueError("continuous_mask length must equal the number of columns.")
    median = np.nanmedian(matrix, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    imputed = matrix.copy()
    missing = np.isnan(imputed)
    if missing.any():
        imputed[missing] = np.take(median, np.where(missing)[1])
    lo = np.full(n_cols, -np.inf)
    hi = np.full(n_cols, np.inf)
    if clip and continuous.any():
        lo_c = np.quantile(imputed[:, continuous], 0.001, axis=0)
        hi_c = np.quantile(imputed[:, continuous], 0.999, axis=0)
        lo_c = np.where(np.isfinite(lo_c), lo_c, median[continuous])
        hi_c = np.where(np.isfinite(hi_c), hi_c, median[continuous])
        swap = hi_c < lo_c
        lo_c[swap], hi_c[swap] = hi_c[swap], lo_c[swap]
        lo[continuous] = lo_c
        hi[continuous] = hi_c
        imputed[:, continuous] = np.clip(imputed[:, continuous], lo_c, hi_c)
    keep = imputed.var(axis=0) > 1e-12
    if not np.any(keep):
        keep = np.zeros(n_cols, dtype=bool)
        keep[0] = True
    kept = imputed[:, keep]
    if scale:
        mean = kept.mean(axis=0)
        std = kept.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    else:
        mean = np.zeros(int(keep.sum()))
        std = np.ones(int(keep.sum()))
    return PrepState(median, lo, hi, keep, mean, std, continuous)


def transform(X: np.ndarray, state: PrepState, *, scale: bool = True) -> np.ndarray:
    """Apply a frozen :class:`PrepState` to any matrix with the same columns."""
    matrix = _as_float(X)
    if matrix.shape[1] != state.median.shape[0]:
        raise ValueError("Column count does not match the fitted preprocessor.")
    imputed = matrix.copy()
    missing = np.isnan(imputed)
    if missing.any():
        imputed[missing] = np.take(state.median, np.where(missing)[1])
    continuous = state.continuous
    imputed[:, continuous] = np.clip(
        imputed[:, continuous], state.lo[continuous], state.hi[continuous]
    )
    kept = imputed[:, state.keep]
    if scale:
        kept = (kept - state.mean) / state.scale
    return kept
