"""Biological-vector coverage and row permutations."""

from __future__ import annotations

import numpy as np


def coverage_from_missingness(missingness: np.ndarray) -> np.ndarray:
    """Fraction of targets observed. ``missingness`` is 1 where the value is absent."""
    miss = np.asarray(missingness, dtype=np.float64)
    if miss.ndim != 2 or miss.shape[1] == 0:
        raise ValueError("missingness must be a non-empty 2-d matrix.")
    return 1.0 - miss.mean(axis=1)


def permute_rows(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Permute rows without replacement.

    The returned matrix has the same multiset of rows, the same shape, and the
    same per-column marginals. Compound-to-row correspondence is broken.
    """
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("Expected a 2-d matrix.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(values.shape[0])
    return values[order], order


def rows_have_same_multiset(left: np.ndarray, right: np.ndarray) -> bool:
    """True when ``left`` and ``right`` contain the same rows regardless of order."""
    if left.shape != right.shape:
        return False
    left_key = np.ascontiguousarray(np.round(left.astype(np.float64), 8))
    right_key = np.ascontiguousarray(np.round(right.astype(np.float64), 8))
    left_sorted = left_key[np.lexsort(left_key.T[::-1])]
    right_sorted = right_key[np.lexsort(right_key.T[::-1])]
    return bool(np.array_equal(left_sorted, right_sorted))
