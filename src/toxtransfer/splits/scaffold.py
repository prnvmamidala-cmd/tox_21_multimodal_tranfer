"""Bemis–Murcko grouped splits and integrity checks."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def group_kfold_indices(
    groups: np.ndarray, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic GroupKFold. Raises if there are fewer groups than folds."""
    groups = np.asarray(groups)
    n_groups = len(set(groups.tolist()))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_groups < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups for GroupKFold; found {n_groups}."
        )
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return [(train, test) for train, test in splitter.split(dummy, groups=groups)]


def group_holdout_indices(
    groups: np.ndarray, *, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """One GroupShuffleSplit. Used only for the labeled target-only reference."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dummy = np.zeros(len(groups))
    train, test = next(splitter.split(dummy, groups=groups))
    return train, test


def split_overlap_counts(
    identifiers: np.ndarray, groups: np.ndarray, train: np.ndarray, test: np.ndarray
) -> dict[str, int]:
    """Count identifier and group collisions across a split."""
    id_overlap = set(identifiers[train].tolist()) & set(identifiers[test].tolist())
    group_overlap = set(groups[train].tolist()) & set(groups[test].tolist())
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_identifier_overlap": len(id_overlap),
        "n_group_overlap": len(group_overlap),
    }


def assert_split_integrity(
    identifiers: np.ndarray, groups: np.ndarray, train: np.ndarray, test: np.ndarray
) -> None:
    """Fail if a compound or scaffold is on both sides of the split."""
    counts = split_overlap_counts(identifiers, groups, train, test)
    if counts["n_identifier_overlap"] or counts["n_group_overlap"]:
        raise RuntimeError(f"Split integrity failed: {counts}")
