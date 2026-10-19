"""Ridge regression with train-only preprocessing."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from toxtransfer.evaluation.metrics import spearman_rho
from toxtransfer.features.preprocess import fit_preprocess, transform
from toxtransfer.splits.scaffold import group_kfold_indices


def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    continuous_mask: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Fit ridge on the training matrix and predict the test matrix."""
    state = fit_preprocess(X_train, continuous_mask, clip=True, scale=True)
    Z_train = transform(X_train, state, scale=True)
    Z_test = transform(X_test, state, scale=True)
    model = Ridge(alpha=float(alpha), solver="lsqr")
    model.fit(Z_train, y_train, sample_weight=sample_weight)
    return model.predict(Z_test)


def select_alpha_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    continuous_mask: np.ndarray,
    alphas: list[float],
    n_splits: int,
) -> float:
    """Pick alpha by inner scaffold GroupKFold. The caller must pass training rows only."""
    folds = group_kfold_indices(groups, n_splits)
    scores = {float(alpha): [] for alpha in alphas}
    for train, test in folds:
        for alpha in scores:
            pred = fit_predict_ridge(
                X[train], y[train], X[test], continuous_mask, alpha
            )
            scores[alpha].append(spearman_rho(y[test], pred))
    means = {alpha: np.nanmean(vals) for alpha, vals in scores.items()}
    # Ties break toward the first alpha in the pre-specified grid.
    best = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        score = means[float(alpha)]
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best = float(alpha)
    return best
