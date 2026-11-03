"""Regression and classification metrics. Accuracy is not a primary metric."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from scipy.stats import spearmanr


def spearman_rho(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return float("nan")
    if np.nanstd(y[mask]) < 1e-12 or np.nanstd(pred[mask]) < 1e-12:
        return float("nan")
    rho, _ = spearmanr(y[mask], pred[mask])
    return float(rho)


def regression_errors(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    residual = y[mask] - pred[mask]
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "n": int(mask.sum()),
    }


def calibration_slope(y: np.ndarray, pred: np.ndarray) -> float:
    """OLS slope of y on pred. NaN when pred has no variance."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3 or np.std(pred[mask]) < 1e-12:
        return float("nan")
    x = pred[mask] - pred[mask].mean()
    yy = y[mask] - y[mask].mean()
    return float(np.dot(x, yy) / np.dot(x, x))


def expected_calibration_error(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width ECE for probabilities in [0, 1]."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (pred >= edges[i]) & (pred < edges[i + 1])
        else:
            mask = (pred >= edges[i]) & (pred <= edges[i + 1])
        if not np.any(mask):
            continue
        total += (mask.sum() / n) * abs(y[mask].mean() - pred[mask].mean())
    return float(total)


def classification_metrics(y: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    out = {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "mcc": float("nan"),
        "balanced_accuracy": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "n": int(len(y)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "prevalence": float(y.mean()) if len(y) else float("nan"),
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        return out
    out["auroc"] = float(roc_auc_score(y, pred))
    out["auprc"] = float(average_precision_score(y, pred))
    out["brier"] = float(brier_score_loss(y, np.clip(pred, 0, 1)))
    out["ece"] = expected_calibration_error(y, pred)
    label = (pred >= threshold).astype(int)
    out["mcc"] = float(matthews_corrcoef(y, label))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, label))
    positive = y == 1
    negative = y == 0
    out["sensitivity"] = float(label[positive].mean()) if positive.any() else float("nan")
    out["specificity"] = float((1 - label[negative]).mean()) if negative.any() else float("nan")
    return out
