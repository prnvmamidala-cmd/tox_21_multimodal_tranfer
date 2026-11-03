"""Unlabeled-target adaptation. These methods are transductive.

They may use target-domain features. They must not use target labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def _center_cov(X: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    centered = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + eps * np.eye(cov.shape[0])
    return centered, cov


def _symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    scaled = eigenvalues**power
    return (eigenvectors * scaled) @ eigenvectors.T


def coral_align(
    source: np.ndarray, target: np.ndarray, eps: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Align source covariance toward the unlabeled target covariance (CORAL)."""
    source_c, source_cov = _center_cov(np.asarray(source, dtype=np.float64), eps)
    target_c, target_cov = _center_cov(np.asarray(target, dtype=np.float64), eps)
    transform = _symmetric_power(source_cov, -0.5) @ _symmetric_power(target_cov, 0.5)
    return source_c @ transform, target_c @ transform


def project_then_coral(
    source: np.ndarray, target: np.ndarray, n_components: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """PCA fit on the source, then CORAL in that space. Target labels are unused."""
    n_components = int(min(n_components, source.shape[0] - 1, source.shape[1], target.shape[0]))
    if n_components < 1:
        raise ValueError("Not enough rows for PCA.")
    pca = PCA(n_components=n_components, random_state=seed)
    source_z = pca.fit_transform(source)
    target_z = pca.transform(target)
    return coral_align(source_z, target_z)


def importance_weights(
    source: np.ndarray, target: np.ndarray, *, clip: tuple[float, float] = (0.05, 10.0)
) -> np.ndarray:
    """Density-ratio weights from a source-vs-target feature classifier.

    Target labels are not used. Weights apply to source rows.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    X = np.vstack([source, target])
    y = np.concatenate([np.zeros(len(source)), np.ones(len(target))])
    classifier = LogisticRegression(max_iter=400, solver="lbfgs")
    classifier.fit(X, y)
    probability = classifier.predict_proba(source)[:, 1]
    raw = probability / np.clip(1.0 - probability, 1e-6, None)
    return np.clip(raw, clip[0], clip[1])
