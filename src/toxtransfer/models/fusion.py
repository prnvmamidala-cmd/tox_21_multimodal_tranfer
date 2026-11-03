"""Reliability-aware and fixed prediction fusion.

The deployed gate uses only information available without the evaluation label.
The oracle gate uses labels and is not a deployable method.
"""

from __future__ import annotations

import hashlib

import numpy as np


def fuse(pred_chem: np.ndarray, pred_other: np.ndarray, gate: np.ndarray | float) -> np.ndarray:
    """``(1 - g) * chem + g * other`` with g clipped to [0, 1]."""
    chem = np.asarray(pred_chem, dtype=np.float64)
    other = np.asarray(pred_other, dtype=np.float64)
    weight = np.clip(np.asarray(gate, dtype=np.float64), 0.0, 1.0)
    return (1.0 - weight) * chem + weight * other


def select_global_weight(
    y: np.ndarray,
    pred_chem: np.ndarray,
    pred_other: np.ndarray,
    grid: list[float],
    metric,
) -> float:
    """Choose one weight on validation predictions. ``metric`` is higher-is-better."""
    best_weight = float(grid[0])
    best_score = -np.inf
    for weight in grid:
        score = metric(y, fuse(pred_chem, pred_other, weight))
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_weight = float(weight)
    return best_weight


def oracle_gate(y: np.ndarray, pred_chem: np.ndarray, pred_other: np.ndarray) -> np.ndarray:
    """1 when the other prediction is closer to the label than chemistry.

    This uses labels. Report it only as a non-deployable ceiling.
    """
    y = np.asarray(y, dtype=np.float64)
    closer = np.abs(y - pred_other) < np.abs(y - pred_chem)
    return closer.astype(np.float64)


def deterministic_uniform_gate(identifiers: np.ndarray, seed: int) -> np.ndarray:
    """Pseudo-uniform gates in [0, 1], keyed by identifier so row order does not matter."""
    gates = np.empty(len(identifiers), dtype=np.float64)
    for i, identifier in enumerate(identifiers):
        digest = hashlib.sha256(f"{int(seed)}:{identifier}".encode()).digest()
        mixed = int.from_bytes(digest[:4], "little")
        gates[i] = mixed / 0xFFFFFFFF
    return gates
