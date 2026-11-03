"""Unit tests for splits, preprocessing, metrics, permutations, and fusion."""

from __future__ import annotations

import numpy as np
import pytest

from toxtransfer.evaluation.leakage import (
    check_permutation_control,
    check_preprocess_ignores_test,
)
from toxtransfer.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    spearman_rho,
)
from toxtransfer.features.biology import permute_rows, rows_have_same_multiset
from toxtransfer.features.preprocess import fit_preprocess, transform
from toxtransfer.models.fusion import fuse, oracle_gate
from toxtransfer.models.linear import select_alpha_groupkfold
from toxtransfer.splits.scaffold import assert_split_integrity, group_kfold_indices
from toxtransfer.statistics.inference import benjamini_hochberg


def test_group_split_has_no_scaffold_or_id_leakage():
    groups = np.array([f"s{i // 3}" for i in range(30)])
    ids = np.array([f"c{i}" for i in range(30)])
    folds = group_kfold_indices(groups, 5)
    for train, test in folds:
        assert_split_integrity(ids, groups, train, test)
        assert len(set(ids[train]) & set(ids[test])) == 0


def test_preprocess_median_ignores_test_matrix():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(40, 5))
    test = rng.normal(size=(10, 5)) + 100
    continuous = np.array([True, True, False, False, True])
    state = fit_preprocess(train, continuous)
    mutated = test * 50
    state2 = fit_preprocess(train, continuous)
    assert np.allclose(state.median, state2.median)
    check_preprocess_ignores_test(train, mutated, continuous)
    transformed = transform(test, state)
    assert transformed.shape[0] == 10
    assert transformed.shape[1] <= 5


def test_permutation_preserves_marginals_and_breaks_order():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(25, 4))
    matrix[rng.random(matrix.shape) < 0.3] = 0
    permuted, order = permute_rows(matrix, 101)
    assert permuted.shape == matrix.shape
    assert rows_have_same_multiset(matrix, permuted)
    assert not np.array_equal(order, np.arange(len(order)))
    report = check_permutation_control(matrix, 202)
    assert report["preserves_multiset"]


def test_fusion_bounds_and_fallback():
    chem = np.array([0.1, 0.2, 0.3])
    other = np.array([1.0, 1.0, 1.0])
    assert np.allclose(fuse(chem, other, 0.0), chem)
    assert np.allclose(fuse(chem, other, 1.0), other)
    mixed = fuse(chem, other, np.array([0.0, 0.5, 1.0]))
    assert mixed[0] == pytest.approx(0.1)
    assert mixed[2] == pytest.approx(1.0)
    gate = oracle_gate(np.array([0.1, 1.0, 0.3]), chem, other)
    assert gate.tolist() == [0.0, 1.0, 0.0]


def test_spearman_and_calibration_metrics():
    y = np.array([0.1, 0.2, 0.4, 0.8, 1.1])
    assert spearman_rho(y, y) == pytest.approx(1.0)
    labels = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(labels, perfect)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert expected_calibration_error(labels, labels) == pytest.approx(0.0)
    assert metrics["mcc"] > 0


def test_benjamini_hochberg_orders_adjustments():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.2]))
    assert adjusted[0] <= adjusted[1] <= adjusted[2] + 1e-12
    assert adjusted[0] == pytest.approx(0.03)


def test_alpha_selection_does_not_accept_a_test_matrix():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=40)
    groups = np.array([i // 8 for i in range(40)])
    continuous = np.ones(3, dtype=bool)
    alpha = select_alpha_groupkfold(X, y, groups, continuous, [0.1, 10.0], 3)
    assert alpha in {0.1, 10.0}


def test_no_absolute_local_paths_in_source():
    root = Path = __import__("pathlib").Path
    banned = ("/workspace", "/tmp/", "/home/ubuntu", "/content/drive", "/Users/")
    for path in (root(__file__).resolve().parents[1] / "src").rglob("*.py"):
        text = path.read_text()
        for token in banned:
            assert token not in text, f"{path} contains {token}"
    for path in (root(__file__).resolve().parents[1] / "scripts").rglob("*.py"):
        text = path.read_text()
        for token in banned:
            assert token not in text, f"{path} contains {token}"
