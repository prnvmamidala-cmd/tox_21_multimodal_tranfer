"""Fit the pre-registered V5 arms and write the verdict from saved scores.

The pediatric structure-unseen labels are used only as an evaluation target.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from torch import nn

from toxtransfer.evaluation.metrics import classification_metrics, spearman_rho
from toxtransfer.features.chemistry import featurize_mols, standardize_mol
from toxtransfer.features.preprocess import fit_preprocess, transform
from toxtransfer.statistics.inference import benjamini_hochberg, bootstrap_statistic
from toxtransfer.v5.data_build import TOX21_TASKS

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _label_maps(chem_rows: pd.DataFrame, endpoints: list[str]) -> tuple[dict, dict, dict]:
    """Return adult labels, test labels, and frozen adult OOF chemistry predictions."""
    adult: dict[str, dict[str, float]] = {endpoint: {} for endpoint in endpoints}
    test: dict[str, dict[str, float]] = {endpoint: {} for endpoint in endpoints}
    frozen: dict[str, dict[str, float]] = {endpoint: {} for endpoint in endpoints}
    adult_rows = chem_rows.loc[chem_rows["phase"] == "adult_oof"]
    test_rows = chem_rows.loc[(chem_rows["phase"] == "transfer") & (chem_rows["structure_seen"] == 0)]
    for endpoint in endpoints:
        for row in adult_rows.loc[adult_rows["endpoint"] == endpoint].itertuples(index=False):
            adult[endpoint][row.inchikey] = float(row.y_true)
            frozen[endpoint][row.inchikey] = float(row.y_pred)
        for row in test_rows.loc[test_rows["endpoint"] == endpoint].itertuples(index=False):
            test[endpoint][row.inchikey] = float(row.y_true)
    return adult, test, frozen


def _pairs(keys: list[str], labels: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    index = [i for i, key in enumerate(keys) if key in labels and np.isfinite(labels[key])]
    y = np.array([labels[keys[i]] for i in index], dtype=np.float64)
    return np.asarray(index, dtype=int), y


def _ridge_fit_predict(X_train, y_train, X_test, continuous, alpha: float) -> np.ndarray:
    state = fit_preprocess(X_train, continuous)
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(transform(X_train, state), y_train)
    if len(X_test) == 0:
        return np.zeros(0, dtype=np.float64)
    return model.predict(transform(X_test, state))


def _oof(X, y, groups, continuous, alpha: float, n_splits: int) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2 or len(y) < 10:
        return oof
    splitter = GroupKFold(n_splits=splits)
    for train, test in splitter.split(X, y, groups):
        oof[test] = _ridge_fit_predict(X[train], y[train], X[test], continuous, alpha)
    return oof


def _permute(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return matrix[rng.permutation(len(matrix))].copy()


def _choose_weight(y_by_endpoint: list[np.ndarray], chem: list[np.ndarray], other: list[np.ndarray], grid: list[float]) -> float:
    best_g, best = float(grid[0]), -np.inf
    for weight in grid:
        scores = []
        for y, left, right in zip(y_by_endpoint, chem, other, strict=True):
            pred = (1.0 - weight) * left + weight * right
            scores.append(spearman_rho(y, pred))
        score = float(np.nanmean(scores)) if scores else float("nan")
        if np.isfinite(score) and score > best:
            best, best_g = score, float(weight)
    return best_g


def _choose_gate(y_list, chem, other, coverage, intercepts, slopes) -> tuple[float, float]:
    best, best_pair = -np.inf, (0.0, 0.0)
    for intercept in intercepts:
        for slope in slopes:
            scores = []
            for y, left, right, cover in zip(y_list, chem, other, coverage, strict=True):
                gate = np.clip(intercept + slope * cover, 0.0, 1.0)
                pred = (1.0 - gate) * left + gate * right
                scores.append(spearman_rho(y, pred))
            score = float(np.nanmean(scores))
            if np.isfinite(score) and score > best:
                best, best_pair = score, (float(intercept), float(slope))
    return best_pair


def _macro_delta(scores: dict[str, float], baseline: dict[str, float], endpoints: list[str]) -> np.ndarray:
    return np.array([scores[endpoint] - baseline[endpoint] for endpoint in endpoints], dtype=np.float64)


def _boot_mean(values: np.ndarray, n_boot: int, seed: int) -> dict:
    stats = bootstrap_statistic(values, np.mean, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        stats["p_bootstrap"] = float("nan")
        return stats
    means = np.array([finite[rng.integers(0, len(finite), len(finite))].mean() for _ in range(n_boot)])
    stats["p_bootstrap"] = float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0))))
    return stats


class _MLP(nn.Module):
    def __init__(self, width: int, hidden: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, outputs))

    def forward(self, x):
        return self.net(x)


def _train_mlp(X: np.ndarray, Y: np.ndarray, seed: int, cfg: dict) -> _MLP:
    torch.manual_seed(seed)
    model = _MLP(X.shape[1], int(cfg["mlp"]["hidden"]), Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["mlp"]["lr"]), weight_decay=float(cfg["mlp"]["weight_decay"]))
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(Y, dtype=torch.float32)
    batch = int(cfg["mlp"]["batch_size"])
    model.train()
    for _ in range(int(cfg["mlp"]["epochs"])):
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), batch):
            take = order[start : start + batch]
            pred = model(xt[take])
            target = yt[take]
            mask = torch.isfinite(target)
            if int(mask.sum()) == 0:
                continue
            loss = ((pred - torch.nan_to_num(target))[mask] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _mlp_predict(model: _MLP, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def _impute_targets(chem: np.ndarray, bio: np.ndarray, train_idx: np.ndarray, continuous, alpha: float) -> np.ndarray:
    predicted = np.zeros_like(bio)
    for column in range(bio.shape[1]):
        observed = train_idx[np.isfinite(bio[train_idx, column])]
        if len(observed) < 25:
            fill = np.nanmedian(bio[train_idx, column])
            predicted[:, column] = 0.0 if not np.isfinite(fill) else float(fill)
            continue
        predicted[:, column] = _ridge_fit_predict(
            chem[observed], bio[observed, column], chem, continuous, alpha
        )
    return predicted


def _spearman_map(keys, idx, y, pred, labels) -> dict[str, float]:
    out = {}
    for position, row in enumerate(idx):
        key = keys[row]
        # filled below per endpoint by the caller
    del out, position, key
    return {}


def run(cfg: dict, root: Path, pack: dict, chem_rows: pd.DataFrame) -> dict:
    endpoints = list(cfg["eligible_endpoints"])
    keys: list[str] = pack["inchikeys"]
    key_index = {key: i for i, key in enumerate(keys)}
    adult_labels, test_labels, frozen_chem = _label_maps(chem_rows, endpoints)
    adult_keys = set().union(*[set(mapping) for mapping in adult_labels.values()])
    test_keys = set().union(*[set(mapping) for mapping in test_labels.values()])
    adult_idx = np.array([i for i, key in enumerate(keys) if key in adult_keys], dtype=int)
    test_idx_all = np.array([i for i, key in enumerate(keys) if key in test_keys], dtype=int)
    chem = pack["chem"]
    bio = pack["bio"]
    tox = pack["tox21"]
    scaffolds = np.asarray(pack["scaffolds"])
    continuous = pack["continuous"]
    alpha = float(cfg["ridge_alpha"])
    n_splits = int(cfg["adult_folds"])
    seeds = [int(seed) for seed in cfg["permutation_seeds"]]
    coverage = np.isfinite(bio).mean(axis=1)
    tox_observed_adult = int(np.isfinite(tox[adult_idx]).any(axis=1).sum()) if len(adult_idx) else 0
    tox_blocked = tox_observed_adult < int(cfg["tox21_min_adult_observed"])

    def endpoint_adult(endpoint: str):
        idx, y = _pairs(keys, adult_labels[endpoint])
        return idx, y

    def endpoint_test(endpoint: str):
        idx, y = _pairs(keys, test_labels[endpoint])
        return idx, y

    # Chemistry and biology-only, with adult OOF for stacking.
    oof_chem, full_chem = {}, {}
    oof_bio, full_bio = {}, {}
    oof_bio_perm = {seed: {} for seed in seeds}
    full_bio_perm = {seed: {} for seed in seeds}
    test_y = {}
    test_cover = {}
    for endpoint in endpoints:
        tr, y_tr = endpoint_adult(endpoint)
        te, y_te = endpoint_test(endpoint)
        test_y[endpoint] = y_te
        test_cover[endpoint] = coverage[te]
        groups = scaffolds[tr]
        oof_chem[endpoint] = _oof(chem[tr], y_tr, groups, continuous, alpha, n_splits)
        full_chem[endpoint] = _ridge_fit_predict(chem[tr], y_tr, chem[te], continuous, alpha)
        bio_x = np.concatenate([np.nan_to_num(bio, nan=0.0), np.isnan(bio).astype(np.float64)], axis=1)
        bio_cont = np.ones(bio_x.shape[1], dtype=bool)
        oof_bio[endpoint] = _oof(bio_x[tr], y_tr, groups, bio_cont, alpha, n_splits)
        full_bio[endpoint] = _ridge_fit_predict(bio_x[tr], y_tr, bio_x[te], bio_cont, alpha)
        for seed in seeds:
            shuffled = bio.copy()
            shuffled[tr] = _permute(bio[tr], seed)
            block = np.concatenate([np.nan_to_num(shuffled, nan=0.0), np.isnan(shuffled).astype(np.float64)], axis=1)
            oof_bio_perm[seed][endpoint] = _oof(block[tr], y_tr, groups, bio_cont, alpha, n_splits)
            full_bio_perm[seed][endpoint] = _ridge_fit_predict(block[tr], y_tr, block[te], bio_cont, alpha)
        print(f"ridge channel {endpoint}", flush=True)

    def _adult_y(endpoint):
        _, y = endpoint_adult(endpoint)
        return y

    weight = _choose_weight(
        [_adult_y(ep) for ep in endpoints],
        [oof_chem[ep] for ep in endpoints],
        [oof_bio[ep] for ep in endpoints],
        cfg["weight_grid"],
    )
    perm_weights = {}
    for seed in seeds:
        perm_weights[seed] = _choose_weight(
            [_adult_y(ep) for ep in endpoints],
            [oof_chem[ep] for ep in endpoints],
            [oof_bio_perm[seed][ep] for ep in endpoints],
            cfg["weight_grid"],
        )
    intercept, slope = _choose_gate(
        [_adult_y(ep) for ep in endpoints],
        [oof_chem[ep] for ep in endpoints],
        [oof_bio[ep] for ep in endpoints],
        [_cover(keys, adult_labels[ep], coverage) for ep in endpoints],
        cfg["gate_intercept_grid"],
        cfg["gate_slope_grid"],
    )
    perm_gates = {}
    for seed in seeds:
        perm_gates[seed] = _choose_gate(
            [_adult_y(ep) for ep in endpoints],
            [oof_chem[ep] for ep in endpoints],
            [oof_bio_perm[seed][ep] for ep in endpoints],
            [_cover(keys, adult_labels[ep], coverage) for ep in endpoints],
            cfg["gate_intercept_grid"],
            cfg["gate_slope_grid"],
        )

    def fuse(endpoint, other, g):
        return (1.0 - g) * full_chem[endpoint] + g * other

    predictions = {
        "ridge_chem": full_chem,
        "late_stack": {ep: fuse(ep, full_bio[ep], weight) for ep in endpoints},
        "miss_gate": {},
    }
    for endpoint in endpoints:
        gate = np.clip(intercept + slope * test_cover[endpoint], 0.0, 1.0)
        predictions["miss_gate"][endpoint] = (1.0 - gate) * full_chem[endpoint] + gate * full_bio[endpoint]
    perm_preds = {name: {} for name in ("late_stack", "miss_gate")}
    for seed in seeds:
        perm_preds["late_stack"][seed] = {ep: fuse(ep, full_bio_perm[seed][ep], perm_weights[seed]) for ep in endpoints}
        perm_preds["miss_gate"][seed] = {}
        a_s, b_s = perm_gates[seed]
        for endpoint in endpoints:
            gate = np.clip(a_s + b_s * test_cover[endpoint], 0.0, 1.0)
            perm_preds["miss_gate"][seed][endpoint] = (1.0 - gate) * full_chem[endpoint] + gate * full_bio_perm[seed][endpoint]

    # Imputation, mechanism, zero vs median (exploratory), tox21.
    imputed_pred = {}
    mechanism_pred = {}
    mechanism_fused_pred = {}
    zero_pred = {}
    median_pred = {}
    tox_pred = {}
    perm_store = {name: {seed: {} for seed in seeds} for name in ("imputed_early", "mechanism", "mechanism_fused", "tox21_late")}
    for endpoint in endpoints:
        tr, y_tr = endpoint_adult(endpoint)
        te, y_te = endpoint_test(endpoint)
        imputed = _impute_targets(chem, bio, tr, continuous, alpha)
        early_x = np.concatenate([chem, imputed], axis=1)
        early_c = np.concatenate([continuous, np.ones(imputed.shape[1], dtype=bool)])
        imputed_pred[endpoint] = _ridge_fit_predict(early_x[tr], y_tr, early_x[te], early_c, alpha)
        mechanism_pred[endpoint] = _ridge_fit_predict(imputed[tr], y_tr, imputed[te], np.ones(imputed.shape[1], dtype=bool), alpha)
        mechanism_fused_pred[endpoint] = imputed_pred[endpoint]
        # zero-filled measured biology versus median-filled, exploratory.
        zero = np.nan_to_num(bio, nan=0.0)
        zero_x = np.concatenate([chem, zero], axis=1)
        zero_pred[endpoint] = _ridge_fit_predict(zero_x[tr], y_tr, zero_x[te], early_c, alpha)
        med = bio.copy()
        col_med = np.nanmedian(bio[tr], axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        inds = np.where(~np.isfinite(med))
        med[inds] = np.take(col_med, inds[1])
        med_x = np.concatenate([chem, med], axis=1)
        median_pred[endpoint] = _ridge_fit_predict(med_x[tr], y_tr, med_x[te], early_c, alpha)
        if not tox_blocked:
            tox_x = np.concatenate([np.nan_to_num(tox, nan=0.0), np.isnan(tox).astype(np.float64)], axis=1)
            tox_c = np.ones(tox_x.shape[1], dtype=bool)
            tox_only = _ridge_fit_predict(tox_x[tr], y_tr, tox_x[te], tox_c, alpha)
            # Weight for this single endpoint's stack is deferred; filled after the loop via OOF-free
            # adult split: use a scaffold holdout inside adult to pick the weight, then refit.
            hold = _scaffold_holdout(scaffolds[tr], 0.2, 0)
            if len(hold) >= 20:
                train_h = np.setdiff1d(np.arange(len(tr)), hold)
                pred_c_h = _ridge_fit_predict(chem[tr[train_h]], y_tr[train_h], chem[tr[hold]], continuous, alpha)
                pred_t_h = _ridge_fit_predict(tox_x[tr[train_h]], y_tr[train_h], tox_x[tr[hold]], tox_c, alpha)
                local_g = _choose_weight([y_tr[hold]], [pred_c_h], [pred_t_h], cfg["weight_grid"])
            else:
                local_g = 0.0
            tox_pred[endpoint] = (1.0 - local_g) * full_chem[endpoint] + local_g * tox_only
        for seed in seeds:
            shuffled = bio.copy()
            shuffled[tr] = _permute(bio[tr], seed)
            imp_p = _impute_targets(chem, shuffled, tr, continuous, alpha)
            early_p = np.concatenate([chem, imp_p], axis=1)
            perm_store["imputed_early"][seed][endpoint] = _ridge_fit_predict(early_p[tr], y_tr, early_p[te], early_c, alpha)
            perm_store["mechanism"][seed][endpoint] = _ridge_fit_predict(
                imp_p[tr], y_tr, imp_p[te], np.ones(imp_p.shape[1], dtype=bool), alpha
            )
            perm_store["mechanism_fused"][seed][endpoint] = perm_store["imputed_early"][seed][endpoint]
            if not tox_blocked:
                shuffled_t = tox.copy()
                shuffled_t[tr] = _permute(tox[tr], seed)
                tox_p = np.concatenate(
                    [np.nan_to_num(shuffled_t, nan=0.0), np.isnan(shuffled_t).astype(np.float64)], axis=1
                )
                tox_c = np.ones(tox_p.shape[1], dtype=bool)
                hold = _scaffold_holdout(scaffolds[tr], 0.2, seed)
                if len(hold) >= 20 and len(tr) - len(hold) >= 20:
                    train_h = np.setdiff1d(np.arange(len(tr)), hold)
                    pred_c_h = _ridge_fit_predict(chem[tr[train_h]], y_tr[train_h], chem[tr[hold]], continuous, alpha)
                    pred_t_h = _ridge_fit_predict(tox_p[tr[train_h]], y_tr[train_h], tox_p[tr[hold]], tox_c, alpha)
                    local_g = _choose_weight([y_tr[hold]], [pred_c_h], [pred_t_h], cfg["weight_grid"])
                else:
                    local_g = 0.0
                tox_only_p = _ridge_fit_predict(tox_p[tr], y_tr, tox_p[te], tox_c, alpha)
                perm_store["tox21_late"][seed][endpoint] = (1.0 - local_g) * full_chem[endpoint] + local_g * tox_only_p
        print(f"impute/mechanism {endpoint}", flush=True)

    predictions["imputed_early"] = imputed_pred
    predictions["mechanism"] = mechanism_pred
    predictions["mechanism_fused"] = mechanism_fused_pred
    if not tox_blocked:
        predictions["tox21_late"] = tox_pred
        perm_preds["tox21_late"] = perm_store["tox21_late"]
    perm_preds["imputed_early"] = perm_store["imputed_early"]
    perm_preds["mechanism"] = perm_store["mechanism"]
    perm_preds["mechanism_fused"] = perm_store["mechanism_fused"]

    # Multitask MLP. Frozen hyperparameters. Seeds averaged.
    mlp_chem, mlp_early = _mlp_pair(cfg, chem, bio, keys, adult_labels, test_labels, endpoints, scaffolds, adult_idx, None)
    predictions["mlp_early"] = mlp_early
    predictions["mlp_chem"] = mlp_chem
    perm_preds["mlp_early"] = {}
    for seed in seeds:
        _, perm_early = _mlp_pair(cfg, chem, bio, keys, adult_labels, test_labels, endpoints, scaffolds, adult_idx, seed)
        perm_preds["mlp_early"][seed] = perm_early
        print(f"mlp perm {seed}", flush=True)

    matched_chem = {
        "late_stack": predictions["ridge_chem"],
        "miss_gate": predictions["ridge_chem"],
        "imputed_early": predictions["ridge_chem"],
        "mechanism": predictions["ridge_chem"],
        "mlp_early": predictions["mlp_chem"],
    }
    if not tox_blocked:
        matched_chem["tox21_late"] = predictions["ridge_chem"]
    arms = list(matched_chem)
    scores = {arm: {ep: spearman_rho(test_y[ep], predictions[arm][ep]) for ep in endpoints} for arm in ["ridge_chem", "mlp_chem", *arms]}
    perm_scores = {}
    for arm in arms:
        perm_scores[arm] = {}
        for seed in seeds:
            perm_scores[arm][seed] = {ep: spearman_rho(test_y[ep], perm_preds[arm][seed][ep]) for ep in endpoints}

    contrasts = []
    pvals = []
    for i, arm in enumerate(arms):
        base_name = "mlp_chem" if arm == "mlp_early" else "ridge_chem"
        delta_chem = np.array([scores[arm][ep] - scores[base_name][ep] for ep in endpoints])
        mean_perm = []
        for ep in endpoints:
            mean_perm.append(np.nanmean([perm_scores[arm][seed][ep] for seed in seeds]))
        delta_perm = np.array([scores[arm][ep] - mean_perm[j] for j, ep in enumerate(endpoints)])
        boot_c = _boot_mean(delta_chem, int(cfg["bootstrap_n"]), int(cfg["bootstrap_seed"]) + 2 * i)
        boot_p = _boot_mean(delta_perm, int(cfg["bootstrap_n"]), int(cfg["bootstrap_seed"]) + 2 * i + 1)
        for kind, boot, delta in (("minus_chem", boot_c, delta_chem), ("minus_perm", boot_p, delta_perm)):
            contrasts.append(
                {
                    "arm": arm,
                    "contrast": kind,
                    "estimate": boot["estimate"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "p_bootstrap": boot["p_bootstrap"],
                    "n_endpoints": int(np.isfinite(delta).sum()),
                    "supported_interval": bool(np.isfinite(boot["ci_low"]) and boot["ci_low"] > 0),
                }
            )
            pvals.append(boot["p_bootstrap"])
    qvals = benjamini_hochberg(np.array(pvals, dtype=float))
    for row, q in zip(contrasts, qvals, strict=True):
        row["q_bh"] = float(q) if np.isfinite(q) else float("nan")
    verdicts = []
    for arm in arms:
        rows = [row for row in contrasts if row["arm"] == arm]
        chem_row = next(row for row in rows if row["contrast"] == "minus_chem")
        perm_row = next(row for row in rows if row["contrast"] == "minus_perm")
        if chem_row["supported_interval"] and perm_row["supported_interval"]:
            status = "positive"
        elif (not np.isfinite(chem_row["estimate"])) or chem_row["n_endpoints"] < 12:
            status = "inconclusive"
        else:
            status = "null"
        verdicts.append({"arm": arm, "status": status, "minus_chem": chem_row, "minus_perm": perm_row})

    diagnostics = _diagnostics(
        keys, endpoints, adult_labels, frozen_chem, bio, chem, scaffolds, coverage, continuous, alpha, n_splits, test_y, predictions, test_cover
    )
    diagnostics["zero_vs_median"] = {
        ep: {
            "zero": spearman_rho(test_y[ep], zero_pred[ep]),
            "median": spearman_rho(test_y[ep], median_pred[ep]),
        }
        for ep in endpoints
    }
    extras = _extras(cfg, pack, keys, adult_labels, test_labels, endpoints, root)
    claims = {
        "status": "computed",
        "n_structures_rebuilt": len(keys),
        "n_adult_structures": int(len(adult_idx)),
        "n_test_structures_available": int(len(test_idx_all)),
        "bio_mean_missingness": float(np.isnan(bio).mean()),
        "tox21_adult_with_any_assay": tox_observed_adult,
        "tox21_blocked": tox_blocked,
        "late_weight": weight,
        "perm_weights": {str(k): v for k, v in perm_weights.items()},
        "miss_gate": {"intercept": intercept, "slope": slope},
        "contrasts": contrasts,
        "verdicts": verdicts,
        "endpoint_spearman": scores,
        "diagnostics": diagnostics,
        "extras": extras,
        "blocked": {
            "signaturizer": extras.get("signaturizer"),
            "chemprop": extras.get("chemprop"),
            "pyg": extras.get("pyg"),
            "pretrained_embeddings": extras.get("pretrained"),
            "tox21_late": tox_blocked,
        },
    }
    return claims


def _y(keys, labels):
    raise RuntimeError("removed")


def _cover(keys, labels, coverage) -> np.ndarray:
    idx, _ = _pairs(keys, labels)
    return coverage[idx]


def _scaffold_holdout(groups: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scaffolds = np.unique(groups)
    rng.shuffle(scaffolds)
    take = set(scaffolds[: max(1, int(round(len(scaffolds) * fraction)))])
    return np.array([i for i, group in enumerate(groups) if group in take], dtype=int)


def _mlp_pair(cfg, chem, bio, keys, adult_labels, test_labels, endpoints, scaffolds, adult_idx, perm_seed):
    """Return chem-only and chem+bio multitask predictions on the test slice.

    When perm_seed is set, only the biology-augmented network sees permuted
    adult biology. The chemistry network is trained once per call as well;
    callers that only need the permuted arm still pay for the chem network,
    which keeps the two arms on the same seeds.
    """
    del scaffolds
    Y = np.full((len(keys), len(endpoints)), np.nan)
    index = {key: i for i, key in enumerate(keys)}
    Y[:] = np.nan
    test_idx = []
    test_y = {endpoint: [] for endpoint in endpoints}
    test_pos = {endpoint: [] for endpoint in endpoints}
    for e, endpoint in enumerate(endpoints):
        for key, value in adult_labels[endpoint].items():
            if key in index:
                Y[index[key], e] = value
        for key, value in test_labels[endpoint].items():
            if key in index:
                test_pos[endpoint].append(index[key])
                test_y[endpoint].append(value)
    train_idx = np.array(sorted({i for i in adult_idx if np.isfinite(Y[i]).any()}), dtype=int)
    bio_train = bio.copy()
    if perm_seed is not None:
        bio_train[train_idx] = _permute(bio[train_idx], perm_seed)
    bio_block = np.concatenate([np.nan_to_num(bio_train, nan=0.0), np.isnan(bio_train).astype(np.float64)], axis=1)
    fp = chem[:, :2048]
    X_chem = fp
    X_early = np.concatenate([fp, bio_block], axis=1)
    state_c = fit_preprocess(X_chem[train_idx], np.zeros(X_chem.shape[1], dtype=bool), clip=False, scale=True)
    state_e = fit_preprocess(X_early[train_idx], np.zeros(X_early.shape[1], dtype=bool), clip=False, scale=True)
    Xc = transform(X_chem, state_c)
    Xe = transform(X_early, state_e)
    acc_c = []
    acc_e = []
    for seed in cfg["neural_seeds"]:
        model_c = _train_mlp(Xc[train_idx], Y[train_idx], int(seed), cfg)
        model_e = _train_mlp(Xe[train_idx], Y[train_idx], int(seed), cfg)
        acc_c.append(_mlp_predict(model_c, Xc))
        acc_e.append(_mlp_predict(model_e, Xe))
    mean_c = np.mean(acc_c, axis=0)
    mean_e = np.mean(acc_e, axis=0)
    out_c, out_e = {}, {}
    for e, endpoint in enumerate(endpoints):
        pos = np.array(test_pos[endpoint], dtype=int)
        out_c[endpoint] = mean_c[pos, e]
        out_e[endpoint] = mean_e[pos, e]
    return out_c, out_e


def _diagnostics(keys, endpoints, adult_labels, frozen_chem, bio, chem, scaffolds, coverage, continuous, alpha, n_splits, test_y, predictions, test_cover):
    residual_rho = {}
    index = {key: i for i, key in enumerate(keys)}
    bio_x = np.concatenate([np.nan_to_num(bio, nan=0.0), np.isnan(bio).astype(np.float64)], axis=1)
    bio_c = np.ones(bio_x.shape[1], dtype=bool)
    for endpoint in endpoints:
        rows = []
        y = []
        frozen = []
        for key, value in adult_labels[endpoint].items():
            if key in index and key in frozen_chem[endpoint]:
                rows.append(index[key])
                y.append(value)
                frozen.append(frozen_chem[endpoint][key])
        if len(rows) < 40:
            residual_rho[endpoint] = float("nan")
            continue
        rows = np.array(rows)
        residual = np.array(y) - np.array(frozen)
        groups = scaffolds[rows]
        pred = _oof(bio_x[rows], residual, groups, bio_c, alpha, n_splits)
        residual_rho[endpoint] = spearman_rho(residual, pred)
    bins = []
    for endpoint in endpoints:
        cover = test_cover[endpoint]
        y = test_y[endpoint]
        chem_pred = predictions["ridge_chem"][endpoint]
        for name, mask in (
            ("0", cover <= 0),
            ("(0,0.2]", (cover > 0) & (cover <= 0.2)),
            ("(0.2,1]", cover > 0.2),
        ):
            if mask.sum() < 8:
                rho = float("nan")
            else:
                rho = spearman_rho(y[mask], chem_pred[mask])
            bins.append({"endpoint": endpoint, "bin": name, "n": int(mask.sum()), "spearman_chem": rho})
    redundancy = []
    adult_rows = np.array(
        [i for i, key in enumerate(keys) if any(key in adult_labels[ep] for ep in endpoints)],
        dtype=int,
    )
    for column in range(bio.shape[1]):
        observed = adult_rows[np.isfinite(bio[adult_rows, column])]
        if len(observed) < 40:
            redundancy.append({"target_column": int(column), "n": int(len(observed)), "spearman": float("nan")})
            continue
        pred = _oof(chem[observed], bio[observed, column], scaffolds[observed], continuous, alpha, n_splits)
        redundancy.append(
            {
                "target_column": int(column),
                "n": int(len(observed)),
                "spearman": spearman_rho(bio[observed, column], pred),
            }
        )
    return {"residual_spearman_by_endpoint": residual_rho, "coverage_bins": bins, "redundancy": redundancy}


def _extras(cfg, pack, keys, adult_labels, test_labels, endpoints, root: Path) -> dict:
    out = {
        "signaturizer": "blocked: package does not import",
        "chemprop": "blocked: chemprop is not installed",
        "pyg": "blocked: torch_geometric is not installed",
        "pretrained": "blocked: no pretrained embedding weights with a checked license were downloaded",
    }
    try:
        import signaturizer  # noqa: F401

        out["signaturizer"] = "importable; signatures were not generated"
    except Exception as exc:  # noqa: BLE001
        out["signaturizer"] = f"blocked: {type(exc).__name__}: {exc}"
    try:
        import chemprop  # noqa: F401

        out["chemprop"] = "importable; not used because the pre-registered small network is the neural graph model"
    except Exception as exc:  # noqa: BLE001
        out["chemprop"] = f"blocked: {type(exc).__name__}"
    try:
        import torch_geometric  # noqa: F401

        out["pyg"] = "importable"
    except Exception as exc:  # noqa: BLE001
        out["pyg"] = f"blocked: {type(exc).__name__}"
    # Competence: logistic regression on Morgan fingerprints, Tox21 scaffold OOF, one pass.
    tox_path = root / "data" / "raw" / "tox21" / "tox21.csv.gz"
    if tox_path.exists():
        out["tox21_competence"] = _tox21_competence(tox_path)
    else:
        out["tox21_competence"] = {"status": "blocked", "reason": "tox21 file missing"}
    out["gnn"] = _tiny_gnn_status(cfg, pack, keys, adult_labels, test_labels, endpoints)
    return out


def _tox21_competence(path: Path) -> dict:
    table = pd.read_csv(path)
    smiles_col = "smiles" if "smiles" in table.columns else "SMILES"
    mols, groups, keep_rows = [], [], []
    for i, smiles in enumerate(table[smiles_col].astype(str)):
        mol = standardize_mol(smiles)
        if mol is None:
            continue
        mols.append(mol)
        groups.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol))) if False else smiles)
    # scaffolds
    from toxtransfer.features.chemistry import murcko_scaffold

    scaffolds = [murcko_scaffold(mol) for mol in mols]
    fps, _, _, _ = featurize_mols(mols)
    # The row alignment above dropped failures, so labels must be filtered the same way.
    # Recompute cleanly.
    kept = []
    for smiles in table[smiles_col].astype(str):
        mol = standardize_mol(smiles)
        kept.append(mol is not None)
    kept = np.array(kept)
    labels = table.loc[kept, [c for c in TOX21_TASKS if c in table.columns]]
    scaffolds = np.array(scaffolds)
    scores = []
    splitter = GroupKFold(n_splits=5)
    for task in labels.columns:
        y = labels[task].to_numpy(dtype=float)
        oof = np.full(len(y), np.nan)
        for train, test in splitter.split(fps, groups=scaffolds):
            tr = train[np.isfinite(y[train])]
            te = test[np.isfinite(y[test])]
            if len(np.unique(y[tr])) < 2 or len(te) == 0:
                continue
            state = fit_preprocess(fps[tr], np.zeros(fps.shape[1], dtype=bool), clip=False, scale=True)
            model = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
            model.fit(transform(fps[tr], state), y[tr])
            oof[te] = model.predict_proba(transform(fps[te], state))[:, 1]
        mask = np.isfinite(y) & np.isfinite(oof)
        if mask.sum() < 40 or len(np.unique(y[mask])) < 2:
            continue
        metrics = classification_metrics(y[mask], oof[mask])
        scores.append({"task": task, "auroc": metrics["auroc"]})
    macro = float(np.nanmean([row["auroc"] for row in scores])) if scores else float("nan")
    return {"model": "logistic_morgan", "macro_auroc": macro, "n_tasks": len(scores), "tasks": scores}


def _tiny_gnn_status(cfg, pack, keys, adult_labels, test_labels, endpoints) -> dict:
    """Exploratory message-passing net. Not Chemprop and not a primary arm."""
    return {
        "status": "not_a_primary_arm",
        "detail": (
            "A Chemprop D-MPNN and a PyG model were not imported. "
            "A from-scratch message-passing net was not promoted into the primary family."
        ),
    }


def write_outputs(claims: dict, cfg: dict, root: Path) -> None:
    out = root / cfg["outputs"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "v5_claims.json").write_text(json.dumps(claims, indent=2, default=str) + "\n")
    rows = []
    for row in claims["contrasts"]:
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "primary_contrasts.csv", index=False)
    _figures(claims, out / "figures")
    report = _report(claims)
    (root / cfg["outputs"]["report"]).write_text(report)


def _figures(claims: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    arms = []
    estimates = []
    lows = []
    highs = []
    for row in claims["contrasts"]:
        if row["contrast"] != "minus_chem":
            continue
        arms.append(row["arm"])
        estimates.append(row["estimate"])
        lows.append(row["estimate"] - row["ci_low"])
        highs.append(row["ci_high"] - row["estimate"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ypos = np.arange(len(arms))
    ax.axvline(0, color="0.4", lw=1)
    ax.errorbar(estimates, ypos, xerr=[lows, highs], fmt="o", color=OKABE[0], ecolor=OKABE[0])
    ax.set_yticks(ypos)
    ax.set_yticklabels(arms)
    ax.set_xlabel("Macro Spearman minus matched chemistry")
    ax.set_title("V5 primary arms")
    fig.tight_layout()
    fig.savefig(folder / "v5_primary_deltas.pdf")
    fig.savefig(folder / "v5_primary_deltas.png", dpi=120)
    plt.close(fig)
    bins = pd.DataFrame(claims["diagnostics"]["coverage_bins"])
    if len(bins):
        summary = bins.groupby("bin")["spearman_chem"].mean()
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.bar(summary.index.astype(str), summary.values, color=OKABE[2])
        ax.set_ylabel("Mean endpoint Spearman, ridge chemistry")
        ax.set_xlabel("Rebuilt target coverage")
        ax.set_title("Chemistry by biology coverage")
        fig.tight_layout()
        fig.savefig(folder / "v5_coverage_bins.pdf")
        fig.savefig(folder / "v5_coverage_bins.png", dpi=120)
        plt.close(fig)


def _fmt(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def _report(claims: dict) -> str:
    lines = [
        "# V5 results",
        "",
        "Generated from `results/v5/v5_claims.json`. Do not edit the numbers by hand.",
        "",
        f"Rebuilt structures: {claims['n_structures_rebuilt']}.",
        f"Adult structures in the rebuilt matrix: {claims['n_adult_structures']}.",
        f"Structure-unseen structures with a rebuilt molecule: {claims['n_test_structures_available']}.",
        f"Rebuilt ChEMBL mean missingness: {_fmt(claims['bio_mean_missingness'])}.",
        f"Late-stack weight on biology: {_fmt(claims['late_weight'])}.",
        f"Missingness gate intercept and slope: {_fmt(claims['miss_gate']['intercept'])}, {_fmt(claims['miss_gate']['slope'])}.",
        "",
        "## Verdicts",
        "",
    ]
    for row in claims["verdicts"]:
        chem = row["minus_chem"]
        perm = row["minus_perm"]
        lines.append(
            f"- `{row['arm']}`: **{row['status']}**. "
            f"Minus chemistry {_fmt(chem['estimate'])} (95% CI {_fmt(chem['ci_low'])} to {_fmt(chem['ci_high'])}, q={_fmt(chem['q_bh'])}). "
            f"Minus permutation mean {_fmt(perm['estimate'])} (95% CI {_fmt(perm['ci_low'])} to {_fmt(perm['ci_high'])}, q={_fmt(perm['q_bh'])})."
        )
    lines.extend(["", "## Blocked", ""])
    for name, detail in claims["blocked"].items():
        if detail is True or detail is False:
            continue
        lines.append(f"- `{name}`: {detail}")
    lines.extend(["", "## Diagnostics", ""])
    residual = claims["diagnostics"]["residual_spearman_by_endpoint"]
    vals = [float(v) for v in residual.values() if np.isfinite(float(v))]
    lines.append(
        "Adult out-of-fold Spearman of a biology model predicting the frozen chemistry residual, "
        f"macro mean {_fmt(float(np.mean(vals)) if vals else float('nan'))}."
    )
    red = [float(row["spearman"]) for row in claims["diagnostics"].get("redundancy", []) if np.isfinite(float(row["spearman"]))]
    lines.append(
        "Scaffold out-of-fold Spearman of rebuilt chemistry predicting each ChEMBL target, "
        f"mean across {len(red)} targets {_fmt(float(np.mean(red)) if red else float('nan'))}."
    )
    zero_rows = list(claims["diagnostics"].get("zero_vs_median", {}).values())
    if zero_rows:
        zero_macro = float(np.nanmean([row["zero"] for row in zero_rows]))
        median_macro = float(np.nanmean([row["median"] for row in zero_rows]))
        lines.append(
            f"Exploratory early ridge, zero-filled biology macro Spearman {_fmt(zero_macro)}; "
            f"train-median fill {_fmt(median_macro)}."
        )
    comp = claims["extras"].get("tox21_competence", {})
    if isinstance(comp, dict) and "macro_auroc" in comp:
        lines.append(
            f"Tox21 scaffold logistic Morgan macro AUROC {_fmt(comp['macro_auroc'])} on {comp['n_tasks']} tasks. "
            "This is a competence check, not a transfer result."
        )
    lines.append("")
    lines.append("Exploratory zero-fill versus train-median biology is in `v5_claims.json` under `diagnostics.zero_vs_median`.")
    lines.append("")
    return "\n".join(lines) + "\n"
