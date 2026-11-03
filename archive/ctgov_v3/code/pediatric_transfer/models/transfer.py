"""Adult→pediatric zero-shot transfer. Never fit on pediatric labels for primary."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import joblib

from ..utils.config import SEEDS, MIN_ADULT_N, MIN_PED_N, GLOBAL_SEED
from ..utils.io_utils import atomic_write_df, atomic_write_json
from ..utils.logging_utils import log
from ..utils.paths import MODELS, PREDICTIONS, RESULTS

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def murcko_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "NA"
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc is not None else "NA"
    except Exception:
        return "NA"


def endpoint_feasibility(adult: pd.DataFrame, ped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = sorted(set(adult["outcome"]).union(set(ped["outcome"])))
    for o in outcomes:
        a = adult[adult.outcome == o]
        p = ped[ped.outcome == o]
        ok = len(a) >= MIN_ADULT_N and len(p) >= MIN_PED_N
        rows.append({
            "outcome": o,
            "n_adult": len(a),
            "n_pediatric": len(p),
            "n_adult_structures": a["inchikey"].nunique() if len(a) else 0,
            "n_ped_structures": p["inchikey"].nunique() if len(p) else 0,
            "eligible": ok,
            "reason": "ok" if ok else f"need adult>={MIN_ADULT_N} ped>={MIN_PED_N}",
        })
    df = pd.DataFrame(rows)
    atomic_write_df(RESULTS / "endpoint_feasibility.csv", df)
    return df


def permute_bio(bio: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(bio))
    rng.shuffle(idx)
    out = bio.copy()
    out.iloc[:, :] = bio.values[idx]
    out.index = bio.index
    return out


def make_xy(labels: pd.DataFrame, feat: pd.DataFrame, outcome: str) -> Tuple[pd.DataFrame, np.ndarray, pd.Index]:
    sub = labels[labels.outcome == outcome].copy()
    common = sub.set_index("inchikey").join(feat, how="inner")
    y = common["logit_incidence"].values.astype(float)
    X = common[feat.columns]
    return X, y, common.index


def adult_tune_xgb(X, y, groups, seed=GLOBAL_SEED) -> dict:
    """Adult-only scaffold/grouped CV; freeze HPs before pediatric eval."""
    if not HAS_XGB or len(X) < 8:
        return {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.9,
                "colsample_bytree": 0.8, "reg_lambda": 1.0, "min_child_weight": 1,
                "random_state": seed, "n_jobs": 2, "objective": "reg:squarederror"}
    grid = [
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 100, "reg_lambda": 1.0},
        {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 150, "reg_lambda": 1.0},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "reg_lambda": 2.0},
    ]
    n_splits = min(3, max(2, len(np.unique(groups))))
    if len(np.unique(groups)) < 2:
        return {**grid[1], "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 1,
                "random_state": seed, "n_jobs": 2, "objective": "reg:squarederror"}
    best, best_score = None, -np.inf
    for g in grid:
        scores = []
        gkf = GroupKFold(n_splits=n_splits)
        for tr, va in gkf.split(X, y, groups):
            model = XGBRegressor(**g, subsample=0.9, colsample_bytree=0.8, min_child_weight=1,
                                 random_state=seed, n_jobs=2, objective="reg:squarederror")
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
            if len(pred) >= 3 and np.std(pred) > 0 and np.std(y[va]) > 0:
                sp = spearmanr(y[va], pred).correlation
                scores.append(sp if sp == sp else 0)
            else:
                scores.append(-mean_squared_error(y[va], pred))
        m = float(np.mean(scores)) if scores else -np.inf
        if m > best_score:
            best_score = m
            best = g
    return {**(best or grid[1]), "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 1,
            "random_state": seed, "n_jobs": 2, "objective": "reg:squarederror"}


def fit_predict(X_tr, y_tr, X_te, model_name: str, hp: dict):
    if model_name == "xgb" and HAS_XGB:
        model = XGBRegressor(**hp)
        model.fit(X_tr, y_tr)
        return model, model.predict(X_te)
    if model_name == "rf":
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=GLOBAL_SEED, n_jobs=2)
        model.fit(X_tr, y_tr)
        return model, model.predict(X_te)
    # elasticnet / ridge
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)
    Xt = scaler.transform(X_te)
    if model_name == "elasticnet":
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=GLOBAL_SEED, max_iter=5000)
    else:
        model = Ridge(alpha=1.0, random_state=GLOBAL_SEED)
    model.fit(Xs, y_tr)
    return (model, scaler), model.predict(Xt)


def eval_regression(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"spearman": np.nan, "rmse": np.nan, "mae": np.nan, "n": len(y_true)}
    sp = spearmanr(y_true, y_pred).correlation
    return {
        "spearman": float(sp) if sp == sp else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def run_transfer_for_endpoint(
    outcome: str,
    adult_lab: pd.DataFrame,
    ped_lab: pd.DataFrame,
    chem: pd.DataFrame,
    bio: pd.DataFrame,
    smiles_map: Dict[str, str],
    models: List[str] = None,
) -> pd.DataFrame:
    models = models or (["xgb", "rf", "ridge"] if HAS_XGB else ["rf", "ridge"])
    rows = []
    # feature variants
    chem_a = chem
    bio_a = bio.reindex(chem.index).fillna(0)
    # align
    adult_idx = set(adult_lab[adult_lab.outcome == outcome]["inchikey"])
    # permute bio among adult compounds only then apply same row order trick for all
    variants = {"chem": chem_a}
    # bio_only
    variants["bio_only"] = bio_a
    # chem_true_bio
    common_cols_ok = True
    chem_true = chem_a.join(bio_a, how="left", rsuffix="_bio").fillna(0)
    variants["chem_true_bio"] = chem_true
    for seed in SEEDS:
        # Row-permute bio among adult compounds
        adult_bio = bio_a.loc[bio_a.index.isin(adult_idx)]
        if len(adult_bio) == 0:
            continue
        perm_adult = permute_bio(adult_bio, seed)
        # For pediatric, also permute? Spec: row-permute bio among adult compounds.
        # For test set, use true bio for ped when evaluating chem_permuted trained model?
        # Standard: shuffle bio features across adult training rows; pediatric keeps its bio
        # but the multimodal association is destroyed in training.
        bio_perm_full = bio_a.copy()
        bio_perm_full.loc[perm_adult.index] = perm_adult.values
        variants[f"chem_permuted_bio_seed_{seed}"] = chem_a.join(bio_perm_full, how="left", rsuffix="_bio").fillna(0)

    # scaffolds for groups
    def groups_for(ids):
        return np.array([murcko_scaffold(smiles_map.get(i, "")) for i in ids])

    for vname, feat in variants.items():
        Xa, ya, ida = make_xy(adult_lab, feat, outcome)
        Xp, yp, idp = make_xy(ped_lab, feat, outcome)
        if len(Xa) < MIN_ADULT_N or len(Xp) < MIN_PED_N:
            continue
        # drop constant cols on adult
        keep = Xa.columns[Xa.nunique() > 1]
        Xa, Xp = Xa[keep], Xp[keep]
        g = groups_for(ida)
        hp = adult_tune_xgb(Xa, ya, g) if "xgb" in models else {}
        atomic_write_json(MODELS / f"hp_{outcome}_{vname}.json", hp)
        for mname in models:
            try:
                model, pred = fit_predict(Xa, ya, Xp, mname, hp if mname == "xgb" else {})
                metrics = eval_regression(yp, pred)
                # save predictions
                pred_df = pd.DataFrame({
                    "inchikey": list(idp),
                    "y_true": yp,
                    "y_pred": pred,
                    "outcome": outcome,
                    "representation": vname,
                    "model": mname,
                })
                atomic_write_df(PREDICTIONS / f"pred_{outcome}_{vname}_{mname}.csv", pred_df)
                joblib.dump(model, MODELS / f"model_{outcome}_{vname}_{mname}.joblib")
                rows.append({
                    "outcome": outcome, "representation": vname, "model": mname,
                    **metrics, "n_adult_train": len(Xa),
                })
                log(f"transfer {outcome} {vname} {mname} spearman={metrics['spearman']:.4f} n={metrics['n']}")
            except Exception as e:
                log(f"transfer fail {outcome} {vname} {mname}: {e}")
                rows.append({
                    "outcome": outcome, "representation": vname, "model": mname,
                    "spearman": np.nan, "rmse": np.nan, "mae": np.nan, "n": 0,
                    "n_adult_train": len(Xa), "error": str(e),
                })
    return pd.DataFrame(rows)


def seen_unseen_split(adult_lab, ped_lab, smiles_map) -> pd.DataFrame:
    adult_ik = set(adult_lab["inchikey"])
    adult_sc = {murcko_scaffold(smiles_map.get(i, "")) for i in adult_ik}
    rows = []
    for ik in ped_lab["inchikey"].unique():
        sc = murcko_scaffold(smiles_map.get(ik, ""))
        rows.append({
            "inchikey": ik,
            "structure_seen_in_adult": ik in adult_ik,
            "scaffold_seen_in_adult": sc in adult_sc and sc != "NA",
            "scaffold": sc,
        })
    df = pd.DataFrame(rows)
    atomic_write_df(RESULTS / "seen_unseen_structures.csv", df)
    return df


def eval_seen_unseen(pred_df: pd.DataFrame, seen_df: pd.DataFrame) -> pd.DataFrame:
    m = pred_df.merge(seen_df, on="inchikey", how="left")
    rows = []
    for rep in m["representation"].dropna().unique():
        for model in m["model"].dropna().unique():
            sub = m[(m.representation == rep) & (m.model == model)]
            for label, mask in [
                ("structure_seen", sub["structure_seen_in_adult"] == True),
                ("structure_unseen", sub["structure_seen_in_adult"] == False),
                ("scaffold_seen", sub["scaffold_seen_in_adult"] == True),
                ("scaffold_unseen", sub["scaffold_seen_in_adult"] == False),
            ]:
                ss = sub[mask]
                met = eval_regression(ss["y_true"].values, ss["y_pred"].values) if len(ss) else {"spearman": np.nan, "rmse": np.nan, "mae": np.nan, "n": 0}
                rows.append({"representation": rep, "model": model, "slice": label, "outcome": "ALL", **met})
            # also per-outcome for primary model slices
            if "outcome" in sub.columns:
                for outcome, so in sub.groupby("outcome"):
                    for label, mask in [
                        ("structure_seen", so["structure_seen_in_adult"] == True),
                        ("structure_unseen", so["structure_seen_in_adult"] == False),
                    ]:
                        ss = so[mask]
                        met = eval_regression(ss["y_true"].values, ss["y_pred"].values) if len(ss) else {"spearman": np.nan, "rmse": np.nan, "mae": np.nan, "n": 0}
                        rows.append({"representation": rep, "model": model, "slice": label, "outcome": outcome, **met})
    out = pd.DataFrame(rows)
    atomic_write_df(RESULTS / "seen_unseen_performance.csv", out)
    return out
