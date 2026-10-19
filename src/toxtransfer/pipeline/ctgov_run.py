"""Adult-trial scaffold CV and pediatric-trial transfer fits.

Hyperparameters used for confirmatory ridge fits are the pre-specified alpha
unless the config sets alpha_mode to inner_groupkfold. Inner selection, when
enabled, sees only the training side of the current split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from toxtransfer.data.ctgov import featurize_compounds, load_ctgov_tables
from toxtransfer.data.synthetic import make_synthetic
from toxtransfer.evaluation.leakage import (
    check_folds,
    check_permutation_control,
    check_preprocess_ignores_test,
    check_unseen_disjoint,
)
from toxtransfer.features.biology import coverage_from_missingness, permute_rows
from toxtransfer.models.adaptation import importance_weights, project_then_coral
from toxtransfer.models.linear import fit_predict_ridge, select_alpha_groupkfold
from toxtransfer.models.trees import fit_predict_rf, fit_predict_xgb
from toxtransfer.splits.scaffold import (
    assert_split_integrity,
    group_holdout_indices,
    group_kfold_indices,
)


def _representation_names(cfg: dict) -> list[str]:
    names = ["chem", "bio", "early", "missingness"]
    if cfg["ablations"].get("early_without_missingness", True):
        names.append("early_imputed_no_indicator")
    for seed in cfg["transfer"]["permutation_seeds"]:
        names.append(f"chem_permuted_bio_seed_{int(seed)}")
    if cfg["transfer"].get("run_random_features", True):
        names.append("chem_random_matched")
    return names


def _matrices(bundle: dict, kind: str, train_idx: np.ndarray, test_idx: np.ndarray):
    chem = bundle["chem"]
    chem_c = bundle["chem_continuous"]
    bio = bundle["bio"]
    bio_c = bundle["bio_continuous"]
    if kind == "chem":
        return chem[train_idx], chem[test_idx], chem_c
    if kind == "bio":
        return bio[train_idx], bio[test_idx], bio_c
    if kind == "early":
        return (
            np.hstack([chem[train_idx], bio[train_idx]]),
            np.hstack([chem[test_idx], bio[test_idx]]),
            np.concatenate([chem_c, bio_c]),
        )
    if kind == "early_imputed_no_indicator":
        values = bundle["bio_values_nan"]
        cont = np.concatenate([chem_c, np.ones(values.shape[1], dtype=bool)])
        return (
            np.hstack([chem[train_idx], values[train_idx]]),
            np.hstack([chem[test_idx], values[test_idx]]),
            cont,
        )
    if kind == "missingness":
        miss = bundle["miss"]
        cont = np.zeros(miss.shape[1], dtype=bool)
        return miss[train_idx], miss[test_idx], cont
    if kind.startswith("chem_permuted_bio_seed_"):
        seed = int(kind.rsplit("_", 1)[-1])
        permuted, _ = permute_rows(bio[train_idx], seed)
        # Test rows keep their own biology. The control destroys the training
        # correspondence; it does not shuffle the evaluation labels.
        return (
            np.hstack([chem[train_idx], permuted]),
            np.hstack([chem[test_idx], bio[test_idx]]),
            np.concatenate([chem_c, bio_c]),
        )
    if kind == "chem_random_matched":
        random_block = bundle["random_features"]
        cont = np.concatenate([chem_c, np.ones(random_block.shape[1], dtype=bool)])
        return (
            np.hstack([chem[train_idx], random_block[train_idx]]),
            np.hstack([chem[test_idx], random_block[test_idx]]),
            cont,
        )
    raise KeyError(kind)


def _alpha_for(cfg: dict, X_train, y_train, groups, continuous) -> float:
    mode = cfg["models"]["ridge"].get("alpha_mode", "prespecified")
    alpha = float(cfg["models"]["ridge"]["alpha"])
    if mode == "prespecified":
        return alpha
    if mode != "inner_groupkfold":
        raise ValueError(f"Unknown alpha_mode {mode}")
    try:
        return select_alpha_groupkfold(
            X_train,
            y_train,
            groups,
            continuous,
            [float(a) for a in cfg["models"]["ridge"]["alphas"]],
            int(cfg["adult_cv"]["inner_splits"]),
        )
    except ValueError:
        return alpha


def _predict(cfg, model_name, X_train, y_train, X_test, continuous, alpha, sample_weight=None):
    if model_name == "ridge":
        return fit_predict_ridge(
            X_train, y_train, X_test, continuous, alpha, sample_weight=sample_weight
        )
    if model_name == "rf":
        spec = cfg["models"]["rf"]
        return fit_predict_rf(
            X_train,
            y_train,
            X_test,
            continuous,
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
        )
    if model_name == "xgb":
        spec = cfg["models"]["xgb"]
        return fit_predict_xgb(
            X_train,
            y_train,
            X_test,
            continuous,
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            learning_rate=float(spec["learning_rate"]),
        )
    raise KeyError(model_name)


def _append(rows, *, dataset, phase, endpoint, model, representation, inchikeys, y, pred, coverage, seen, scaffold_seen, fold):
    pred = np.asarray(pred, dtype=np.float64)
    for i in range(len(inchikeys)):
        rows.append(
            {
                "dataset": dataset,
                "phase": phase,
                "endpoint": endpoint,
                "model": model,
                "representation": representation,
                "inchikey": inchikeys[i],
                "y_true": float(y[i]),
                "y_pred": float(pred[i]),
                "bio_coverage": float(coverage[i]),
                "structure_seen": int(seen[i]),
                "scaffold_seen": int(scaffold_seen[i]),
                "fold": int(fold),
            }
        )


def _endpoint_indices(bundle, endpoint: str, domain: str) -> tuple[np.ndarray, np.ndarray]:
    index = bundle["index"]
    rows = [row for row in bundle["labels"] if row["endpoint"] == endpoint and row["domain"] == domain]
    if not rows:
        return np.array([], dtype=int), np.array([])
    idx = np.array([index[row["inchikey"]] for row in rows], dtype=int)
    y = np.array([row["y"] for row in rows], dtype=np.float64)
    order = np.argsort(bundle["inchikey"][idx], kind="mergesort")
    return idx[order], y[order]


def run_bundle(bundle: dict, cfg: dict, output_dir: Path) -> Path:
    """Fit the configured models and write predictions plus a leakage report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "synthetic" if bundle["source"] == "synthetic" else "ctgov_adult_pediatric_trial"
    rng = np.random.default_rng(101)
    bundle["random_features"] = rng.normal(size=(len(bundle["inchikey"]), bundle["bio"].shape[1]))
    bundle["index"] = {ik: i for i, ik in enumerate(bundle["inchikey"].tolist())}
    coverage = coverage_from_missingness(bundle["miss"])
    adult_ids = set(bundle["adult_ids"])
    adult_scaffolds = set(bundle["scaffold"][bundle["in_adult"]].tolist())
    endpoints = sorted({row["endpoint"] for row in bundle["labels"]})
    representations = _representation_names(cfg)
    n_splits = int(cfg["adult_cv"]["n_splits"])
    rows: list[dict] = []
    leakage = {"folds": [], "permutation": None, "preprocess": None, "unseen": None}
    checked_perm = False
    t0 = time.time()

    tree_reps = set(cfg["models"]["rf"].get("representations", [])) | set(
        cfg["models"]["xgb"].get("representations", [])
    )
    models = ["ridge"]
    if cfg["models"]["rf"].get("enabled", False):
        models.append("rf")
    if cfg["models"]["xgb"].get("enabled", False):
        models.append("xgb")

    for endpoint in endpoints:
        adult_idx, y_adult = _endpoint_indices(bundle, endpoint, "adult")
        ped_idx, y_ped = _endpoint_indices(bundle, endpoint, "pediatric")
        if len(adult_idx) < n_splits or len(ped_idx) == 0:
            continue
        groups = bundle["scaffold"][adult_idx]
        ids = bundle["inchikey"][adult_idx]
        folds = group_kfold_indices(groups, n_splits)
        check_folds(ids, groups, folds)
        leakage["folds"].append({"endpoint": endpoint, "n_folds": len(folds)})
        if not checked_perm:
            train0, test0 = folds[0]
            leakage["permutation"] = check_permutation_control(bundle["bio"][adult_idx[train0]], 101)
            leakage["preprocess"] = check_preprocess_ignores_test(
                bundle["chem"][adult_idx[train0]],
                bundle["chem"][adult_idx[test0]],
                bundle["chem_continuous"],
            )
            checked_perm = True

        ped_ids = bundle["inchikey"][ped_idx]
        unseen_ids = set(ped_ids.tolist()) - adult_ids
        leakage["unseen"] = check_unseen_disjoint(adult_ids, unseen_ids)
        seen_flag = np.array([1 if ik in adult_ids else 0 for ik in ped_ids])
        scaffold_flag = np.array(
            [1 if sc in adult_scaffolds else 0 for sc in bundle["scaffold"][ped_idx]]
        )
        adult_seen = np.zeros(len(adult_idx), dtype=int)
        adult_scaf = np.zeros(len(adult_idx), dtype=int)

        for model_name in models:
            reps = representations if model_name == "ridge" else [r for r in representations if r in tree_reps]
            for kind in reps:
                # Adult scaffold OOF. Trees skip OOF unless configured, to keep the reduced run finite.
                do_oof = model_name == "ridge" or cfg["models"][model_name].get("adult_oof", False)
                if do_oof:
                    for fold_id, (train, test) in enumerate(folds):
                        assert_split_integrity(ids, groups, train, test)
                        X_train, X_test, cont = _matrices(bundle, kind, adult_idx[train], adult_idx[test])
                        alpha = _alpha_for(cfg, X_train, y_adult[train], groups[train], cont)
                        pred_fold = _predict(
                            cfg, model_name, X_train, y_adult[train], X_test, cont, alpha
                        )
                        _append(
                            rows,
                            dataset=dataset_name,
                            phase="adult_oof",
                            endpoint=endpoint,
                            model=model_name,
                            representation=kind,
                            inchikeys=ids[test],
                            y=y_adult[test],
                            pred=pred_fold,
                            coverage=coverage[adult_idx[test]],
                            seen=adult_seen[test],
                            scaffold_seen=adult_scaf[test],
                            fold=fold_id,
                        )
                X_train, X_test, cont = _matrices(bundle, kind, adult_idx, ped_idx)
                alpha = _alpha_for(cfg, X_train, y_adult, groups, cont)
                pred = _predict(cfg, model_name, X_train, y_adult, X_test, cont, alpha)
                _append(
                    rows,
                    dataset=dataset_name,
                    phase="transfer",
                    endpoint=endpoint,
                    model=model_name,
                    representation=kind,
                    inchikeys=ped_ids,
                    y=y_ped,
                    pred=pred,
                    coverage=coverage[ped_idx],
                    seen=seen_flag,
                    scaffold_seen=scaffold_flag,
                    fold=-1,
                )

        if cfg["transfer"].get("run_label_permutation", True):
            y_perm, _ = permute_rows(y_adult.reshape(-1, 1), 101)
            X_train, X_test, cont = _matrices(bundle, "chem", adult_idx, ped_idx)
            alpha = float(cfg["models"]["ridge"]["alpha"])
            pred = fit_predict_ridge(X_train, y_perm.ravel(), X_test, cont, alpha)
            _append(
                rows,
                dataset=dataset_name,
                phase="transfer",
                endpoint=endpoint,
                model="ridge",
                representation="chem_ylabel_permuted_seed_101",
                inchikeys=ped_ids,
                y=y_ped,
                pred=pred,
                coverage=coverage[ped_idx],
                seen=seen_flag,
                scaffold_seen=scaffold_flag,
                fold=-1,
            )

        if cfg["transfer"].get("run_coral", True) or cfg["transfer"].get("run_importance_weights", True):
            from toxtransfer.features.preprocess import fit_preprocess, transform

            state = fit_preprocess(bundle["chem"][adult_idx], bundle["chem_continuous"])
            Z_adult = transform(bundle["chem"][adult_idx], state)
            Z_ped = transform(bundle["chem"][ped_idx], state)
            alpha = float(cfg["models"]["ridge"]["alpha"])
            if cfg["transfer"].get("run_coral", True):
                source_a, target_a = project_then_coral(
                    Z_adult, Z_ped, int(cfg["transfer"]["pca_components"])
                )
                ones = np.ones(source_a.shape[1], dtype=bool)
                pred = fit_predict_ridge(source_a, y_adult, target_a, ones, alpha)
                _append(
                    rows,
                    dataset=dataset_name,
                    phase="transfer",
                    endpoint=endpoint,
                    model="ridge",
                    representation="coral_chem_transductive",
                    inchikeys=ped_ids,
                    y=y_ped,
                    pred=pred,
                    coverage=coverage[ped_idx],
                    seen=seen_flag,
                    scaffold_seen=scaffold_flag,
                    fold=-1,
                )
            if cfg["transfer"].get("run_importance_weights", True):
                weights = importance_weights(Z_adult, Z_ped)
                X_train, X_test, cont = _matrices(bundle, "chem", adult_idx, ped_idx)
                pred = fit_predict_ridge(
                    X_train, y_adult, X_test, cont, alpha, sample_weight=weights
                )
                _append(
                    rows,
                    dataset=dataset_name,
                    phase="transfer",
                    endpoint=endpoint,
                    model="ridge",
                    representation="importance_weighted_chem_transductive",
                    inchikeys=ped_ids,
                    y=y_ped,
                    pred=pred,
                    coverage=coverage[ped_idx],
                    seen=seen_flag,
                    scaffold_seen=scaffold_flag,
                    fold=-1,
                )

        if cfg["transfer"].get("run_target_reference", True):
            unseen_local = np.where(seen_flag == 0)[0]
            if len(unseen_local) >= 8 and len(set(bundle["scaffold"][ped_idx[unseen_local]].tolist())) >= 2:
                rel_train, rel_test = group_holdout_indices(
                    bundle["scaffold"][ped_idx[unseen_local]], test_size=0.5, seed=0
                )
                assert_split_integrity(
                    ped_ids[unseen_local],
                    bundle["scaffold"][ped_idx[unseen_local]],
                    rel_train,
                    rel_test,
                )
                for kind in ("chem", "early"):
                    X_train, X_test, cont = _matrices(
                        bundle, kind, ped_idx[unseen_local][rel_train], ped_idx[unseen_local][rel_test]
                    )
                    alpha = float(cfg["models"]["ridge"]["alpha"])
                    pred = fit_predict_ridge(
                        X_train, y_ped[unseen_local][rel_train], X_test, cont, alpha
                    )
                    _append(
                        rows,
                        dataset=dataset_name,
                        phase="target_reference",
                        endpoint=endpoint,
                        model="ridge",
                        representation=kind,
                        inchikeys=ped_ids[unseen_local][rel_test],
                        y=y_ped[unseen_local][rel_test],
                        pred=pred,
                        coverage=coverage[ped_idx[unseen_local][rel_test]],
                        seen=np.zeros(len(rel_test), dtype=int),
                        scaffold_seen=np.array(
                            [
                                1 if sc in adult_scaffolds else 0
                                for sc in bundle["scaffold"][ped_idx[unseen_local][rel_test]]
                            ]
                        ),
                        fold=0,
                    )

        print(f"finished endpoint {endpoint} in {time.time() - t0:.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    pred_path = output_dir / "predictions" / "predictions.csv.gz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(pred_path, index=False, compression="gzip")
    characterization = _characterize(bundle, cfg)
    characterization["seconds"] = time.time() - t0
    (output_dir / "data_characterization.json").write_text(json.dumps(characterization, indent=2) + "\n")
    (output_dir / "leakage_checks.json").write_text(json.dumps(leakage, indent=2) + "\n")
    return pred_path


def _characterize(bundle: dict, cfg: dict) -> dict:
    adult = bundle["in_adult"]
    ped = bundle["in_pediatric"]
    adult_ids = set(bundle["inchikey"][adult].tolist())
    ped_ids = set(bundle["inchikey"][ped].tolist())
    overlap = adult_ids & ped_ids
    adult_scaf = set(bundle["scaffold"][adult].tolist())
    ped_scaf = set(bundle["scaffold"][ped].tolist())
    coverage = coverage_from_missingness(bundle["miss"])
    endpoints = []
    for endpoint in sorted({row["endpoint"] for row in bundle["labels"]}):
        a = [row for row in bundle["labels"] if row["endpoint"] == endpoint and row["domain"] == "adult"]
        p = [row for row in bundle["labels"] if row["endpoint"] == endpoint and row["domain"] == "pediatric"]
        unseen = [row for row in p if row["inchikey"] not in adult_ids]
        endpoints.append(
            {
                "endpoint": endpoint,
                "n_adult": len(a),
                "n_pediatric": len(p),
                "n_structure_unseen": len(unseen),
                "mean_y_adult": float(np.mean([row["y"] for row in a])) if a else None,
                "mean_y_pediatric_unseen": float(np.mean([row["y"] for row in unseen])) if unseen else None,
            }
        )
    return {
        "source": bundle["source"],
        "n_compounds": int(len(bundle["inchikey"])),
        "n_adult_structures": len(adult_ids),
        "n_pediatric_structures": len(ped_ids),
        "n_structure_overlap": len(overlap),
        "n_structure_unseen": len(ped_ids - adult_ids),
        "n_adult_scaffolds": len(adult_scaf),
        "n_pediatric_scaffolds": len(ped_scaf),
        "n_scaffold_overlap": len(adult_scaf & ped_scaf),
        "bio_n_targets": int(bundle["n_targets"]),
        "bio_mean_missingness": float(bundle["miss"].mean()),
        "bio_frac_any_observed": float((bundle["miss"].mean(axis=1) < 1).mean()),
        "bio_mean_coverage": float(coverage.mean()),
        "n_chem_features": int(bundle["chem"].shape[1]),
        "descriptor_names": list(bundle["descriptor_names"]),
        "smiles_conflicts": int(bundle["smiles_conflicts"]),
        "featurization_failures": int(bundle["featurization_failures"]),
        "logit_max_abs_diff_vs_recompute": float(bundle["logit_max_abs_diff"]),
        "estimate_types": list(bundle["estimate_types"]),
        "min_adult_n": int(cfg["data"]["min_adult_n"]),
        "min_unseen_n": int(cfg["data"]["min_unseen_n"]),
        "endpoints": endpoints,
        "alpha_mode": cfg["models"]["ridge"].get("alpha_mode", "prespecified"),
        "ridge_alpha": float(cfg["models"]["ridge"]["alpha"]),
        "permutation_seeds": [int(s) for s in cfg["transfer"]["permutation_seeds"]],
    }


def _signature(raw: Path) -> list[list]:
    files = sorted(raw.glob("*.csv"))
    return [[path.name, path.stat().st_size] for path in files]


def _bundle_from_features(features: dict, labels: pd.DataFrame, tables: dict) -> dict:
    kept = features["kept_inchikeys"]
    labels = labels[labels["inchikey"].isin(kept)].rename(columns={"outcome": "endpoint"})
    adult_ids = set(labels.loc[labels["domain"] == "adult", "inchikey"])
    ped_ids = set(labels.loc[labels["domain"] == "pediatric", "inchikey"])
    features["in_adult"] = np.array([ik in adult_ids for ik in features["inchikey"]])
    features["in_pediatric"] = np.array([ik in ped_ids for ik in features["inchikey"]])
    features["adult_ids"] = adult_ids
    features["pediatric_ids"] = ped_ids
    features["labels"] = labels.to_dict("records")
    features["smiles_conflicts"] = tables["smiles_conflicts"]
    features["logit_max_abs_diff"] = tables["logit_max_abs_diff"]
    features["estimate_types"] = tables["estimate_types"]
    features["source"] = "ctgov_handoff"
    return features


def _save_feature_cache(path: Path, bundle: dict, signature: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inchikey=np.asarray(bundle["inchikey"], dtype="U64"),
        scaffold=np.asarray(bundle["scaffold"], dtype="U2048"),
        chem=bundle["chem"],
        chem_continuous=bundle["chem_continuous"],
        bio=bundle["bio"],
        bio_values_nan=bundle["bio_values_nan"],
        miss=bundle["miss"],
        bio_continuous=bundle["bio_continuous"],
        in_adult=bundle["in_adult"],
        in_pediatric=bundle["in_pediatric"],
    )
    meta = {
        "signature": signature,
        "n_targets": bundle["n_targets"],
        "descriptor_names": bundle["descriptor_names"],
        "featurization_failures": bundle["featurization_failures"],
        "smiles_conflicts": bundle["smiles_conflicts"],
        "logit_max_abs_diff": bundle["logit_max_abs_diff"],
        "estimate_types": bundle["estimate_types"],
        "adult_ids": sorted(bundle["adult_ids"]),
        "pediatric_ids": sorted(bundle["pediatric_ids"]),
        "labels": bundle["labels"],
    }
    path.with_suffix(".json").write_text(json.dumps(meta) + "\n")


def _load_feature_cache(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text())
    arrays = np.load(path, allow_pickle=False)
    return {
        "inchikey": arrays["inchikey"],
        "scaffold": arrays["scaffold"],
        "chem": arrays["chem"],
        "chem_continuous": arrays["chem_continuous"],
        "bio": arrays["bio"],
        "bio_values_nan": arrays["bio_values_nan"],
        "miss": arrays["miss"],
        "bio_continuous": arrays["bio_continuous"],
        "in_adult": arrays["in_adult"],
        "in_pediatric": arrays["in_pediatric"],
        "adult_ids": set(meta["adult_ids"]),
        "pediatric_ids": set(meta["pediatric_ids"]),
        "labels": meta["labels"],
        "n_targets": meta["n_targets"],
        "descriptor_names": meta["descriptor_names"],
        "featurization_failures": meta["featurization_failures"],
        "smiles_conflicts": meta["smiles_conflicts"],
        "logit_max_abs_diff": meta["logit_max_abs_diff"],
        "estimate_types": meta["estimate_types"],
        "source": "ctgov_handoff",
        "kept_inchikeys": set(arrays["inchikey"].tolist()),
    }


def prepare_bundle(cfg: dict, root: Path) -> dict:
    """Load synthetic data or the gitignored ClinicalTrials.gov handoff."""
    if cfg["data"].get("dataset") == "synthetic":
        bundle = make_synthetic(int(cfg.get("seed", 0)))
        bundle["adult_ids"] = set(bundle["inchikey"][bundle["in_adult"]].tolist())
        bundle["pediatric_ids"] = set(bundle["inchikey"][bundle["in_pediatric"]].tolist())
        return bundle
    raw = root / cfg["data"]["ctgov_dir"]
    signature = _signature(raw)
    cache = root / cfg["data"]["interim_dir"] / "ctgov_features.npz"
    if cache.exists() and cache.with_suffix(".json").exists():
        meta = json.loads(cache.with_suffix(".json").read_text())
        if meta.get("signature") == signature:
            print("loading cached chemistry features", flush=True)
            return _load_feature_cache(cache)
    tables = load_ctgov_tables(raw)
    features = featurize_compounds(tables["compounds"], tables["tgt_cols"], tables["miss_cols"])
    bundle = _bundle_from_features(features, tables["labels"], tables)
    _save_feature_cache(cache, bundle, signature)
    return bundle
