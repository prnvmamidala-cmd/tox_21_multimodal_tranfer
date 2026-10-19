#!/usr/bin/env python3
"""Finalize mapping from cache; run aggregate → features → transfer → bootstrap → reports."""
from __future__ import annotations
import sys
from pathlib import Path
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from run_pipeline import *
import pandas as pd
import numpy as np
import json

def finalize_mapping_from_cache():
    cache = load_json(CACHE / "drug_name_cache.json", {})
    rows = list(cache.values())
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty drug cache")
    # ensure columns
    for c in ["drug_name_norm","drug_name_raw","status","source","confidence","canonical_smiles","inchikey"]:
        if c not in df.columns:
            df[c] = None
    atomic_write_df(PROCESSED / "drug_name_mapping.csv", df)
    atomic_write_df(PROCESSED / "unresolved_drug_names.csv", df[df.status == "unresolved"])
    atomic_write_df(PROCESSED / "excluded_combination_products.csv",
                    df[df.status.isin(["excluded_control_or_nondrug","excluded_mixture_or_invalid"])])
    ok = df[df.status == "ok"].copy()
    if len(ok):
        aliases = ok.groupby("inchikey").agg(
            canonical_smiles=("canonical_smiles", "first"),
            names=("drug_name_norm", lambda s: "|".join(sorted(set(map(str,s))))),
            sources=("source", lambda s: "|".join(sorted(set(map(str,s))))),
            mean_confidence=("confidence", "mean"),
        ).reset_index()
        atomic_write_df(PROCESSED / "structure_aliases.csv", aliases)
    log(f"Finalized mapping from cache: n={len(df)} ok={len(ok)} unresolved={(df.status=='unresolved').sum()}")
    note = (
        f"Mapping stopped early after sufficient endpoint coverage.\n"
        f"Cache entries: {len(df)}; ok: {len(ok)}; unique inchikeys: {ok['inchikey'].nunique() if len(ok) else 0}.\n"
        f"Remaining prioritized names not yet queried remain absent from cache / deferred.\n"
        f"See deferred_drug_names.csv and excluded_nondrug_labels.csv.\n"
    )
    atomic_write_text(PROCESSED / "MAPPING_EARLY_STOP.txt", note)
    return df

def main():
    log("=== RESUME FROM FEATURES ===")
    manifest = load_json(MANIFEST, {}) or {}
    manifest.setdefault("stages", {})
    manifest.setdefault("completed", [])
    manifest.setdefault("failed", [])
    manifest["package_versions"] = package_versions()
    manifest["seeds"] = SEEDS
    manifest["bootstrap_n"] = BOOTSTRAP_N

    ev = ensure_eudra_status()
    manifest["stages"]["eudra"] = ev

    arm = pd.read_csv(PROCESSED / "ctgov_arm_level.csv")
    counts = load_json(PROCESSED / "ctgov_trial_counts.json", {})
    manifest["stages"]["ctgov"] = {**counts, "arm_rows": len(arm)}

    from pediatric_transfer.faers.fetch import _load_all
    ped_rep = _load_all(RAW / "faers" / "pediatric")
    ad_rep = _load_all(RAW / "faers" / "adult")
    ped_pairs = reports_to_pairs(ped_rep, "pediatric")
    ad_pairs = reports_to_pairs(ad_rep, "adult")
    ror = compute_ror(ped_pairs, ad_pairs)
    manifest["stages"]["faers"] = {"ped_reports": len(ped_rep), "adult_reports": len(ad_rep), "ror_rows": len(ror)}

    mapping = finalize_mapping_from_cache()
    manifest["stages"]["mapping"] = {
        "n": len(mapping), "ok": int((mapping.status=="ok").sum()),
        "unresolved": int((mapping.status=="unresolved").sum()),
        "early_stop": True,
    }
    ok_map = mapping[mapping.status=="ok"][["drug_name_norm","inchikey","canonical_smiles","source","confidence"]].drop_duplicates("drug_name_norm")

    agg = split_and_save(arm, ok_map)
    adult_lab = agg.get("adult", pd.DataFrame())
    ped_lab = agg.get("pediatric", pd.DataFrame())
    manifest["stages"]["aggregate"] = {
        "adult_rows": len(adult_lab), "ped_rows": len(ped_lab),
        "adult_structures": int(adult_lab.inchikey.nunique()) if len(adult_lab) else 0,
        "ped_structures": int(ped_lab.inchikey.nunique()) if len(ped_lab) else 0,
    }
    atomic_write_json(MANIFEST, manifest)

    # long table + overlaps
    long_rows = []
    for bucket, lab in [("adult", adult_lab), ("pediatric", ped_lab)]:
        if lab is None or lab.empty: continue
        for _, r in lab.iterrows():
            long_rows.append({
                "inchikey": r["inchikey"], "outcome": r["outcome"], "age_bucket": bucket,
                "estimate_type": "clinical_trial_smoothed_incidence",
                "value": r["logit_incidence"], "incidence_smooth": r["incidence_smooth"],
                "num_affected": r["num_affected"], "num_at_risk": r["num_at_risk"],
                "n_trials": r.get("n_trials", np.nan), "source": "ClinicalTrials.gov",
            })
    if not ror.empty and len(ok_map):
        rr = ror.merge(ok_map[["drug_name_norm","inchikey"]], on="drug_name_norm", how="inner")
        for _, r in rr.iterrows():
            long_rows.append({
                "inchikey": r["inchikey"], "outcome": r["outcome"], "age_bucket": "ped_vs_adult_ror",
                "estimate_type": "faers_ror_not_incidence", "value": r["logROR"],
                "incidence_smooth": np.nan, "num_affected": r["ped_a"], "num_at_risk": r["ped_drug_reports"],
                "n_trials": np.nan, "source": "FAERS_openFDA",
            })
    atomic_write_df(PROCESSED / "harmonized_long_table.csv", pd.DataFrame(long_rows))
    if len(adult_lab) and len(ped_lab):
        a_ik, p_ik = set(adult_lab.inchikey), set(ped_lab.inchikey)
        overlap_j = {
            "adult_structures": len(a_ik), "pediatric_structures": len(p_ik),
            "structure_overlap": len(a_ik & p_ik),
            "adult_structure_outcome": len(adult_lab), "ped_structure_outcome": len(ped_lab),
        }
        if not ror.empty:
            f_ik = set(ror.merge(ok_map, on="drug_name_norm")["inchikey"])
            overlap_j.update({
                "faers_mapped_structures": len(f_ik),
                "faers_adult_overlap": len(f_ik & a_ik),
                "faers_ped_overlap": len(f_ik & p_ik),
            })
        atomic_write_json(RESULTS / "source_overlaps.json", overlap_j)
        manifest["stages"]["overlaps"] = overlap_j
    atomic_write_json(MANIFEST, manifest)

    structs = ok_map[["inchikey","canonical_smiles"]].drop_duplicates("inchikey")
    smiles_map = dict(zip(structs.inchikey, structs.canonical_smiles))
    need = set(adult_lab.inchikey)|set(ped_lab.inchikey) if len(adult_lab) or len(ped_lab) else set(structs.inchikey)
    structs = structs[structs.inchikey.isin(need)]
    log(f"Building chem features for {len(structs)} structures")
    chem_raw, meta = build_chem_matrix(structs)
    adult_iks = list(adult_lab.inchikey.unique()) if len(adult_lab) else list(structs.inchikey)
    params = fit_preprocess_adult(chem_raw, adult_iks)
    chem_x = transform_chem(chem_raw, params)
    atomic_write_df(FEATURES / "chem_processed.csv", chem_x.reset_index())
    log(f"Building chembl_target_profile for {len(structs)} structures")
    bio = build_chembl_target_profiles(list(structs.inchikey), adult_iks)
    manifest["stages"]["features"] = {"chem": list(chem_x.shape), "bio": list(bio.shape)}
    atomic_write_json(MANIFEST, manifest)

    feas = endpoint_feasibility(adult_lab, ped_lab)
    eligible = feas[feas.eligible]["outcome"].tolist()
    log(f"Eligible endpoints ({len(eligible)}): {eligible}")
    all_metrics = []
    for o in eligible:
        log(f"=== Transfer endpoint {o} ===")
        res = run_transfer_for_endpoint(o, adult_lab, ped_lab, chem_x, bio, smiles_map)
        all_metrics.append(res)
        atomic_write_df(RESULTS / "transfer_metrics_partial.csv", pd.concat(all_metrics, ignore_index=True))
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    atomic_write_df(RESULTS / "transfer_metrics.csv", metrics)

    seen = seen_unseen_split(adult_lab, ped_lab, smiles_map)
    preds = [pd.read_csv(p) for p in PREDICTIONS.glob("pred_*.csv")]
    if preds:
        eval_seen_unseen(pd.concat(preds, ignore_index=True), seen)
    manifest["stages"]["transfer"] = {"eligible": eligible, "n_metric_rows": len(metrics)}
    atomic_write_json(MANIFEST, manifest)

    log("Bootstrap comparisons...")
    boot = run_all_comparisons(PREDICTIONS, eligible, model="xgb")
    if boot.empty:
        boot = run_all_comparisons(PREDICTIONS, eligible, model="ridge")
    manifest["stages"]["bootstrap"] = {"rows": len(boot)}

    ped_pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    fv = faers_validation(ped_pred, ror, mapping)
    manifest["stages"]["faers_validation"] = {"rows": len(fv)}

    write_reports(manifest, metrics, boot, adult_lab, ped_lab, ev)
    manifest["ended_at"] = now_pt()
    hashes = {}
    for p in list(RESULTS.glob("*")) + list(REPORTS.glob("*")) + [PROCESSED/"harmonized_long_table.csv"]:
        if Path(p).is_file() and Path(p).stat().st_size < 80_000_000:
            try: hashes[str(p)] = sha256_file(Path(p))
            except Exception: pass
    manifest["artifact_sha256"] = hashes
    manifest["completed"] = list(dict.fromkeys(manifest.get("completed", []) + [
        "ctgov","faers","mapping","aggregate","features","transfer","bootstrap","faers_validation","reports"]))
    atomic_write_json(MANIFEST, manifest)
    log("=== RESUME FROM FEATURES COMPLETE ===")

if __name__ == "__main__":
    main()
