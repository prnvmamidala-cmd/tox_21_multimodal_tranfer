#!/usr/bin/env python3
"""Resume pipeline from mapping onward when CT.gov+FAERS artifacts exist."""
from __future__ import annotations
import sys
from pathlib import Path
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from run_pipeline import *
import pandas as pd
import numpy as np

def main():
    log("=== RESUME FROM MAPPING ===")
    manifest = load_json(MANIFEST, {}) or {
        "started_at": now_pt(), "seeds": SEEDS, "global_seed": GLOBAL_SEED,
        "bootstrap_n": BOOTSTRAP_N, "package_versions": package_versions(),
        "api": {"ctgov": CTGOV_BASE, "openfda": OPENFDA_BASE, "chembl": CHEMBL_BASE},
        "outcomes": OUTCOMES, "stages": {}, "completed": [], "failed": [],
    }
    ev = ensure_eudra_status()
    manifest["stages"]["eudra"] = ev

    arm = pd.read_csv(PROCESSED / "ctgov_arm_level.csv")
    counts = load_json(PROCESSED / "ctgov_trial_counts.json", {})
    manifest["stages"]["ctgov_parse"] = {**counts, "arm_rows": len(arm), "resumed": True}
    log(f"Loaded arm-level rows={len(arm)}")

    # FAERS
    from pediatric_transfer.faers.fetch import _load_all
    ped_rep = _load_all(RAW / "faers" / "pediatric")
    ad_rep = _load_all(RAW / "faers" / "adult")
    ped_pairs = reports_to_pairs(ped_rep, "pediatric")
    ad_pairs = reports_to_pairs(ad_rep, "adult")
    atomic_write_df(PROCESSED / "faers_ped_pairs.csv", ped_pairs)
    atomic_write_df(PROCESSED / "faers_adult_pairs.csv", ad_pairs)
    ror = compute_ror(ped_pairs, ad_pairs)
    manifest["stages"]["faers"] = {"ped_reports": len(ped_rep), "adult_reports": len(ad_rep), "ror_rows": len(ror)}
    log(f"FAERS ROR rows={len(ror)}")

    # Prioritized mapping
    freq_adult = arm[arm.age_bucket == "adult"].groupby("drug_name_norm").size().sort_values(ascending=False)
    ped_names = set(arm.loc[arm.age_bucket == "pediatric", "drug_name_norm"])
    adult_names = set(arm.loc[arm.age_bucket == "adult", "drug_name_norm"])
    overlap = ped_names & adult_names
    raw_map = arm.drop_duplicates("drug_name_norm").set_index("drug_name_norm")["drug_name_raw"].to_dict()
    ordered = list(sorted(overlap)) + list(sorted(ped_names - overlap))
    for n in freq_adult.index:
        if n not in ped_names:
            ordered.append(n)
    MAX_MAP = 6000
    deferred = ordered[MAX_MAP:]
    ordered = ordered[:MAX_MAP]
    if deferred:
        pd.DataFrame({"drug_name_norm": deferred, "reason": "deferred_overnight_cap"}).to_csv(
            PROCESSED / "deferred_drug_names.csv", index=False)
        log(f"Deferred {len(deferred)} names (cap {MAX_MAP})")
    
    def looks_like_drug(n: str) -> bool:
        n = (n or "").lower()
        bad = ["dose ", "after dose", "informed consent", "post infant", "mdv:", "sds:",
               "arm ", "group ", "cohort ", "week ", "cycle ", "placebo", "vaccine schedule"]
        if any(b in n for b in bad):
            return False
        if len(n) > 80:
            return False
        if n.count(":") >= 2:
            return False
        return True
    skipped_garbage = [n for n in ordered if not looks_like_drug(n)]
    ordered = [n for n in ordered if looks_like_drug(n)]
    if skipped_garbage:
        pd.DataFrame({"drug_name_norm": skipped_garbage, "reason": "non_drug_arm_label"}).to_csv(
            PROCESSED / "excluded_nondrug_labels.csv", index=False)
        log(f"Excluded {len(skipped_garbage)} non-drug arm labels before mapping")
    names = [(n, raw_map.get(n, n)) for n in ordered]
    if not ror.empty:
        for d in ror["drug_name_norm"].unique()[:300]:
            names.append((d, d))
    uniq = {}
    for n, r in names:
        uniq[n] = r
    names = list(uniq.items())
    log(f"Mapping {len(names)} unique drug names (prioritized)")
    mapping = map_drug_names(names, checkpoint_every=25)
    manifest["stages"]["mapping"] = {
        "n": len(mapping), "ok": int((mapping.status == "ok").sum()),
        "unresolved": int((mapping.status == "unresolved").sum()),
        "deferred": len(deferred),
    }
    manifest["completed"].append("mapping")
    atomic_write_json(MANIFEST, manifest)

    ok_map = mapping[mapping.status == "ok"][["drug_name_norm", "inchikey", "canonical_smiles", "source", "confidence"]].drop_duplicates("drug_name_norm")

    agg = split_and_save(arm, ok_map)
    adult_lab = agg.get("adult", pd.DataFrame())
    ped_lab = agg.get("pediatric", pd.DataFrame())
    manifest["stages"]["aggregate"] = {
        "adult_rows": len(adult_lab), "ped_rows": len(ped_lab),
        "adult_structures": int(adult_lab["inchikey"].nunique()) if len(adult_lab) else 0,
        "ped_structures": int(ped_lab["inchikey"].nunique()) if len(ped_lab) else 0,
    }
    atomic_write_json(MANIFEST, manifest)

    # harmonized long
    long_rows = []
    for bucket, lab in [("adult", adult_lab), ("pediatric", ped_lab)]:
        if lab is None or lab.empty:
            continue
        for _, r in lab.iterrows():
            long_rows.append({
                "inchikey": r["inchikey"], "outcome": r["outcome"], "age_bucket": bucket,
                "estimate_type": "clinical_trial_smoothed_incidence",
                "value": r["logit_incidence"], "incidence_smooth": r["incidence_smooth"],
                "num_affected": r["num_affected"], "num_at_risk": r["num_at_risk"],
                "n_trials": r.get("n_trials", np.nan), "source": "ClinicalTrials.gov",
            })
    if not ror.empty and len(ok_map):
        rr = ror.merge(ok_map[["drug_name_norm", "inchikey"]], on="drug_name_norm", how="inner")
        for _, r in rr.iterrows():
            long_rows.append({
                "inchikey": r["inchikey"], "outcome": r["outcome"], "age_bucket": "ped_vs_adult_ror",
                "estimate_type": "faers_ror_not_incidence", "value": r["logROR"],
                "incidence_smooth": np.nan, "num_affected": r["ped_a"], "num_at_risk": r["ped_drug_reports"],
                "n_trials": np.nan, "source": "FAERS_openFDA",
            })
    long_df = pd.DataFrame(long_rows)
    atomic_write_df(PROCESSED / "harmonized_long_table.csv", long_df)
    if len(adult_lab) and len(ped_lab):
        a_ik = set(adult_lab.inchikey); p_ik = set(ped_lab.inchikey)
        overlap_j = {
            "adult_structures": len(a_ik), "pediatric_structures": len(p_ik),
            "structure_overlap": len(a_ik & p_ik),
            "adult_structure_outcome": len(adult_lab), "ped_structure_outcome": len(ped_lab),
        }
        if not ror.empty:
            f_ik = set(ror.merge(ok_map, on="drug_name_norm")["inchikey"])
            overlap_j["faers_mapped_structures"] = len(f_ik)
            overlap_j["faers_adult_overlap"] = len(f_ik & a_ik)
            overlap_j["faers_ped_overlap"] = len(f_ik & p_ik)
        atomic_write_json(RESULTS / "source_overlaps.json", overlap_j)
        manifest["stages"]["overlaps"] = overlap_j
    atomic_write_json(MANIFEST, manifest)

    # Features
    structs = ok_map[["inchikey", "canonical_smiles"]].drop_duplicates("inchikey")
    smiles_map = dict(zip(structs.inchikey, structs.canonical_smiles))
    need = set()
    if len(adult_lab): need |= set(adult_lab.inchikey)
    if len(ped_lab): need |= set(ped_lab.inchikey)
    structs = structs[structs.inchikey.isin(need)] if need else structs
    log(f"Building chem features for {len(structs)} structures")
    chem_raw, meta = build_chem_matrix(structs)
    adult_iks = list(adult_lab["inchikey"].unique()) if len(adult_lab) else list(structs.inchikey)
    params = fit_preprocess_adult(chem_raw, adult_iks)
    chem_x = transform_chem(chem_raw, params)
    atomic_write_df(FEATURES / "chem_processed.csv", chem_x.reset_index())
    log(f"Building chembl_target_profile for {len(structs)} structures")
    bio = build_chembl_target_profiles(list(structs.inchikey), adult_iks)
    manifest["stages"]["features"] = {"chem": list(chem_x.shape), "bio": list(bio.shape)}
    atomic_write_json(MANIFEST, manifest)

    # Transfer
    feas = endpoint_feasibility(adult_lab, ped_lab) if len(adult_lab) and len(ped_lab) else pd.DataFrame()
    eligible = feas[feas.eligible]["outcome"].tolist() if len(feas) else []
    log(f"Eligible endpoints: {eligible}")
    all_metrics = []
    for o in eligible:
        res = run_transfer_for_endpoint(o, adult_lab, ped_lab, chem_x, bio, smiles_map)
        all_metrics.append(res)
        atomic_write_df(RESULTS / "transfer_metrics_partial.csv", pd.concat(all_metrics, ignore_index=True))
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    atomic_write_df(RESULTS / "transfer_metrics.csv", metrics)
    if len(adult_lab) and len(ped_lab):
        seen = seen_unseen_split(adult_lab, ped_lab, smiles_map)
        preds = [pd.read_csv(p) for p in PREDICTIONS.glob("pred_*.csv")]
        if preds:
            eval_seen_unseen(pd.concat(preds, ignore_index=True), seen)
    manifest["stages"]["transfer"] = {"eligible": eligible, "n_metric_rows": len(metrics)}
    atomic_write_json(MANIFEST, manifest)

    boot = run_all_comparisons(PREDICTIONS, eligible or [], model="xgb")
    if boot.empty and eligible:
        boot = run_all_comparisons(PREDICTIONS, eligible, model="ridge")
    manifest["stages"]["bootstrap"] = {"rows": len(boot)}

    preds = [pd.read_csv(p) for p in PREDICTIONS.glob("pred_*.csv")]
    ped_pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    fv = faers_validation(ped_pred, ror, mapping)
    manifest["stages"]["faers_validation"] = {"rows": len(fv)}

    write_reports(manifest, metrics, boot, adult_lab, ped_lab, ev)
    manifest["ended_at"] = now_pt()
    hashes = {}
    for p in list(RESULTS.glob("*")) + list(REPORTS.glob("*")):
        if p.is_file() and p.stat().st_size < 50_000_000:
            try: hashes[str(p)] = sha256_file(p)
            except Exception: pass
    manifest["artifact_sha256"] = hashes
    atomic_write_json(MANIFEST, manifest)
    log("=== RESUME COMPLETE ===")

if __name__ == "__main__":
    main()
