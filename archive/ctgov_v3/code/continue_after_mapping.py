#!/usr/bin/env python3
"""After mapping complete: rebuild SO, features, transfer, bootstrap, comparison. Uses FAERS v3 25k."""
from __future__ import annotations
import os, sys, json, time, zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

# load key silently
envf = Path("/workspace/pediatric_transfer_aligned_v3/cache/.openfda_env")
if envf.exists():
    for line in envf.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        if "=" in line:
            k,v=line.split("=",1); os.environ[k.strip()]=v.strip().strip('"').strip("'")
        else:
            os.environ["OPENFDA_API_KEY"]=line

from run_pipeline import now_pt, package_versions, write_reports, MANIFEST
from pediatric_transfer.utils.paths import *
from pediatric_transfer.utils.logging_utils import log, log_exc
from pediatric_transfer.utils.io_utils import *
from pediatric_transfer.utils.config import *
from pediatric_transfer.eudra.detect import ensure_eudra_status
from pediatric_transfer.ctgov.aggregate import split_and_save
from pediatric_transfer.faers.fetch import _load_all
from pediatric_transfer.faers.ror import reports_to_pairs, compute_ror
from pediatric_transfer.features.chem import build_chem_matrix, fit_preprocess_adult, transform_chem
from pediatric_transfer.features.chembl_bio import build_chembl_target_profiles
from pediatric_transfer.models.transfer import endpoint_feasibility, run_transfer_for_endpoint, seen_unseen_split, eval_seen_unseen
from pediatric_transfer.stats.bootstrap import run_all_comparisons, faers_validation
import pandas as pd
import numpy as np

PRELIM = Path("/workspace/pediatric_transfer_aligned_v2_PRELIMINARY_CHECKPOINT")
FAERS_V3 = CACHE / "faers_v3"
PT = ZoneInfo("America/Los_Angeles")

def main():
    log("=== CONTINUE AFTER MAPPING ===")
    mapping = pd.read_csv(PROCESSED / "drug_name_mapping.csv")
    report = load_json(PROCESSED / "mapping_completion_report.json", {})
    manifest = load_json(MANIFEST, {}) or {}
    manifest.update({
        "run_id": "pediatric_transfer_aligned_v3", "run_type": "FINAL",
        "continued_at": now_pt(), "seeds": SEEDS, "global_seed": GLOBAL_SEED,
        "bootstrap_n": BOOTSTRAP_N, "package_versions": package_versions(),
    })
    manifest.setdefault("stages", {})
    manifest.setdefault("completed", [])
    manifest.setdefault("failed", [])
    manifest["stages"]["mapping"] = report
    if "mapping" not in manifest["completed"]:
        manifest["completed"].append("mapping")

    # FAERS 25k
    ped = _load_all(FAERS_V3 / "pediatric")
    ad = _load_all(FAERS_V3 / "adult")
    ped_pairs = reports_to_pairs(ped, "pediatric")
    ad_pairs = reports_to_pairs(ad, "adult")
    atomic_write_df(PROCESSED / "faers_ped_pairs.csv", ped_pairs)
    atomic_write_df(PROCESSED / "faers_adult_pairs.csv", ad_pairs)
    ror = compute_ror(ped_pairs, ad_pairs)
    atomic_write_df(PROCESSED / "faers_ror.csv", ror)
    atomic_write_df(PROCESSED / "faers_ror_v3_25k.csv", ror)
    manifest["stages"]["faers"] = {
        "source": "faers_v3", "ped_reports": len(ped), "adult_reports": len(ad),
        "ror_rows": len(ror), "key_used": True, "expansion_complete": True,
    }
    log(f"FAERS ped={len(ped)} adult={len(ad)} ror={len(ror)}")

    arm = pd.read_csv(PROCESSED / "ctgov_arm_level.csv")
    ok_map = mapping[mapping.status == "ok"][["drug_name_norm","inchikey","canonical_smiles","source","confidence"]].drop_duplicates("drug_name_norm")
    agg = split_and_save(arm, ok_map)
    adult_lab = agg.get("adult", pd.DataFrame())
    ped_lab = agg.get("pediatric", pd.DataFrame())
    manifest["stages"]["aggregate"] = {
        "adult_rows": len(adult_lab), "ped_rows": len(ped_lab),
        "adult_structures": int(adult_lab["inchikey"].nunique()) if len(adult_lab) else 0,
        "ped_structures": int(ped_lab["inchikey"].nunique()) if len(ped_lab) else 0,
    }
    manifest["completed"].append("aggregate")
    atomic_write_json(MANIFEST, manifest)

    # harmonized
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
    if len(ror) and len(ok_map):
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
        overlap = {
            "adult_structures": len(a_ik), "pediatric_structures": len(p_ik),
            "structure_overlap": len(a_ik & p_ik),
            "adult_structure_outcome": len(adult_lab), "ped_structure_outcome": len(ped_lab),
        }
        f_ik = set(ror.merge(ok_map, on="drug_name_norm")["inchikey"]) if len(ror) else set()
        overlap["faers_mapped_structures"] = len(f_ik)
        overlap["faers_adult_overlap"] = len(f_ik & a_ik)
        overlap["faers_ped_overlap"] = len(f_ik & p_ik)
        atomic_write_json(RESULTS / "source_overlaps.json", overlap)
        manifest["stages"]["overlaps"] = overlap
    atomic_write_json(MANIFEST, manifest)

    # features
    structs = ok_map[["inchikey","canonical_smiles"]].drop_duplicates("inchikey")
    smiles_map = dict(zip(structs.inchikey, structs.canonical_smiles))
    need = set(adult_lab.inchikey)|set(ped_lab.inchikey) if len(adult_lab) or len(ped_lab) else set()
    structs = structs[structs.inchikey.isin(need)] if need else structs
    log(f"chem features n={len(structs)}")
    chem_raw, meta = build_chem_matrix(structs)
    adult_iks = list(adult_lab["inchikey"].unique()) if len(adult_lab) else list(structs.inchikey)
    params = fit_preprocess_adult(chem_raw, adult_iks)
    chem_x = transform_chem(chem_raw, params)
    atomic_write_df(FEATURES / "chem_processed.csv", chem_x.reset_index())
    log(f"chembl_target_profile n={len(structs)}")
    bio = build_chembl_target_profiles(list(structs.inchikey), adult_iks)
    manifest["stages"]["features"] = {"chem": list(chem_x.shape), "bio": list(bio.shape)}
    manifest["completed"].append("features")
    atomic_write_json(MANIFEST, manifest)

    # transfer
    for p in PREDICTIONS.glob("pred_*.csv"): p.unlink()
    feas = endpoint_feasibility(adult_lab, ped_lab) if len(adult_lab) and len(ped_lab) else pd.DataFrame()
    if len(feas): atomic_write_df(RESULTS / "endpoint_feasibility.csv", feas)
    eligible = feas[feas.eligible]["outcome"].tolist() if len(feas) else []
    log(f"eligible={eligible}")
    all_metrics = []
    for o in eligible:
        log(f"transfer {o}")
        res = run_transfer_for_endpoint(o, adult_lab, ped_lab, chem_x, bio, smiles_map)
        all_metrics.append(res)
        atomic_write_df(RESULTS / "transfer_metrics_partial.csv", pd.concat(all_metrics, ignore_index=True))
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    atomic_write_df(RESULTS / "transfer_metrics.csv", metrics)
    if len(adult_lab) and len(ped_lab):
        seen = seen_unseen_split(adult_lab, ped_lab, smiles_map)
        preds = [pd.read_csv(p) for p in PREDICTIONS.glob("pred_*.csv")]
        if preds: eval_seen_unseen(pd.concat(preds, ignore_index=True), seen)
    manifest["stages"]["transfer"] = {"eligible": eligible, "n_metric_rows": len(metrics)}
    manifest["completed"].append("transfer")
    atomic_write_json(MANIFEST, manifest)

    boot = run_all_comparisons(PREDICTIONS, eligible or [], model="xgb")
    manifest["stages"]["bootstrap"] = {"rows": len(boot)}
    manifest["completed"].append("bootstrap")
    preds = [pd.read_csv(p) for p in PREDICTIONS.glob("pred_*.csv")]
    ped_pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    fv = faers_validation(ped_pred, ror, mapping)
    manifest["stages"]["faers_validation"] = {"rows": len(fv)}
    atomic_write_json(MANIFEST, manifest)

    ev = ensure_eudra_status()
    write_reports(manifest, metrics, boot, adult_lab, ped_lab, ev)
    old_zip = RESULTS / "pediatric_transfer_aligned_v2_deliverables.zip"
    if old_zip.exists(): old_zip.unlink()

    # comparison (inline simplified from resume_complete_mapping)
    from resume_complete_mapping import stage_comparison, write_zip
    stage_comparison(manifest, metrics, boot, adult_lab, ped_lab, report)
    write_zip()
    manifest["ended_at"] = now_pt()
    manifest["completed"].append("reports")
    manifest["completed"].append("zip")
    atomic_write_json(MANIFEST, manifest)
    log("=== V3 CONTINUE COMPLETE ===")

if __name__ == "__main__":
    main()
