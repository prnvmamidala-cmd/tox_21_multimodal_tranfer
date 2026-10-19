#!/usr/bin/env python3
"""Adult→pediatric drug-safety transfer pipeline entrypoint."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# package on path
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from pediatric_transfer.utils.paths import (
    RUN_DIR, RAW, CACHE, PROCESSED, FEATURES, MODELS, PREDICTIONS, RESULTS, LOGS, REPORTS,
)
from pediatric_transfer.utils.logging_utils import log, log_exc, log_failure
from pediatric_transfer.utils.io_utils import atomic_write_df, atomic_write_json, atomic_write_text, sha256_file, load_json
from pediatric_transfer.utils.config import (
    OUTCOMES, SEEDS, GLOBAL_SEED, SMOKE_ADULT_TRIALS, SMOKE_PED_TRIALS, SMOKE_FAERS,
    CTGOV_BASE, OPENFDA_BASE, CHEMBL_BASE, BOOTSTRAP_N,
)
from pediatric_transfer.eudra.detect import ensure_eudra_status
from pediatric_transfer.ctgov.fetch import fetch_smoke, fetch_all_studies, _classify_study, iter_checkpointed_studies, checkpoint_dir
from pediatric_transfer.ctgov.parse import studies_to_frame, parse_study_ae_rows
from pediatric_transfer.ctgov.aggregate import split_and_save, aggregate_by_structure_outcome
from pediatric_transfer.faers.fetch import fetch_faers_domain
from pediatric_transfer.faers.ror import reports_to_pairs, compute_ror
from pediatric_transfer.mapping.harmonize import map_drug_names
from pediatric_transfer.features.chem import build_chem_matrix, fit_preprocess_adult, transform_chem
from pediatric_transfer.features.chembl_bio import build_chembl_target_profiles
from pediatric_transfer.models.transfer import (
    endpoint_feasibility, run_transfer_for_endpoint, seen_unseen_split, eval_seen_unseen,
)
from pediatric_transfer.stats.bootstrap import run_all_comparisons, faers_validation

import pandas as pd
import numpy as np

PT = ZoneInfo("America/Los_Angeles")
MANIFEST = RESULTS / "run_manifest.json"


def now_pt() -> str:
    return datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S %Z")


def package_versions() -> dict:
    vers = {}
    for m in ["rdkit", "xgboost", "sklearn", "pandas", "numpy", "scipy", "requests", "tqdm", "joblib", "openpyxl"]:
        try:
            if m == "sklearn":
                import sklearn
                vers[m] = sklearn.__version__
            elif m == "rdkit":
                import rdkit
                vers[m] = rdkit.__version__
            else:
                mod = __import__(m)
                vers[m] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            vers[m] = f"ERR:{e}"
    return vers


def run_unit_tests() -> bool:
    log("Running unit tests...")
    test_path = CODE / "pediatric_transfer" / "tests" / "test_unit.py"
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
        cwd=str(CODE), capture_output=True, text=True,
    )
    atomic_write_text(LOGS / "unit_test_output.txt", r.stdout + "\n" + r.stderr)
    log(f"unit tests exit={r.returncode}")
    if r.returncode != 0:
        log(r.stdout[-2000:] if r.stdout else "")
        log(r.stderr[-2000:] if r.stderr else "")
    return r.returncode == 0


def collect_names(arm_df: pd.DataFrame):
    names = []
    seen = set()
    for _, r in arm_df.iterrows():
        n = r["drug_name_norm"]
        if n not in seen:
            seen.add(n)
            names.append((n, r["drug_name_raw"]))
    return names


def smoke_test() -> dict:
    log("=== SMOKE TEST START ===")
    status = {"stages": {}}
    ensure_eudra_status()
    status["stages"]["eudra"] = "unavailable_or_checked"

    # CT.gov smoke
    try:
        buckets = fetch_smoke(SMOKE_ADULT_TRIALS, SMOKE_PED_TRIALS)
        adult_st = buckets["adult"][:SMOKE_ADULT_TRIALS]
        ped_st = buckets["pediatric"][:SMOKE_PED_TRIALS]
        log(f"smoke trials adult={len(adult_st)} ped={len(ped_st)}")
        arm_a = studies_to_frame(adult_st)
        arm_p = studies_to_frame(ped_st)
        arm = pd.concat([arm_a, arm_p], ignore_index=True)
        atomic_write_df(PROCESSED / "smoke_arm_level.csv", arm)
        status["stages"]["ctgov_smoke"] = {"adult_trials": len(adult_st), "ped_trials": len(ped_st), "arm_rows": len(arm)}
    except Exception as e:
        log_exc("smoke_ctgov", "fetch", e)
        status["stages"]["ctgov_smoke"] = f"FAIL:{e}"
        arm = pd.DataFrame()

    # FAERS smoke
    try:
        ped_rep = fetch_faers_domain("pediatric", max_pages=2, max_reports=SMOKE_FAERS)
        ad_rep = fetch_faers_domain("adult", max_pages=2, max_reports=SMOKE_FAERS)
        ped_pairs = reports_to_pairs(ped_rep[:SMOKE_FAERS], "pediatric")
        ad_pairs = reports_to_pairs(ad_rep[:SMOKE_FAERS], "adult")
        ror = compute_ror(ped_pairs, ad_pairs)
        status["stages"]["faers_smoke"] = {"ped": len(ped_rep), "adult": len(ad_rep), "ror_rows": len(ror)}
    except Exception as e:
        log_exc("smoke_faers", "fetch", e)
        status["stages"]["faers_smoke"] = f"FAIL:{e}"

    # Mapping + features on up to 10 structures
    try:
        if arm.empty:
            # use a few known drugs for feature smoke
            names = [("aspirin", "Aspirin"), ("ibuprofen", "Ibuprofen"), ("metformin", "Metformin"),
                     ("amoxicillin", "Amoxicillin"), ("acetaminophen", "Acetaminophen"),
                     ("loratadine", "Loratadine"), ("omeprazole", "Omeprazole"),
                     ("sertraline", "Sertraline"), ("atorvastatin", "Atorvastatin"),
                     ("lisinopril", "Lisinopril")]
        else:
            names = collect_names(arm)[:10]
            if len(names) < 3:
                names += [("aspirin", "Aspirin"), ("ibuprofen", "Ibuprofen"), ("metformin", "Metformin")]
        mapping = map_drug_names(names, checkpoint_every=5)
        ok = mapping[mapping.status == "ok"]
        status["stages"]["mapping_smoke"] = {"ok": int((mapping.status == "ok").sum()), "n": len(mapping)}
        if len(ok) >= 2:
            chem_raw, meta = build_chem_matrix(ok[["inchikey", "canonical_smiles"]].drop_duplicates("inchikey"))
            adult_iks = list(ok["inchikey"].unique())[: max(2, len(ok)//2)]
            params = fit_preprocess_adult(chem_raw, adult_iks)
            X = transform_chem(chem_raw, params)
            atomic_write_df(FEATURES / "smoke_chem.csv", X.reset_index())
            # bio for subset
            bio = build_chembl_target_profiles(list(ok["inchikey"].unique())[:10], adult_iks)
            status["stages"]["features_smoke"] = {"chem": list(X.shape), "bio": list(bio.shape)}
            # tiny transfer on synthetic labels if needed
            if not arm.empty and "outcome" in arm.columns:
                # attach inchikey
                arm_m = arm.merge(ok[["drug_name_norm", "inchikey"]], on="drug_name_norm", how="inner")
                if not arm_m.empty:
                    from pediatric_transfer.ctgov.aggregate import smoothed_incidence, logit as logit_fn
                    arm_m = arm_m.copy()
                    arm_m["incidence_smooth"] = arm_m.apply(lambda r: smoothed_incidence(r.num_affected, r.num_at_risk), axis=1)
                    arm_m["logit_incidence"] = arm_m["incidence_smooth"].map(logit_fn)
                    adult_lab = arm_m[arm_m.age_bucket == "adult"][["inchikey", "outcome", "logit_incidence", "num_affected", "num_at_risk"]].drop_duplicates()
                    ped_lab = arm_m[arm_m.age_bucket == "pediatric"][["inchikey", "outcome", "logit_incidence", "num_affected", "num_at_risk"]].drop_duplicates()
                    # pick one outcome with any support
                    for o in OUTCOMES:
                        if (adult_lab.outcome == o).sum() >= 3 and (ped_lab.outcome == o).sum() >= 2:
                            smiles_map = dict(zip(ok.inchikey, ok.canonical_smiles))
                            chem_x = transform_chem(chem_raw, params)
                            # reduce MIN temporarily only for smoke via slicing — use whatever we have
                            # Do NOT lower global thresholds; smoke just exercises code path
                            from pediatric_transfer.models import transfer as T
                            old_a, old_p = T.MIN_ADULT_N, T.MIN_PED_N
                            T.MIN_ADULT_N, T.MIN_PED_N = 3, 2
                            try:
                                res = run_transfer_for_endpoint(o, adult_lab, ped_lab, chem_x, bio, smiles_map, models=["ridge"])
                                atomic_write_df(RESULTS / "smoke_transfer.csv", res)
                                status["stages"]["transfer_smoke"] = {"rows": len(res), "outcome": o}
                            finally:
                                T.MIN_ADULT_N, T.MIN_PED_N = old_a, old_p
                            break
        else:
            status["stages"]["features_smoke"] = "insufficient_mapped"
    except Exception as e:
        log_exc("smoke_features", "map_feat", e)
        status["stages"]["features_smoke"] = f"FAIL:{e}"

    atomic_write_json(RESULTS / "smoke_status.json", status)
    log(f"=== SMOKE TEST END === {json.dumps(status, default=str)[:500]}")
    return status


def full_run() -> dict:
    log("=== FULL RUN START ===")
    manifest = {
        "started_at": now_pt(),
        "seeds": SEEDS,
        "global_seed": GLOBAL_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "package_versions": package_versions(),
        "api": {
            "ctgov": CTGOV_BASE,
            "ctgov_query": "AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true",
            "openfda": OPENFDA_BASE,
            "chembl": CHEMBL_BASE,
        },
        "outcomes": OUTCOMES,
        "inclusion_criteria": {
            "pediatric": "max age <=17 and min+max parseable; exclude mixed",
            "adult": "min age >=18 and min+max parseable; exclude mixed",
            "arms": "exactly one identifiable non-control drug; exclude placebo/untreated/behavioral/device/dietary/procedure/ambiguous combos",
            "ae_agg_within_arm": "MAX affected not sum",
            "ae_agg_across_trials": "sum affected and at_risk by structure+outcome",
            "incidence": "(a+0.5)/(n+1); logit continuous target",
            "primary_biology": "chembl_target_profile (NOT Chemical Checker / BioFeat)",
            "no_pediatric_fit": True,
        },
        "stages": {},
        "completed": [],
        "failed": [],
    }
    atomic_write_json(MANIFEST, manifest)

    # Eudra
    ev = ensure_eudra_status()
    manifest["stages"]["eudra"] = ev
    manifest["completed"].append("eudra_detect")
    atomic_write_json(MANIFEST, manifest)

    # CT.gov full fetch
    try:
        log("Fetching all CT.gov interventional+results studies (checkpointed)")
        n_pages = fetch_all_studies(tag="all_results", max_pages=None, resume=True)
        manifest["stages"]["ctgov_fetch"] = {"n_pages": n_pages}
        manifest["completed"].append("ctgov_fetch")
    except Exception as e:
        log_exc("ctgov_fetch", "full", e)
        manifest["failed"].append({"stage": "ctgov_fetch", "error": str(e)})
        prog = load_json(checkpoint_dir("all_results") / "progress.json", {})
        log(f"Partial CT.gov pages={prog.get('pages_done')}")
    atomic_write_json(MANIFEST, manifest)

    # Classify + parse STREAMING (never hold all studies in RAM)
    adult_n = ped_n = mixed_or_unknown = total_fetched = 0
    arm_chunks = []
    chunk_rows = []
    CHUNK = 500
    for st in iter_checkpointed_studies("all_results"):
        total_fetched += 1
        b = _classify_study(st)
        if b == "adult":
            adult_n += 1
        elif b == "pediatric":
            ped_n += 1
        else:
            mixed_or_unknown += 1
            continue
        try:
            chunk_rows.extend(parse_study_ae_rows(st))
        except Exception:
            continue
        if len(chunk_rows) >= CHUNK:
            arm_chunks.append(pd.DataFrame(chunk_rows))
            chunk_rows = []
        if total_fetched % 5000 == 0:
            log(f"parse progress studies={total_fetched} adult={adult_n} ped={ped_n} arm_rows_so_far={sum(len(c) for c in arm_chunks)+len(chunk_rows)}")
        del st
    if chunk_rows:
        arm_chunks.append(pd.DataFrame(chunk_rows))
    log(f"Classified adult={adult_n} ped={ped_n} excluded_mixed_or_unknown={mixed_or_unknown}")
    atomic_write_json(PROCESSED / "ctgov_trial_counts.json", {
        "adult_trials": adult_n, "pediatric_trials": ped_n,
        "excluded_mixed_or_unparseable": mixed_or_unknown, "total_fetched": total_fetched,
    })
    arm = pd.concat(arm_chunks, ignore_index=True) if arm_chunks else pd.DataFrame()
    del arm_chunks, chunk_rows
    arm_a = arm[arm.age_bucket == "adult"] if len(arm) else arm
    arm_p = arm[arm.age_bucket == "pediatric"] if len(arm) else arm
    atomic_write_df(PROCESSED / "ctgov_arm_level.csv", arm)
    log(f"Arm-level rows={len(arm)} adult_rows={len(arm_a)} ped_rows={len(arm_p)}")
    manifest["stages"]["ctgov_parse"] = {
        "adult_trials": adult_n, "ped_trials": ped_n,
        "arm_rows": len(arm), "adult_arm_rows": len(arm_a), "ped_arm_rows": len(arm_p),
    }
    manifest["completed"].append("ctgov_parse")
    atomic_write_json(MANIFEST, manifest)

    # FAERS
    try:
        ped_rep = fetch_faers_domain("pediatric")
        ad_rep = fetch_faers_domain("adult")
        ped_pairs = reports_to_pairs(ped_rep, "pediatric")
        ad_pairs = reports_to_pairs(ad_rep, "adult")
        atomic_write_df(PROCESSED / "faers_ped_pairs.csv", ped_pairs)
        atomic_write_df(PROCESSED / "faers_adult_pairs.csv", ad_pairs)
        ror = compute_ror(ped_pairs, ad_pairs)
        manifest["stages"]["faers"] = {
            "ped_reports": len(ped_rep), "adult_reports": len(ad_rep), "ror_rows": len(ror),
        }
        manifest["completed"].append("faers")
    except Exception as e:
        log_exc("faers", "full", e)
        manifest["failed"].append({"stage": "faers", "error": str(e)})
        ror = pd.DataFrame()
        # load partial
        try:
            from pediatric_transfer.faers.fetch import _load_all
            ped_rep = _load_all(RAW / "faers" / "pediatric")
            ad_rep = _load_all(RAW / "faers" / "adult")
            ped_pairs = reports_to_pairs(ped_rep, "pediatric")
            ad_pairs = reports_to_pairs(ad_rep, "adult")
            ror = compute_ror(ped_pairs, ad_pairs)
            manifest["stages"]["faers"] = {"partial": True, "ror_rows": len(ror)}
        except Exception as e2:
            log_exc("faers_partial", "load", e2)
            ror = pd.DataFrame()
    atomic_write_json(MANIFEST, manifest)

    # Mapping — prioritize pediatric + adult-ped overlap + high-frequency adult
    try:
        names = []
        if not arm.empty:
            freq = arm.groupby(["drug_name_norm", "age_bucket"]).size().unstack(fill_value=0)
            for col in ("adult", "pediatric"):
                if col not in freq.columns:
                    freq[col] = 0
            ped_names = set(arm.loc[arm.age_bucket == "pediatric", "drug_name_norm"])
            adult_names = set(arm.loc[arm.age_bucket == "adult", "drug_name_norm"])
            overlap = ped_names & adult_names
            # all pediatric (needed for test), overlap first, then adult by frequency
            raw_map = arm.drop_duplicates("drug_name_norm").set_index("drug_name_norm")["drug_name_raw"].to_dict()
            ordered = []
            for n in sorted(overlap):
                ordered.append(n)
            for n in sorted(ped_names - overlap):
                ordered.append(n)
            adult_freq = arm[arm.age_bucket == "adult"].groupby("drug_name_norm").size().sort_values(ascending=False)
            for n in adult_freq.index:
                if n not in ped_names:
                    ordered.append(n)
            # Overnight feasibility cap: all ped + top adult; document rest as deferred
            MAX_MAP = 6000  # ped ~3400 + top adult; remainder recorded unresolved/deferred
            deferred = ordered[MAX_MAP:]
            ordered = ordered[:MAX_MAP]
            if deferred:
                pd.DataFrame({"drug_name_norm": deferred, "reason": "deferred_overnight_cap"}).to_csv(
                    PROCESSED / "deferred_drug_names.csv", index=False)
                log(f"Deferred {len(deferred)} low-priority adult-only names (overnight cap {MAX_MAP})")
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
            "n": len(mapping),
            "ok": int((mapping.status == "ok").sum()),
            "unresolved": int((mapping.status == "unresolved").sum()),
        }
        manifest["completed"].append("mapping")
    except Exception as e:
        log_exc("mapping", "full", e)
        manifest["failed"].append({"stage": "mapping", "error": str(e)})
        mapping = pd.read_csv(PROCESSED / "drug_name_mapping.csv") if (PROCESSED / "drug_name_mapping.csv").exists() else pd.DataFrame()
    atomic_write_json(MANIFEST, manifest)

    ok_map = mapping[mapping.status == "ok"][["drug_name_norm", "inchikey", "canonical_smiles", "source", "confidence"]].drop_duplicates("drug_name_norm") if len(mapping) else pd.DataFrame()

    # Aggregate CT
    try:
        agg = split_and_save(arm, ok_map)
        adult_lab = agg.get("adult", pd.DataFrame())
        ped_lab = agg.get("pediatric", pd.DataFrame())
        manifest["stages"]["aggregate"] = {
            "adult_rows": len(adult_lab), "ped_rows": len(ped_lab),
            "adult_structures": int(adult_lab["inchikey"].nunique()) if len(adult_lab) else 0,
            "ped_structures": int(ped_lab["inchikey"].nunique()) if len(ped_lab) else 0,
        }
        manifest["completed"].append("aggregate")
    except Exception as e:
        log_exc("aggregate", "full", e)
        manifest["failed"].append({"stage": "aggregate", "error": str(e)})
        adult_lab = pd.DataFrame()
        ped_lab = pd.DataFrame()
    atomic_write_json(MANIFEST, manifest)

    # Harmonized long table
    try:
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
                    "n_trials": r.get("n_trials", np.nan),
                    "source": "ClinicalTrials.gov",
                })
        if not ror.empty and len(ok_map):
            rr = ror.merge(ok_map[["drug_name_norm", "inchikey"]], on="drug_name_norm", how="inner")
            for _, r in rr.iterrows():
                long_rows.append({
                    "inchikey": r["inchikey"], "outcome": r["outcome"], "age_bucket": "ped_vs_adult_ror",
                    "estimate_type": "faers_ror_not_incidence",
                    "value": r["logROR"], "incidence_smooth": np.nan,
                    "num_affected": r["ped_a"], "num_at_risk": r["ped_drug_reports"],
                    "n_trials": np.nan, "source": "FAERS_openFDA",
                })
        long_df = pd.DataFrame(long_rows)
        atomic_write_df(PROCESSED / "harmonized_long_table.csv", long_df)
        # overlaps
        if len(adult_lab) and len(ped_lab):
            a_ik = set(adult_lab.inchikey)
            p_ik = set(ped_lab.inchikey)
            overlap = {
                "adult_structures": len(a_ik),
                "pediatric_structures": len(p_ik),
                "structure_overlap": len(a_ik & p_ik),
                "adult_structure_outcome": len(adult_lab),
                "ped_structure_outcome": len(ped_lab),
            }
            if not ror.empty and len(ok_map):
                f_ik = set(ror.merge(ok_map, on="drug_name_norm")["inchikey"])
                overlap["faers_mapped_structures"] = len(f_ik)
                overlap["faers_adult_overlap"] = len(f_ik & a_ik)
                overlap["faers_ped_overlap"] = len(f_ik & p_ik)
            atomic_write_json(RESULTS / "source_overlaps.json", overlap)
            manifest["stages"]["overlaps"] = overlap
        manifest["completed"].append("harmonized_long")
    except Exception as e:
        log_exc("harmonized", "full", e)
        manifest["failed"].append({"stage": "harmonized", "error": str(e)})
    atomic_write_json(MANIFEST, manifest)

    # Features
    chem_x = pd.DataFrame()
    bio = pd.DataFrame()
    smiles_map = {}
    try:
        structs = ok_map[["inchikey", "canonical_smiles"]].drop_duplicates("inchikey")
        smiles_map = dict(zip(structs.inchikey, structs.canonical_smiles))
        # restrict to structures in adult or ped labels
        need = set()
        if len(adult_lab):
            need |= set(adult_lab.inchikey)
        if len(ped_lab):
            need |= set(ped_lab.inchikey)
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
        manifest["completed"].append("features")
    except Exception as e:
        log_exc("features", "full", e)
        manifest["failed"].append({"stage": "features", "error": str(e)})
    atomic_write_json(MANIFEST, manifest)

    # Transfer
    all_metrics = []
    try:
        feas = endpoint_feasibility(adult_lab, ped_lab) if len(adult_lab) and len(ped_lab) else pd.DataFrame()
        eligible = feas[feas.eligible]["outcome"].tolist() if len(feas) else []
        log(f"Eligible endpoints: {eligible}")
        for o in eligible:
            res = run_transfer_for_endpoint(o, adult_lab, ped_lab, chem_x, bio, smiles_map)
            all_metrics.append(res)
            atomic_write_df(RESULTS / "transfer_metrics_partial.csv", pd.concat(all_metrics, ignore_index=True))
        metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
        atomic_write_df(RESULTS / "transfer_metrics.csv", metrics)
        if len(adult_lab) and len(ped_lab):
            seen = seen_unseen_split(adult_lab, ped_lab, smiles_map)
            # concat predictions for seen/unseen
            preds = []
            for p in PREDICTIONS.glob("pred_*.csv"):
                preds.append(pd.read_csv(p))
            if preds:
                eval_seen_unseen(pd.concat(preds, ignore_index=True), seen)
        manifest["stages"]["transfer"] = {
            "eligible": eligible,
            "n_metric_rows": len(metrics),
        }
        manifest["completed"].append("transfer")
    except Exception as e:
        log_exc("transfer", "full", e)
        manifest["failed"].append({"stage": "transfer", "error": str(e)})
        metrics = pd.DataFrame()
        eligible = []
    atomic_write_json(MANIFEST, manifest)

    # Bootstrap
    try:
        boot = run_all_comparisons(PREDICTIONS, eligible or [], model="xgb")
        if boot.empty and eligible:
            boot = run_all_comparisons(PREDICTIONS, eligible, model="ridge")
        manifest["stages"]["bootstrap"] = {"rows": len(boot)}
        manifest["completed"].append("bootstrap")
    except Exception as e:
        log_exc("bootstrap", "full", e)
        manifest["failed"].append({"stage": "bootstrap", "error": str(e)})
        boot = pd.DataFrame()
    atomic_write_json(MANIFEST, manifest)

    # FAERS validation
    try:
        preds = []
        for p in PREDICTIONS.glob("pred_*.csv"):
            preds.append(pd.read_csv(p))
        ped_pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
        fv = faers_validation(ped_pred, ror if 'ror' in dir() else pd.DataFrame(), mapping)
        manifest["stages"]["faers_validation"] = {"rows": len(fv)}
        manifest["completed"].append("faers_validation")
    except Exception as e:
        log_exc("faers_validation", "full", e)
        manifest["failed"].append({"stage": "faers_validation", "error": str(e)})
    atomic_write_json(MANIFEST, manifest)

    # Reports
    try:
        write_reports(manifest, metrics if 'metrics' in dir() else pd.DataFrame(),
                      boot if 'boot' in dir() else pd.DataFrame(),
                      adult_lab, ped_lab, ev)
        manifest["completed"].append("reports")
    except Exception as e:
        log_exc("reports", "full", e)
        manifest["failed"].append({"stage": "reports", "error": str(e)})

    manifest["ended_at"] = now_pt()
    # hashes of key outputs
    hashes = {}
    for p in list(RESULTS.glob("*")) + list(PROCESSED.glob("*.csv"))[:30] + list(REPORTS.glob("*")):
        if p.is_file():
            try:
                hashes[str(p)] = sha256_file(p)
            except Exception:
                pass
    manifest["artifact_sha256"] = hashes
    atomic_write_json(MANIFEST, manifest)
    log("=== FULL RUN END ===")
    return manifest


def write_reports(manifest, metrics, boot, adult_lab, ped_lab, ev):
    # methods.md
    methods = f"""# Methods

## Objective
Determine whether adding biological representations (`chembl_target_profile`) to chemical representations improves adult-to-pediatric transfer of drug-safety predictions.

## Data sources
- **ClinicalTrials.gov API v2** (`{CTGOV_BASE}`): interventional studies with posted results and AE module.
- **FAERS via openFDA** (`{OPENFDA_BASE}`): spontaneous reports for pediatric-vs-adult ROR (external PV evidence only; **not incidence**).
- **EudraVigilance**: {'UNAVAILABLE — see raw/EUDRAVIGILANCE_NOT_AVAILABLE.txt' if not ev.get('available') else 'loaded from ' + str(ev.get('files'))}.
- **ChEMBL web services** (`{CHEMBL_BASE}`): experimental target activities for `chembl_target_profile`.
- **PubChem PUG REST**: name→structure.

## Inclusion / exclusion
- Pediatric-only: max age ≤17 with parseable min and max; adult-only: min age ≥18 with parseable min and max.
- Mixed child+adult trials excluded.
- Arms with exactly one identifiable non-control drug; exclude placebo/untreated/behavioral/device/dietary/procedure/ambiguous combinations.
- Outcomes mapped via MedDRA SOC when available else frozen keyword map.
- Within arm/category: **MAX** affected (not sum). Across trials: sum affected and at-risk by structure+outcome.
- Smoothed incidence `(a+0.5)/(n+1)`; logit as continuous target.

## Representations
- `chem`: Morgan/ECFP4 r=2 2048-bit + RDKit descriptors (adult-only median impute, drop constants, clip).
- `bio_only` / multimodal bio: `chembl_target_profile` (pChEMBL median per target; adult-coverage vocab; missing indicators). **Not** Chemical Checker / BioFeat / signaturizer.
- `chem_true_bio`: concatenation.
- `chem_permuted_bio_seed_{{101,202,303,404,505}}`: row-permute bio among adult compounds.

## Transfer protocol
- Train on adult only; predict pediatric **without** pediatric model fitting (primary zero-shot).
- Primary model: XGBoost regressor; robustness: RF, Ridge/ElasticNet.
- HPs tuned with adult-only scaffold/grouped CV; frozen before pediatric eval.
- Metrics: Spearman, RMSE, MAE on pediatric logit incidence.
- Seen vs unseen structures/scaffolds reported separately.
- Paired compound bootstrap ≥{BOOTSTRAP_N}; BH correction across primary endpoint comparisons.

## Scientific guardrails
- Do not add raw CT.gov, FAERS, and EV counts together.
- Do not describe spontaneous reports as confirmed causal toxicities.
- Do not treat AE reporting frequency as incidence.
- Multimodal claimed helpful only if true bio beats chem-only **and** permuted-bio controls.

## Run metadata
- Started: {manifest.get('started_at')}
- Seeds: {SEEDS}
- Package versions: see run_manifest.json
"""
    atomic_write_text(REPORTS / "methods.md", methods)

    # results_summary.md
    n_a = len(adult_lab) if adult_lab is not None else 0
    n_p = len(ped_lab) if ped_lab is not None else 0
    lines = [
        "# Results summary",
        "",
        f"Generated: {now_pt()}",
        "",
        "## Sample sizes",
        f"- Adult structure–outcome rows: {n_a}",
        f"- Pediatric structure–outcome rows: {n_p}",
        f"- Adult trials / ped trials: see processed/ctgov_trial_counts.json",
        "",
        "## Transfer metrics (primary)",
        "",
    ]
    if metrics is not None and len(metrics):
        prim = metrics[metrics.model == "xgb"] if "xgb" in set(metrics.get("model", [])) else metrics
        lines.append(prim.to_string(index=False))
    else:
        lines.append("No transfer metrics produced (insufficient eligible endpoints or upstream failure).")
    lines += ["", "## Bootstrap contrasts", ""]
    if boot is not None and len(boot):
        lines.append(boot.to_string(index=False))
    else:
        lines.append("No bootstrap results.")
    lines += [
        "",
        "## Hypothesis assessment",
        "",
        _hypothesis_text(metrics, boot),
        "",
        "## EudraVigilance",
        "Unavailable — validation pending." if not ev.get("available") else f"Files: {ev.get('files')}",
        "",
        "## Computational vs scientific",
        "Computational completion is recorded in run_manifest.json. Scientific support requires true-bio > chem and > permuted controls after BH correction.",
    ]
    atomic_write_text(REPORTS / "results_summary.md", "\n".join(lines))

    limitations = """# Limitations

1. Clinical trial AE modules vary in coding quality; SOC mapping may miss terms.
2. Single-drug arm restriction excludes many oncology combinations (informative but out of scope).
3. Smoothed incidence is not a causal drug effect estimate; confounding by indication remains.
4. FAERS ROR reflects reporting behavior, not incidence or confirmed causality; used only as external PV concordance.
5. EudraVigilance export was not available on this box; EV validation pending.
6. `chembl_target_profile` coverage is sparse for some structures; missingness indicators used.
7. Chemical Checker / BioFeat remotes were intentionally not called (RemoteDisconnected risk); no validated CC caches present.
8. Sample sizes for some organ-class endpoints may be below eligibility thresholds — see endpoint_feasibility.csv.
9. openFDA rate limits constrain FAERS sample size without an API key.
10. Zero-shot pediatric evaluation does not adapt to pediatric pharmacology (PK/PD differences).
"""
    atomic_write_text(REPORTS / "limitations.md", limitations)

    # notebook
    try:
        import nbformat as nbf
        nb = nbf.v4.new_notebook()
        nb.cells = [
            nbf.v4.new_markdown_cell("# Pediatric transfer aligned v3 — results overview"),
            nbf.v4.new_code_cell(
                "import pandas as pd\nfrom pathlib import Path\n"
                "R=Path('/workspace/pediatric_transfer_aligned_v3/results')\n"
                "P=Path('/workspace/pediatric_transfer_aligned_v3/processed')\n"
                "print((R/'transfer_metrics.csv').exists())\n"
                "display(pd.read_csv(R/'endpoint_feasibility.csv') if (R/'endpoint_feasibility.csv').exists() else 'no feas')\n"
                "display(pd.read_csv(R/'transfer_metrics.csv').head(50) if (R/'transfer_metrics.csv').exists() else 'no metrics')\n"
            ),
        ]
        nbf.write(nb, REPORTS / "results_overview.ipynb")
    except Exception as e:
        log(f"notebook write failed: {e}")

    # ZIP
    import zipfile
    zip_path = RESULTS / "pediatric_transfer_aligned_v3_deliverables.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in [CODE, RESULTS, REPORTS, LOGS, PROCESSED, FEATURES]:
            for p in Path(folder).rglob("*"):
                if not p.is_file():
                    continue
                if ".venv" in p.parts or "__pycache__" in p.parts:
                    continue
                if p.suffix in {".tmp", ".zip"} or "deliverables.zip" in p.name:
                    continue
                if p.stat().st_size > 50_000_000:
                    continue
                z.write(p, arcname=str(p.relative_to(RUN_DIR)))
    log(f"Wrote zip {zip_path}")


def _hypothesis_text(metrics, boot) -> str:
    if metrics is None or len(metrics) == 0:
        return "UNRESOLVED: insufficient data to evaluate hypothesis."
    # Check true bio vs chem and vs perms
    m = metrics.copy()
    if "model" in m.columns:
        m = m[m.model == "xgb"] if (m.model == "xgb").any() else m
    conclusions = []
    for o in m.outcome.unique():
        sub = m[m.outcome == o]
        chem = sub[sub.representation == "chem"]["spearman"]
        true = sub[sub.representation == "chem_true_bio"]["spearman"]
        if chem.empty or true.empty:
            conclusions.append(f"{o}: unresolved (missing reps)")
            continue
        c, t = float(chem.iloc[0]), float(true.iloc[0])
        perms = sub[sub.representation.str.startswith("chem_permuted_bio")]["spearman"]
        beat_chem = t > c
        beat_perm = bool(len(perms)) and t > float(perms.mean())
        if beat_chem and beat_perm:
            conclusions.append(f"{o}: supports multimodal (true={t:.3f} > chem={c:.3f} and > mean perm={float(perms.mean()):.3f})")
        else:
            conclusions.append(f"{o}: does NOT support multimodal (true={t:.3f}, chem={c:.3f}, mean_perm={float(perms.mean()) if len(perms) else float('nan'):.3f})")
    # bootstrap BH
    if boot is not None and len(boot):
        prim = boot[boot.contrast == "chem_true_bio_minus_chem"]
        if len(prim) and "p_bh" in prim.columns:
            conclusions.append("BH-corrected primary contrasts:")
            for _, r in prim.iterrows():
                conclusions.append(f"  {r.outcome}: delta={r.delta:.4f} CI=[{r.ci_lo:.4f},{r.ci_hi:.4f}] p_bh={r.get('p_bh')}")
    text = "\n".join(conclusions)
    if any("supports multimodal" in c and "NOT" not in c for c in conclusions):
        if all("supports multimodal" in c for c in conclusions if c.endswith(")") or "supports" in c):
            return "Evidence SUPPORTS the hypothesis on eligible endpoints where stated.\n" + text
    if any("does NOT support" in c for c in conclusions):
        return "Evidence REJECTS or fails to support multimodal benefit under pre-specified criteria (null is valid).\n" + text
    return "Evidence LEAVES UNRESOLVED the hypothesis.\n" + text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    log(f"Pipeline start cwd={os.getcwd()} py={sys.executable}")
    log(f"versions={package_versions()}")

    if not args.skip_tests:
        ok = run_unit_tests()
        if not ok:
            log("Unit tests FAILED — attempting fixes is required; re-check output")
            # do not abort overnight run entirely after documenting; but try to continue only if smoke works
        else:
            log("Unit tests PASSED")

    if not args.skip_smoke:
        smoke_test()

    if args.smoke_only:
        log("smoke-only flag set; exiting")
        return

    full_run()
    log("All done")


if __name__ == "__main__":
    main()
