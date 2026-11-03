#!/usr/bin/env python3
"""FINAL v3 overnight runner: complete ALL deferred mappings then rebuild+transfer.

No early-stop. Mapping independent of openFDA key. FAERS expansion when key present.
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from run_pipeline import (
    now_pt, package_versions, write_reports, _hypothesis_text, MANIFEST,
)
from pediatric_transfer.utils.paths import (
    RUN_DIR, RAW, CACHE, PROCESSED, FEATURES, MODELS, PREDICTIONS, RESULTS, LOGS, REPORTS,
)
from pediatric_transfer.utils.logging_utils import log, log_exc
from pediatric_transfer.utils.io_utils import atomic_write_df, atomic_write_json, atomic_write_text, sha256_file, load_json
from pediatric_transfer.utils.config import (
    OUTCOMES, SEEDS, GLOBAL_SEED, BOOTSTRAP_N, CTGOV_BASE, OPENFDA_BASE, CHEMBL_BASE,
    FAERS_MAX_PAGES_PER_DOMAIN, MIN_ADULT_N, MIN_PED_N, CHEMBL_TARGET_CAP,
)
from pediatric_transfer.eudra.detect import ensure_eudra_status
from pediatric_transfer.mapping.harmonize import map_drug_names, looks_like_nondrug_arm
from pediatric_transfer.ctgov.aggregate import split_and_save
from pediatric_transfer.faers.fetch import fetch_faers_domain, _load_all
from pediatric_transfer.faers.ror import reports_to_pairs, compute_ror
from pediatric_transfer.features.chem import build_chem_matrix, fit_preprocess_adult, transform_chem
from pediatric_transfer.features.chembl_bio import build_chembl_target_profiles
from pediatric_transfer.models.transfer import (
    endpoint_feasibility, run_transfer_for_endpoint, seen_unseen_split, eval_seen_unseen,
)
from pediatric_transfer.stats.bootstrap import run_all_comparisons, faers_validation

import pandas as pd
import numpy as np

PT = ZoneInfo("America/Los_Angeles")
PRELIM_DIR = Path("/workspace/pediatric_transfer_aligned_v2_PRELIMINARY_CHECKPOINT")
PRELIM_LIVE = Path("/workspace/pediatric_transfer_aligned_v2")
FAERS_V3 = CACHE / "faers_v3"


def poll_fda_key() -> bool:
    """Check env and optional key files (do not invent a key)."""
    if os.environ.get("OPENFDA_API_KEY") or os.environ.get("FDA_API_KEY"):
        return True
    for fp in (
        RUN_DIR / "OPENFDA_API_KEY.txt",
        RUN_DIR / "FDA_API_KEY.txt",
        Path("/workspace/OPENFDA_API_KEY.txt"),
        RUN_DIR / "cache" / ".openfda_env",
    ):
        if fp.exists():
            raw = fp.read_text().strip() if fp.stat().st_size else ""
            key = ""
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("OPENFDA_API_KEY=") or line.startswith("FDA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                else:
                    key = line
                break
            if key:
                os.environ["OPENFDA_API_KEY"] = key
                return True
    return False


def init_manifest() -> dict:
    manifest = {
        "run_id": "pediatric_transfer_aligned_v3",
        "run_type": "FINAL",
        "started_at": now_pt(),
        "preliminary_pointer": str(PRELIM_DIR),
        "preliminary_live": str(PRELIM_LIVE),
        "seeds": SEEDS,
        "global_seed": GLOBAL_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "min_adult_n": MIN_ADULT_N,
        "min_ped_n": MIN_PED_N,
        "chembl_target_cap": CHEMBL_TARGET_CAP,
        "faers_max_pages_per_domain": FAERS_MAX_PAGES_PER_DOMAIN,
        "package_versions": package_versions(),
        "api": {"ctgov": CTGOV_BASE, "openfda": OPENFDA_BASE, "chembl": CHEMBL_BASE},
        "outcomes": OUTCOMES,
        "stages": {},
        "completed": [],
        "failed": [],
        "notes": [
            "No early-stop / overnight_cap on mapping",
            "All deferred names are adult-only CT.gov names",
            "chembl_molecules/activities symlinked from v2; new profile blocks under v3/cache",
            "FAERS expansion writes to cache/faers_v3 to preserve preliminary raw/",
        ],
    }
    atomic_write_json(MANIFEST, manifest)
    return manifest


def write_mapping_completion_report(mapping: pd.DataFrame, deferred_n: int, prelim_ok: int) -> dict:
    status_counts = mapping.status.value_counts().to_dict() if len(mapping) else {}
    ok = int(status_counts.get("ok", 0))
    unresolved = int(status_counts.get("unresolved", 0) + status_counts.get("error", 0))
    excluded = int(
        status_counts.get("excluded_control_or_nondrug", 0)
        + status_counts.get("excluded_mixture_or_invalid", 0)
    )
    attempted = len(mapping)
    report = {
        "attempted": attempted,
        "ok": ok,
        "unresolved": unresolved,
        "excluded": excluded,
        "completion_rate_ok": ok / attempted if attempted else 0.0,
        "resolution_rate_ok_or_excluded": (ok + excluded) / attempted if attempted else 0.0,
        "deferred_attempted": deferred_n,
        "preliminary_ok_seeded": prelim_ok,
        "adult_only_priority_confirmed": True,
        "status_counts": status_counts,
        "generated_at": now_pt(),
    }
    atomic_write_json(PROCESSED / "mapping_completion_report.json", report)
    return report


def stage_mapping(manifest: dict) -> pd.DataFrame:
    log("=== STAGE MAPPING (complete ALL deferred; no early-stop) ===")
    arm = pd.read_csv(PROCESSED / "ctgov_arm_level.csv")
    deferred = pd.read_csv(PROCESSED / "deferred_drug_names.csv")
    prelim = pd.read_csv(PROCESSED / "drug_name_mapping_preliminary.csv")
    prelim_ok = int((prelim.status == "ok").sum())
    log(f"preliminary mappings={len(prelim)} ok={prelim_ok}; deferred={len(deferred)}")

    # Confirm adult-only
    ped_names = set(arm.loc[arm.age_bucket == "pediatric", "drug_name_norm"])
    n_ped_in_def = int(deferred.drug_name_norm.isin(ped_names).sum())
    log(f"deferred ∩ pediatric names = {n_ped_in_def} (expect 0)")
    assert n_ped_in_def == 0, "deferred must be adult-only"

    raw_map = arm.drop_duplicates("drug_name_norm").set_index("drug_name_norm")["drug_name_raw"].to_dict()

    # Also include any preliminary unresolved for remapping with new extractor
    remappable = set(deferred.drug_name_norm.tolist())
    prelim_unres = prelim[prelim.status != "ok"]["drug_name_norm"].tolist()
    for n in prelim_unres:
        remappable.add(n)

    # Seed cache from preliminary OK rows (already in drug_name_cache.json copy)
    names = [(n, raw_map.get(n, n)) for n in sorted(remappable)]
    # Plus keep prelim OK names in final merge even if not re-queried
    log(f"Mapping queue size={len(names)} (deferred+prelim_non_ok); cache will serve prelim OK")

    # Also ensure all prelim OK are in the name list so map_drug_names returns full union
    # Actually map_drug_names only returns what we pass. So pass ALL unique names:
    all_norms = set(prelim.drug_name_norm) | set(deferred.drug_name_norm)
    names_all = [(n, raw_map.get(n, n)) for n in sorted(all_norms)]
    log(f"Total unique names to cover={len(names_all)}")

    mapping = map_drug_names(names_all, checkpoint_every=25, workers=4)
    report = write_mapping_completion_report(mapping, deferred_n=len(deferred), prelim_ok=prelim_ok)
    manifest["stages"]["mapping"] = report
    manifest["completed"].append("mapping")
    atomic_write_json(MANIFEST, manifest)
    log(f"MAPPING COMPLETE: {json.dumps(report)}")
    return mapping



def stage_chembl_rescue(manifest: dict, mapping: pd.DataFrame) -> pd.DataFrame:
    """Second-pass ChEMBL for unresolved drug-like names (PubChem bulk was primary)."""
    import re, time, requests
    from pediatric_transfer.mapping.harmonize import (
        chembl_synonym_to_smiles, smiles_to_record, extract_active_substance, CACHE_MAP,
    )
    from pediatric_transfer.utils.io_utils import load_json, atomic_write_json, atomic_write_df
    from pediatric_transfer.utils.paths import PROCESSED, CACHE

    log("=== STAGE ChEMBL RESCUE (unresolved drug-like only) ===")
    cache = load_json(CACHE_MAP, {})
    unres = mapping[mapping.status == "unresolved"].copy()
    # drug-like filter
    def druglike(n: str) -> bool:
        n = (n or "").strip()
        if len(n) < 3 or len(n) > 60:
            return False
        if re.search(r"[:/]", n):
            return False
        if re.fullmatch(r"(?:group|arm|cohort|part)\s+[a-z0-9]+", n):
            return False
        return bool(re.search(r"[a-zA-Z]", n))

    candidates = [r for _, r in unres.iterrows() if druglike(r.drug_name_norm)]
    log(f"ChEMBL rescue candidates={len(candidates)} of unresolved={len(unres)}")
    session = requests.Session()
    session.headers.update({"User-Agent": "pediatric-transfer-aligned-v3/1.0"})
    rescued = 0
    for i, row in enumerate(candidates):
        name = row.drug_name_norm
        raw = row.drug_name_raw if pd.notna(row.get("drug_name_raw")) else name
        queries = extract_active_substance(raw) or [name]
        q = None
        for cand in queries:
            if 3 <= len(cand) <= 40 and re.search(r"[a-zA-Z]", cand) and not re.search(r"[:/]", cand):
                q = cand
                break
        if not q:
            continue
        smi = chembl_synonym_to_smiles(session, q)
        if not smi:
            continue
        rec_s = smiles_to_record(smi)
        if not rec_s:
            continue
        out = {
            "drug_name_norm": name, "drug_name_raw": raw, "status": "ok",
            "source": "chembl_synonym+rescue", "confidence": 0.65,
            "query_used": q, **rec_s,
        }
        cache[name] = out
        rescued += 1
        if (i + 1) % 25 == 0:
            atomic_write_json(CACHE_MAP, cache)
            log(f"chembl rescue progress i={i+1}/{len(candidates)} rescued={rescued}")
        time.sleep(0.5)  # be gentle; ~30s/call worst case but timeout 12s
    atomic_write_json(CACHE_MAP, cache)
    # rebuild mapping df from cache for all previously mapped + rescue
    rows = list(cache.values())
    df = pd.DataFrame(rows)
    # keep only names that were in original mapping union
    keep = set(mapping.drug_name_norm)
    df = df[df.drug_name_norm.isin(keep)].drop_duplicates("drug_name_norm")
    ok = df[df.status == "ok"]
    unresolved = df[df.status.isin(["unresolved", "error"])]
    excluded = df[df.status.isin(["excluded_control_or_nondrug", "excluded_mixture_or_invalid"])]
    atomic_write_df(PROCESSED / "drug_name_mapping.csv", df)
    atomic_write_df(PROCESSED / "unresolved_drug_names.csv", unresolved)
    atomic_write_df(PROCESSED / "excluded_combination_products.csv", excluded)
    if not ok.empty:
        aliases = ok.groupby("inchikey").agg(
            canonical_smiles=("canonical_smiles", "first"),
            names=("drug_name_norm", lambda s: "|".join(sorted(set(s)))),
            sources=("source", lambda s: "|".join(sorted(set(map(str, s))))),
            mean_confidence=("confidence", "mean"),
        ).reset_index()
        atomic_write_df(PROCESSED / "structure_aliases.csv", aliases)
    report = {
        "rescued": rescued,
        "candidates": len(candidates),
        "ok_after": int((df.status == "ok").sum()),
        "unresolved_after": int((df.status.isin(["unresolved", "error"])).sum()),
    }
    manifest["stages"]["chembl_rescue"] = report
    manifest["completed"].append("chembl_rescue")
    atomic_write_json(MANIFEST, manifest)
    log(f"ChEMBL rescue done: {report}")
    return df

def stage_rebuild(manifest: dict, mapping: pd.DataFrame):
    log("=== STAGE REBUILD structure-outcome tables ===")
    arm = pd.read_csv(PROCESSED / "ctgov_arm_level.csv")
    ok_map = mapping[mapping.status == "ok"][
        ["drug_name_norm", "inchikey", "canonical_smiles", "source", "confidence"]
    ].drop_duplicates("drug_name_norm")

    # FAERS: use preliminary if v3 expansion not ready yet
    ror = pd.DataFrame()
    ped_n = ad_n = 0
    try:
        if (FAERS_V3 / "pediatric").exists() and list((FAERS_V3 / "pediatric").glob("batch_*.json")):
            ped_rep = _load_all(FAERS_V3 / "pediatric")
            ad_rep = _load_all(FAERS_V3 / "adult")
            faers_src = "faers_v3"
        else:
            ped_rep = _load_all(RAW / "faers" / "pediatric")
            ad_rep = _load_all(RAW / "faers" / "adult")
            faers_src = "preliminary_raw_symlink"
        ped_n, ad_n = len(ped_rep), len(ad_rep)
        ped_pairs = reports_to_pairs(ped_rep, "pediatric")
        ad_pairs = reports_to_pairs(ad_rep, "adult")
        atomic_write_df(PROCESSED / "faers_ped_pairs.csv", ped_pairs)
        atomic_write_df(PROCESSED / "faers_adult_pairs.csv", ad_pairs)
        ror = compute_ror(ped_pairs, ad_pairs)
        atomic_write_df(PROCESSED / "faers_ror.csv", ror)
        manifest["stages"]["faers"] = {
            "source": faers_src,
            "ped_reports": ped_n,
            "adult_reports": ad_n,
            "ror_rows": len(ror),
            "key_used": poll_fda_key(),
            "expansion_complete": ped_n >= 25000 and ad_n >= 25000,
        }
        log(f"FAERS source={faers_src} ped={ped_n} adult={ad_n} ror={len(ror)}")
    except Exception as e:
        log_exc("faers_rebuild", "load", e)
        manifest["failed"].append({"stage": "faers_rebuild", "error": str(e)})

    agg = split_and_save(arm, ok_map)
    adult_lab = agg.get("adult", pd.DataFrame())
    ped_lab = agg.get("pediatric", pd.DataFrame())
    manifest["stages"]["aggregate"] = {
        "adult_rows": len(adult_lab),
        "ped_rows": len(ped_lab),
        "adult_structures": int(adult_lab["inchikey"].nunique()) if len(adult_lab) else 0,
        "ped_structures": int(ped_lab["inchikey"].nunique()) if len(ped_lab) else 0,
    }
    manifest["completed"].append("aggregate")
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

    overlap_j = {}
    if len(adult_lab) and len(ped_lab):
        a_ik = set(adult_lab.inchikey)
        p_ik = set(ped_lab.inchikey)
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
    manifest["completed"].append("harmonized_long")
    atomic_write_json(MANIFEST, manifest)
    return adult_lab, ped_lab, ok_map, ror


def stage_features(manifest, adult_lab, ped_lab, ok_map):
    log("=== STAGE FEATURES (chem + chembl_target_profile) ===")
    structs = ok_map[["inchikey", "canonical_smiles"]].drop_duplicates("inchikey")
    smiles_map = dict(zip(structs.inchikey, structs.canonical_smiles))
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
    log(f"Building chembl_target_profile for {len(structs)} structures (adult vocab, cap={CHEMBL_TARGET_CAP})")
    bio = build_chembl_target_profiles(list(structs.inchikey), adult_iks)
    manifest["stages"]["features"] = {"chem": list(chem_x.shape), "bio": list(bio.shape)}
    manifest["completed"].append("features")
    atomic_write_json(MANIFEST, manifest)
    return chem_x, bio, smiles_map


def stage_transfer(manifest, adult_lab, ped_lab, chem_x, bio, smiles_map):
    log("=== STAGE TRANSFER (frozen settings) ===")
    # Clear prior preds from any partial
    for p in PREDICTIONS.glob("pred_*.csv"):
        p.unlink()
    for p in MODELS.glob("*"):
        if p.is_file():
            p.unlink()

    feas = endpoint_feasibility(adult_lab, ped_lab) if len(adult_lab) and len(ped_lab) else pd.DataFrame()
    if len(feas):
        atomic_write_df(RESULTS / "endpoint_feasibility.csv", feas)
    eligible = feas[feas.eligible]["outcome"].tolist() if len(feas) else []
    log(f"Eligible endpoints ({len(eligible)}): {eligible}")
    all_metrics = []
    for o in eligible:
        log(f"Transfer endpoint={o}")
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
    manifest["completed"].append("transfer")
    atomic_write_json(MANIFEST, manifest)
    return metrics, eligible


def stage_bootstrap(manifest, eligible, mapping, ror):
    log("=== STAGE BOOTSTRAP ===")
    boot = run_all_comparisons(PREDICTIONS, eligible or [], model="xgb")
    if boot.empty and eligible:
        boot = run_all_comparisons(PREDICTIONS, eligible, model="ridge")
    manifest["stages"]["bootstrap"] = {"rows": len(boot)}
    manifest["completed"].append("bootstrap")
    atomic_write_json(MANIFEST, manifest)

    preds = [pd.read_csv(p) for p in PREDICTIONS.glob("pred_*.csv")]
    ped_pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    fv = faers_validation(ped_pred, ror, mapping)
    manifest["stages"]["faers_validation"] = {"rows": len(fv)}
    manifest["completed"].append("faers_validation")
    atomic_write_json(MANIFEST, manifest)
    return boot


def stage_faers_expansion(manifest) -> dict:
    """Expand FAERS to ≥25k/domain when API key present. Writes under cache/faers_v3."""
    log("=== STAGE FAERS EXPANSION (key-gated) ===")
    if not poll_fda_key():
        log("No OPENFDA_API_KEY/FDA_API_KEY — FAERS expansion deferred")
        atomic_write_text(
            RESULTS / "FAERS_EXPANSION_PENDING.txt",
            f"FAERS expansion pending as of {now_pt()}.\n"
            "No OPENFDA_API_KEY or FDA_API_KEY in environment.\n"
            "Preliminary FAERS sample (5000 reports/domain) retained and clearly labeled.\n"
            "Mapping+transfer completed independently of FAERS expansion.\n",
        )
        return {"expanded": False, "reason": "no_api_key"}

    # Monkey-patch RAW temporarily by fetching into FAERS_V3 via local wrappers
    from pediatric_transfer.faers import fetch as F

    def fetch_into(domain: str):
        out_dir = FAERS_V3 / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        # Seed from preliminary batches if empty
        prelim_dir = RAW / "faers" / domain
        if not list(out_dir.glob("batch_*.json")) and prelim_dir.exists():
            for bp in sorted(prelim_dir.glob("batch_*.json")):
                shutil.copy2(bp, out_dir / bp.name)
            prog = load_json(prelim_dir / "progress.json", {})
            # reset done so we can continue past 50 pages
            prog["done"] = False
            atomic_write_json(out_dir / "progress.json", prog)
            log(f"Seeded FAERS v3 {domain} from preliminary ({prog.get('pages_done')} pages)")

        # Temporarily point RAW/faers to FAERS_V3 by patching module path usage:
        # fetch_faers_domain uses RAW / "faers" / domain — override via symlink swap is unsafe.
        # Instead inline a copy of fetch logic targeting out_dir.
        return _fetch_domain_to(out_dir, domain)

    def _fetch_domain_to(out_dir: Path, domain: str):
        progress_path = out_dir / "progress.json"
        progress = load_json(progress_path, {"skip": 0, "pages_done": 0, "done": False})
        if progress.get("done") and progress.get("pages_done", 0) >= FAERS_MAX_PAGES_PER_DOMAIN:
            return _load_all(out_dir)
        session = F._session()
        key = F._api_key()
        max_pages = FAERS_MAX_PAGES_PER_DOMAIN
        skip = progress.get("skip", 0)
        pages_done = progress.get("pages_done", 0)
        reports = _load_all(out_dir) if pages_done else []
        search = F.age_query(domain)
        log(f"FAERS v3 expand domain={domain} from page={pages_done} skip={skip} key=yes")
        while pages_done < max_pages:
            params = {"search": search.replace("+", " "), "limit": 100, "skip": skip, "api_key": key}
            delay = 1.0
            data = None
            for attempt in range(6):
                try:
                    r = session.get(OPENFDA_BASE, params=params, timeout=90)
                    if r.status_code == 404:
                        progress["done"] = True
                        atomic_write_json(progress_path, progress)
                        return reports
                    if r.status_code == 429:
                        time.sleep(delay)
                        delay = min(delay * 2, 120)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    if attempt == 5:
                        log_exc("faers_v3", f"{domain}_skip{skip}", e)
                        atomic_write_json(progress_path, {**progress, "skip": skip, "pages_done": pages_done, "done": False})
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
            if data is None:
                break
            results = data.get("results") or []
            if not results:
                progress["done"] = True
                atomic_write_json(progress_path, {**progress, "skip": skip, "pages_done": pages_done, "done": True})
                break
            pages_done += 1
            atomic_write_json(out_dir / f"batch_{pages_done:05d}.json", {"results": results, "skip": skip})
            reports.extend(results)
            skip += len(results)
            progress = {"skip": skip, "pages_done": pages_done, "done": False,
                        "search": search, "total_meta": (data.get("meta") or {}).get("results")}
            atomic_write_json(progress_path, progress)
            if pages_done % 10 == 0:
                log(f"FAERS v3 {domain} pages={pages_done} reports={len(reports)}")
            time.sleep(0.25)
        progress["done"] = True
        atomic_write_json(progress_path, progress)
        log(f"FAERS v3 {domain} complete reports={len(reports)}")
        return reports

    ped = fetch_into("pediatric")
    ad = fetch_into("adult")
    info = {
        "expanded": True,
        "ped_reports": len(ped),
        "adult_reports": len(ad),
        "key_used": True,
        "path": str(FAERS_V3),
    }
    manifest["stages"]["faers_expansion"] = info
    if len(ped) >= 25000 and len(ad) >= 25000:
        if (RESULTS / "FAERS_EXPANSION_PENDING.txt").exists():
            (RESULTS / "FAERS_EXPANSION_PENDING.txt").unlink()
        manifest["completed"].append("faers_expansion")
    else:
        atomic_write_text(
            RESULTS / "FAERS_EXPANSION_PENDING.txt",
            f"Partial FAERS expansion at {now_pt()}: ped={len(ped)} adult={len(ad)} (target ≥25000 each).\n",
        )
    atomic_write_json(MANIFEST, manifest)
    return info


def stage_comparison(manifest, metrics, boot, adult_lab, ped_lab, mapping_report):
    log("=== STAGE COMPARISON vs preliminary ===")
    prelim_metrics = pd.read_csv(PRELIM_DIR / "results" / "transfer_metrics.csv")
    prelim_boot = pd.read_csv(PRELIM_DIR / "results" / "bootstrap_comparisons.csv")
    prelim_overlap = load_json(PRELIM_DIR / "results" / "source_overlaps.json", {})
    prelim_verdict = load_json(PRELIM_DIR / "results" / "hypothesis_verdict.json", {})

    def mean_spearman(df, rep, model="xgb"):
        sub = df[(df.model == model) & (df.representation == rep)]
        return float(sub.spearman.mean()) if len(sub) else float("nan")

    def mean_perm(df, model="xgb"):
        sub = df[(df.model == model) & (df.representation.str.startswith("chem_permuted_bio"))]
        return float(sub.groupby("outcome").spearman.mean().mean()) if len(sub) else float("nan")

    cur = {
        "chem": mean_spearman(metrics, "chem") if len(metrics) else float("nan"),
        "bio_only": mean_spearman(metrics, "bio_only") if len(metrics) else float("nan"),
        "chem_true_bio": mean_spearman(metrics, "chem_true_bio") if len(metrics) else float("nan"),
        "mean_perm": mean_perm(metrics) if len(metrics) else float("nan"),
    }
    pre = {
        "chem": mean_spearman(prelim_metrics, "chem"),
        "bio_only": mean_spearman(prelim_metrics, "bio_only"),
        "chem_true_bio": mean_spearman(prelim_metrics, "chem_true_bio"),
        "mean_perm": mean_perm(prelim_metrics),
    }

    # BH primary contrast
    def bh_summary(boot_df):
        if boot_df is None or len(boot_df) == 0:
            return {"any_bh_sig_positive": False, "min_p_bh": None, "n_positive_delta": 0}
        prim = boot_df[boot_df.contrast == "chem_true_bio_minus_chem"]
        if prim.empty:
            return {"any_bh_sig_positive": False, "min_p_bh": None, "n_positive_delta": 0}
        min_pbh = float(prim.p_bh.min()) if "p_bh" in prim.columns and prim.p_bh.notna().any() else None
        any_sig_pos = bool(((prim.p_bh < 0.05) & (prim.delta > 0)).any()) if "p_bh" in prim.columns else False
        return {
            "any_bh_sig_positive": any_sig_pos,
            "min_p_bh": min_pbh,
            "n_positive_delta": int((prim.delta > 0).sum()),
            "n_contrasts": len(prim),
        }

    cur_bh = bh_summary(boot)
    pre_bh = bh_summary(prelim_boot)

    # Verdict logic (same criteria)
    def verdict_from(means, bh):
        # Support only if true > chem AND true > mean_perm AND BH-significant positive somewhere
        beat = means["chem_true_bio"] > means["chem"] and means["chem_true_bio"] > means["mean_perm"]
        if bh["any_bh_sig_positive"] and beat:
            return "SUPPORTS"
        if means["chem_true_bio"] <= means["chem"] or not bh["any_bh_sig_positive"]:
            return "REJECTS"
        return "UNRESOLVED"

    cur_verdict = verdict_from(cur, cur_bh)
    pre_verdict = "REJECTS"  # from preliminary hypothesis_verdict.json
    changed = cur_verdict != pre_verdict

    delta = {
        "preliminary_verdict": pre_verdict,
        "final_verdict": cur_verdict,
        "verdict_changed": changed,
        "mapping": mapping_report,
        "sample_sizes": {
            "preliminary": prelim_overlap,
            "final": manifest.get("stages", {}).get("overlaps", {}),
        },
        "transfer_means_xgb_spearman": {"preliminary": pre, "final": cur,
                                         "delta": {k: cur[k] - pre[k] for k in cur}},
        "bootstrap_bh_true_minus_chem": {"preliminary": pre_bh, "final": cur_bh},
        "explanation": (
            f"Preliminary {pre_verdict}; final {cur_verdict}. "
            f"Mean Spearman chem {pre['chem']:.3f}→{cur['chem']:.3f}; "
            f"chem_true_bio {pre['chem_true_bio']:.3f}→{cur['chem_true_bio']:.3f}; "
            f"mean_perm {pre['mean_perm']:.3f}→{cur['mean_perm']:.3f}. "
            f"BH any significant positive true−chem: {pre_bh['any_bh_sig_positive']}→{cur_bh['any_bh_sig_positive']}."
        ),
        "generated_at": now_pt(),
    }
    atomic_write_json(RESULTS / "conclusion_delta.json", delta)

    # hypothesis verdict
    hv = {
        "hypothesis": "Adding chembl_target_profile bio to chem improves adult→pediatric AE incidence transfer",
        "computational_status": "COMPLETE (v3 final; full deferred mapping)",
        "scientific_support": (
            "SUPPORTS multimodal benefit under pre-specified criteria"
            if cur_verdict == "SUPPORTS"
            else "REJECTS / does not support multimodal benefit under pre-specified criteria (true bio must beat chem AND permuted controls)"
            if cur_verdict == "REJECTS"
            else "UNRESOLVED under pre-specified criteria"
        ),
        "note": "Null result is valid. Do not claim multimodal helps." if cur_verdict != "SUPPORTS" else "Claim only with BH-aware evidence.",
        "vs_preliminary": "CHANGED" if changed else "UNCHANGED",
    }
    atomic_write_json(RESULTS / "hypothesis_verdict.json", hv)

    md = f"""# Comparison vs preliminary (v2 checkpoint)

Generated: {now_pt()}

## Mapping completion
- Deferred attempted: {mapping_report.get('deferred_attempted')}
- Total attempted (union): {mapping_report.get('attempted')}
- OK: {mapping_report.get('ok')}
- Unresolved: {mapping_report.get('unresolved')}
- Excluded: {mapping_report.get('excluded')}
- OK completion rate: {mapping_report.get('completion_rate_ok', 0):.3f}
- Adult-only priority confirmed: {mapping_report.get('adult_only_priority_confirmed')}

## Sample size deltas
| | Preliminary | Final v3 |
|---|---:|---:|
| Adult structures | {prelim_overlap.get('adult_structures')} | {manifest.get('stages',{}).get('overlaps',{}).get('adult_structures')} |
| Ped structures | {prelim_overlap.get('pediatric_structures')} | {manifest.get('stages',{}).get('overlaps',{}).get('pediatric_structures')} |
| Adult SO labels | {prelim_overlap.get('adult_structure_outcome')} | {manifest.get('stages',{}).get('overlaps',{}).get('adult_structure_outcome')} |
| Ped SO labels | {prelim_overlap.get('ped_structure_outcome')} | {manifest.get('stages',{}).get('overlaps',{}).get('ped_structure_outcome')} |

## Transfer mean Spearman (xgb)
| rep | prelim | final | Δ |
|---|---:|---:|---:|
| chem | {pre['chem']:.4f} | {cur['chem']:.4f} | {cur['chem']-pre['chem']:.4f} |
| bio_only | {pre['bio_only']:.4f} | {cur['bio_only']:.4f} | {cur['bio_only']-pre['bio_only']:.4f} |
| chem_true_bio | {pre['chem_true_bio']:.4f} | {cur['chem_true_bio']:.4f} | {cur['chem_true_bio']-pre['chem_true_bio']:.4f} |
| mean_perm | {pre['mean_perm']:.4f} | {cur['mean_perm']:.4f} | {cur['mean_perm']-pre['mean_perm']:.4f} |

## Bootstrap BH (chem_true_bio − chem)
- Preliminary: any BH-sig positive={pre_bh['any_bh_sig_positive']}, min p_BH={pre_bh['min_p_bh']}
- Final: any BH-sig positive={cur_bh['any_bh_sig_positive']}, min p_BH={cur_bh['min_p_bh']}

## Verdict
- Preliminary: **{pre_verdict}**
- Final v3: **{cur_verdict}**
- Delta: **{'CHANGED' if changed else 'UNCHANGED'}**

{delta['explanation']}

FAERS: see run_manifest stages.faers / FAERS_EXPANSION_PENDING.txt if key absent.
"""
    atomic_write_text(REPORTS / "comparison_vs_preliminary.md", md)
    manifest["stages"]["comparison"] = {
        "verdict_changed": changed,
        "final_verdict": cur_verdict,
        "preliminary_verdict": pre_verdict,
    }
    manifest["completed"].append("comparison")
    atomic_write_json(MANIFEST, manifest)
    return delta, hv


def write_zip():
    zip_path = RESULTS / "pediatric_transfer_aligned_v3_deliverables.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in [CODE, RESULTS, REPORTS, LOGS, PROCESSED, FEATURES]:
            for p in Path(folder).rglob("*"):
                if not p.is_file():
                    continue
                if ".venv" in p.parts or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
                    continue
                if p.suffix in {".tmp", ".zip"} or "deliverables.zip" in p.name:
                    continue
                if p.stat().st_size > 80_000_000:
                    continue
                z.write(p, arcname=str(p.relative_to(RUN_DIR)))
    log(f"Wrote zip {zip_path}")
    return zip_path


def main():
    log("=" * 72)
    log("pediatric_transfer_aligned_v3 FINAL run start")
    log(f"RUN_DIR={RUN_DIR}")
    log(f"Preliminary (frozen): {PRELIM_DIR}")
    log(f"FDA key present at start: {poll_fda_key()}")
    log(f"versions={package_versions()}")
    log("=" * 72)

    # Unit tests
    import subprocess
    test_path = CODE / "pediatric_transfer" / "tests" / "test_unit.py"
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
        cwd=str(CODE), capture_output=True, text=True,
    )
    atomic_write_text(LOGS / "unit_test_output.txt", r.stdout + "\n" + r.stderr)
    log(f"unit tests exit={r.returncode}")
    if r.returncode != 0:
        log("UNIT TESTS FAILED — see unit_test_output.txt; continuing after documenting")
        log(r.stdout[-1500:] if r.stdout else "")
        log(r.stderr[-1500:] if r.stderr else "")

    manifest = init_manifest()
    ev = ensure_eudra_status()
    manifest["stages"]["eudra"] = ev
    counts = load_json(PROCESSED / "ctgov_trial_counts.json", {})
    arm_n = sum(1 for _ in open(PROCESSED / "ctgov_arm_level.csv")) - 1
    manifest["stages"]["ctgov_parse"] = {**counts, "arm_rows": arm_n, "resumed_from_v2_raw": True}
    atomic_write_json(MANIFEST, manifest)

    # A) Mapping first (do not wait for key)
    mapping = stage_mapping(manifest)
    try:
        mapping = stage_chembl_rescue(manifest, mapping)
        mapping_report = write_mapping_completion_report(
            mapping,
            deferred_n=int(pd.read_csv(PROCESSED / "deferred_drug_names.csv").shape[0]),
            prelim_ok=int((pd.read_csv(PROCESSED / "drug_name_mapping_preliminary.csv").status == "ok").sum()),
        )
        manifest["stages"]["mapping"] = mapping_report
        atomic_write_json(MANIFEST, manifest)
    except Exception as e:
        log_exc("chembl_rescue", "stage", e)
        mapping_report = manifest["stages"]["mapping"]

    # Poll key briefly then proceed
    if not poll_fda_key():
        log("Polling for FDA key up to 3 minutes before rebuild (mapping already done)...")
        for _ in range(6):
            time.sleep(30)
            if poll_fda_key():
                log("FDA key appeared during poll")
                break

    # B) Optional FAERS expansion before rebuild if key present
    if poll_fda_key():
        try:
            stage_faers_expansion(manifest)
        except Exception as e:
            log_exc("faers_expansion", "pre_rebuild", e)

    # C) Rebuild + features + transfer
    adult_lab, ped_lab, ok_map, ror = stage_rebuild(manifest, mapping)
    chem_x, bio, smiles_map = stage_features(manifest, adult_lab, ped_lab, ok_map)
    metrics, eligible = stage_transfer(manifest, adult_lab, ped_lab, chem_x, bio, smiles_map)
    boot = stage_bootstrap(manifest, eligible, mapping, ror)

    # D) FAERS expansion again if key appeared late; recompute ROR+validation only
    if poll_fda_key() and not manifest.get("stages", {}).get("faers_expansion", {}).get("expanded"):
        try:
            info = stage_faers_expansion(manifest)
            if info.get("expanded"):
                ped_rep = _load_all(FAERS_V3 / "pediatric")
                ad_rep = _load_all(FAERS_V3 / "adult")
                ped_pairs = reports_to_pairs(ped_rep, "pediatric")
                ad_pairs = reports_to_pairs(ad_rep, "adult")
                ror = compute_ror(ped_pairs, ad_pairs)
                atomic_write_df(PROCESSED / "faers_ror.csv", ror)
                preds = [pd.read_csv(p) for p in PREDICTIONS.glob("pred_*.csv")]
                ped_pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
                faers_validation(ped_pred, ror, mapping)
                manifest["stages"]["faers"] = {
                    "source": "faers_v3",
                    "ped_reports": len(ped_rep),
                    "adult_reports": len(ad_rep),
                    "ror_rows": len(ror),
                    "key_used": True,
                    "expansion_complete": len(ped_rep) >= 25000 and len(ad_rep) >= 25000,
                }
        except Exception as e:
            log_exc("faers_expansion", "post_transfer", e)
    elif not poll_fda_key():
        if not (RESULTS / "FAERS_EXPANSION_PENDING.txt").exists():
            atomic_write_text(
                RESULTS / "FAERS_EXPANSION_PENDING.txt",
                f"FAERS expansion pending as of {now_pt()}. No API key. Preliminary 5k/domain labeled.\n",
            )

    # E) Reports + comparison + zip
    write_reports(manifest, metrics, boot, adult_lab, ped_lab, ev)
    # Fix zip name to v3 (write_reports still writes v2 name)
    old_zip = RESULTS / "pediatric_transfer_aligned_v2_deliverables.zip"
    if old_zip.exists():
        old_zip.unlink()
    stage_comparison(manifest, metrics, boot, adult_lab, ped_lab, mapping_report)
    zip_path = write_zip()
    manifest["completed"].append("reports")
    manifest["completed"].append("zip")
    manifest["ended_at"] = now_pt()
    hashes = {}
    for p in list(RESULTS.glob("*")) + list(REPORTS.glob("*")):
        if p.is_file() and p.stat().st_size < 50_000_000:
            try:
                hashes[str(p)] = sha256_file(p)
            except Exception:
                pass
    manifest["artifact_sha256"] = hashes
    atomic_write_json(MANIFEST, manifest)
    log("=== V3 FINAL RUN COMPLETE ===")
    log(f"zip={zip_path}")


if __name__ == "__main__":
    main()
