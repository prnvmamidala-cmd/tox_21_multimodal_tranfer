#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from run_pipeline import *
import pandas as pd

def main():
    log("=== FINISH STATS + REPORTS ===")
    manifest = load_json(MANIFEST, {}) or {}
    ev = ensure_eudra_status()
    adult_lab = pd.read_csv(PROCESSED / "ctgov_adult_structure_outcome.csv")
    ped_lab = pd.read_csv(PROCESSED / "ctgov_pediatric_structure_outcome.csv")
    mapping = pd.read_csv(PROCESSED / "drug_name_mapping.csv")
    ok = mapping[mapping.status=="ok"]
    smiles_map = dict(zip(ok.inchikey, ok.canonical_smiles))
    ror = pd.read_csv(PROCESSED / "faers_ror.csv") if (PROCESSED/"faers_ror.csv").exists() else pd.DataFrame()
    metrics = pd.read_csv(RESULTS / "transfer_metrics.csv")
    feas = pd.read_csv(RESULTS / "endpoint_feasibility.csv")
    eligible = feas[feas.eligible]["outcome"].tolist()

    seen = seen_unseen_split(adult_lab, ped_lab, smiles_map)
    preds = [pd.read_csv(p) for p in PREDICTIONS.glob("pred_*.csv")]
    ped_pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if len(ped_pred):
        eval_seen_unseen(ped_pred, seen)
        log("seen/unseen performance written")

    log("Bootstrap comparisons (may take a while)...")
    boot = run_all_comparisons(PREDICTIONS, eligible, model="xgb")
    if boot.empty:
        boot = run_all_comparisons(PREDICTIONS, eligible, model="ridge")
    log(f"bootstrap rows={len(boot)}")

    fv = faers_validation(ped_pred, ror, mapping)
    log(f"faers validation rows={len(fv)}")

    manifest.setdefault("stages", {})
    manifest["stages"]["bootstrap"] = {"rows": len(boot)}
    manifest["stages"]["faers_validation"] = {"rows": len(fv)}
    manifest["stages"]["transfer"] = {"eligible": eligible, "n_metric_rows": len(metrics)}
    write_reports(manifest, metrics, boot, adult_lab, ped_lab, ev)
    manifest["ended_at"] = now_pt()
    hashes = {}
    for folder in [RESULTS, REPORTS, PROCESSED]:
        for p in Path(folder).glob("*"):
            if p.is_file() and p.stat().st_size < 80_000_000:
                try: hashes[str(p)] = sha256_file(p)
                except Exception: pass
    manifest["artifact_sha256"] = hashes
    manifest["package_versions"] = package_versions()
    atomic_write_json(MANIFEST, manifest)
    log("=== ALL FINISHED ===")

if __name__ == "__main__":
    main()
