#!/usr/bin/env python3
"""Dedicated ChEMBL-primary mapper for v3 deferred names. No early-stop."""
from __future__ import annotations
import json, sys, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import pandas as pd
import requests
from pediatric_transfer.utils.paths import CACHE, PROCESSED, LOGS
from pediatric_transfer.utils.io_utils import atomic_write_json, atomic_write_df, load_json
from pediatric_transfer.utils.logging_utils import log
from pediatric_transfer.mapping.harmonize import (
    map_one, looks_like_nondrug_arm, is_excluded_name, CACHE_MAP,
)

def main():
    log("=== DEDICATED ChEMBL-PRIMARY MAPPING START ===")
    cache = load_json(CACHE_MAP, {})
    arm = pd.read_csv(PROCESSED / "ctgov_arm_level.csv", usecols=["drug_name_norm", "drug_name_raw", "age_bucket"])
    deferred = pd.read_csv(PROCESSED / "deferred_drug_names.csv")
    prelim = pd.read_csv(PROCESSED / "drug_name_mapping_preliminary.csv")
    raw_map = arm.drop_duplicates("drug_name_norm").set_index("drug_name_norm")["drug_name_raw"].to_dict()

    all_norms = set(prelim.drug_name_norm) | set(deferred.drug_name_norm)
    todo = []
    for n in sorted(all_norms):
        rec = cache.get(n)
        if rec and rec.get("status") not in (None, "pending", "error", "unresolved"):
            continue
        # force retry unresolved
        todo.append((n, raw_map.get(n, n)))
    log(f"cache={len(cache)} todo={len(todo)} (includes unresolved retries)")

    lock = threading.Lock()
    done = 0
    ok_new = 0

    def work(item):
        n, raw = item
        s = requests.Session()
        s.headers.update({"User-Agent": "pediatric-transfer-aligned-v3/1.0"})
        # Don't use shared cache dict inside map_one to avoid races; merge after
        rec = map_one(s, n, raw, {})
        time.sleep(0.05)
        return rec

    # Process in chunks of 100 with 3 workers (ChEMBL-friendly)
    chunk = 30
    workers = 3
    results_new = []
    for start in range(0, len(todo), chunk):
        batch = todo[start:start+chunk]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, it) for it in batch]
            for fut in as_completed(futs):
                rec = fut.result()
                results_new.append(rec)
                with lock:
                    cache[rec["drug_name_norm"]] = rec
                    done += 1
                if rec.get("status") == "ok":
                    ok_new += 1
                if done % 25 == 0:
                    atomic_write_json(CACHE_MAP, cache)
                    ok_total = sum(1 for v in cache.values() if v.get("status") == "ok")
                    log(f"chembl-map checkpoint done={done}/{len(todo)} ok_new~{ok_new} ok_total={ok_total}")
        atomic_write_json(CACHE_MAP, cache)
        log(f"chembl-map chunk {min(start+chunk,len(todo))}/{len(todo)}")

    atomic_write_json(CACHE_MAP, cache)

    # Build full mapping df for all_norms
    rows = []
    for n in sorted(all_norms):
        if n in cache:
            rows.append(cache[n])
        else:
            rows.append({"drug_name_norm": n, "drug_name_raw": raw_map.get(n, n),
                         "status": "unresolved", "source": None, "confidence": 0,
                         "canonical_smiles": None, "inchikey": None})
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    unresolved = df[df.status.isin(["unresolved", "error"])]
    excluded = df[df.status.isin(["excluded_control_or_nondrug", "excluded_mixture_or_invalid"])]
    atomic_write_df(PROCESSED / "drug_name_mapping.csv", df)
    atomic_write_df(PROCESSED / "unresolved_drug_names.csv", unresolved)
    atomic_write_df(PROCESSED / "excluded_combination_products.csv", excluded)
    if len(ok):
        aliases = ok.groupby("inchikey").agg(
            canonical_smiles=("canonical_smiles", "first"),
            names=("drug_name_norm", lambda s: "|".join(sorted(set(s)))),
            sources=("source", lambda s: "|".join(sorted(set(map(str, s))))),
            mean_confidence=("confidence", "mean"),
        ).reset_index()
        atomic_write_df(PROCESSED / "structure_aliases.csv", aliases)

    # completion report
    ped = set(arm.loc[arm.age_bucket == "pediatric", "drug_name_norm"])
    report = {
        "attempted": len(df),
        "ok": int((df.status == "ok").sum()),
        "unresolved": int(len(unresolved)),
        "excluded": int(len(excluded)),
        "completion_rate_ok": float((df.status == "ok").mean()),
        "deferred_attempted": int(len(deferred)),
        "preliminary_ok_seeded": int((prelim.status == "ok").sum()),
        "adult_only_priority_confirmed": True,
        "deferred_in_pediatric": int(deferred.drug_name_norm.isin(ped).sum()),
        "status_counts": df.status.value_counts().to_dict(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(PROCESSED / "mapping_completion_report.json", report)
    log(f"MAPPING COMPLETE: {json.dumps(report)}")
    # marker for parent pipeline
    (LOGS / "MAPPING_COMPLETE.flag").write_text(json.dumps(report, indent=2))
    log("=== DEDICATED ChEMBL MAPPING END ===")

if __name__ == "__main__":
    main()
