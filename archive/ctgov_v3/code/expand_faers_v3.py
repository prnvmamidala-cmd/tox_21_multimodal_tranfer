#!/usr/bin/env python3
"""Parallel FAERS expansion for v3 — ≥25k ped + ≥25k adult. Does not touch mapping."""
from __future__ import annotations
import os, sys, time, shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# Load key without printing
env_file = Path("/workspace/pediatric_transfer_aligned_v3/cache/.openfda_env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
        else:
            os.environ["OPENFDA_API_KEY"] = line

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from pediatric_transfer.utils.paths import CACHE, RAW, RESULTS, PROCESSED, LOGS
from pediatric_transfer.utils.logging_utils import log, log_exc
from pediatric_transfer.utils.io_utils import atomic_write_json, atomic_write_text, load_json, atomic_write_df
from pediatric_transfer.utils.config import OPENFDA_BASE, FAERS_MAX_PAGES_PER_DOMAIN, FAERS_PAGE_LIMIT
from pediatric_transfer.faers.fetch import _session, age_query, _load_all, _api_key
from pediatric_transfer.faers.ror import reports_to_pairs, compute_ror
import pandas as pd

PT = ZoneInfo("America/Los_Angeles")
FAERS_V3 = CACHE / "faers_v3"
LOG = LOGS / "faers_expansion.log"

def flog(msg: str):
    line = f"[{datetime.now(PT).strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    # also append short note to run.log
    with open(LOGS / "run.log", "a") as f:
        f.write(line + "\n")

def fetch_domain(domain: str):
    out_dir = FAERS_V3 / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    prelim = RAW / "faers" / domain
    if not list(out_dir.glob("batch_*.json")) and prelim.exists():
        for bp in sorted(prelim.glob("batch_*.json")):
            shutil.copy2(bp, out_dir / bp.name)
        prog = load_json(prelim / "progress.json", {})
        prog["done"] = False
        atomic_write_json(out_dir / "progress.json", prog)
        flog(f"Seeded {domain} from preliminary pages={prog.get('pages_done')}")

    progress_path = out_dir / "progress.json"
    progress = load_json(progress_path, {"skip": 0, "pages_done": 0, "done": False})
    if progress.get("done") and progress.get("pages_done", 0) >= FAERS_MAX_PAGES_PER_DOMAIN:
        reports = _load_all(out_dir)
        flog(f"{domain} already complete reports={len(reports)}")
        return reports

    session = _session()
    key = _api_key()
    if not key:
        flog("ERROR: no API key in environment")
        raise SystemExit(2)
    flog(f"FAERS expand domain={domain} key=yes pages_done={progress.get('pages_done')} target_pages={FAERS_MAX_PAGES_PER_DOMAIN}")

    skip = progress.get("skip", 0)
    pages_done = progress.get("pages_done", 0)
    reports = _load_all(out_dir) if pages_done else []
    search = age_query(domain)

    while pages_done < FAERS_MAX_PAGES_PER_DOMAIN:
        params = {"search": search.replace("+", " "), "limit": FAERS_PAGE_LIMIT, "skip": skip, "api_key": key}
        delay = 1.0
        data = None
        for attempt in range(8):
            try:
                r = session.get(OPENFDA_BASE, params=params, timeout=90)
                if r.status_code == 404:
                    progress["done"] = True
                    atomic_write_json(progress_path, progress)
                    flog(f"{domain} 404 at skip={skip}; done reports={len(reports)}")
                    return reports
                if r.status_code == 429:
                    flog(f"{domain} 429 backoff {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 7:
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
        progress = {
            "skip": skip, "pages_done": pages_done, "done": False,
            "search": search, "total_meta": (data.get("meta") or {}).get("results"),
            "n_reports": len(reports),
        }
        atomic_write_json(progress_path, progress)
        if pages_done % 10 == 0 or pages_done <= 55:
            flog(f"FAERS {domain} pages={pages_done} reports={len(reports)}")
        time.sleep(0.2)
    progress["done"] = True
    progress["n_reports"] = len(reports)
    atomic_write_json(progress_path, progress)
    flog(f"FAERS {domain} COMPLETE reports={len(reports)}")
    return reports

def main():
    flog("=== FAERS v3 EXPANSION START ===")
    key = _api_key()
    flog(f"key_present={bool(key)}")
    ped = fetch_domain("pediatric")
    ad = fetch_domain("adult")
    info = {
        "ped_reports": len(ped),
        "adult_reports": len(ad),
        "key_used": True,
        "target_met": len(ped) >= 25000 and len(ad) >= 25000,
        "path": str(FAERS_V3),
        "ended_at": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    atomic_write_json(RESULTS / "faers_expansion_status.json", info)
    flog(f"status={info}")

    # Recompute ROR immediately
    ped_pairs = reports_to_pairs(ped, "pediatric")
    ad_pairs = reports_to_pairs(ad, "adult")
    atomic_write_df(PROCESSED / "faers_ped_pairs.csv", ped_pairs)
    atomic_write_df(PROCESSED / "faers_adult_pairs.csv", ad_pairs)
    ror = compute_ror(ped_pairs, ad_pairs)
    atomic_write_df(PROCESSED / "faers_ror.csv", ror)
    flog(f"ROR rows={len(ror)} ped_pairs={len(ped_pairs)} adult_pairs={len(ad_pairs)}")

    if info["target_met"]:
        pending = RESULTS / "FAERS_EXPANSION_PENDING.txt"
        if pending.exists():
            pending.unlink()
        atomic_write_text(
            RESULTS / "FAERS_EXPANSION_COMPLETE.txt",
            f"FAERS expansion complete at {info['ended_at']}\n"
            f"pediatric_reports={len(ped)}\nadult_reports={len(ad)}\n"
            f"ror_rows={len(ror)}\npath={FAERS_V3}\n",
        )
    else:
        atomic_write_text(
            RESULTS / "FAERS_EXPANSION_PENDING.txt",
            f"Partial FAERS expansion: ped={len(ped)} adult={len(ad)} (target ≥25000 each)\n",
        )
    flog("=== FAERS v3 EXPANSION END ===")

if __name__ == "__main__":
    main()
