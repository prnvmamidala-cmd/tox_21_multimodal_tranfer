"""FAERS via openFDA — spontaneous reports, NOT incidence."""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ..utils.config import OPENFDA_BASE, FAERS_PAGE_LIMIT, FAERS_MAX_PAGES_PER_DOMAIN
from ..utils.io_utils import atomic_write_json, load_json
from ..utils.logging_utils import log, log_exc
from ..utils.paths import RAW


def _api_key() -> Optional[str]:
    return os.environ.get("OPENFDA_API_KEY") or os.environ.get("FDA_API_KEY") or None


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "pediatric-transfer-aligned-v2/1.0 (research)"})
    return s


def age_query(domain: str) -> str:
    """Pediatric 0-17; adult 18-100. patient.patientonsetageunit:801 = year."""
    if domain == "pediatric":
        # age in years 0-17 OR months/days/weeks converted loosely via years only for consistency
        return "(patient.patientonsetage:[0+TO+17]+AND+patient.patientonsetageunit:801)"
    return "(patient.patientonsetage:[18+TO+100]+AND+patient.patientonsetageunit:801)"


def fetch_faers_domain(domain: str, max_pages: Optional[int] = None, max_reports: Optional[int] = None) -> List[dict]:
    out_dir = RAW / "faers" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    progress = load_json(progress_path, {"skip": 0, "pages_done": 0, "done": False})
    if progress.get("done"):
        return _load_all(out_dir)

    session = _session()
    key = _api_key()
    max_pages = max_pages if max_pages is not None else FAERS_MAX_PAGES_PER_DOMAIN
    skip = progress.get("skip", 0)
    pages_done = progress.get("pages_done", 0)
    reports: List[dict] = _load_all(out_dir) if pages_done else []

    # Search: primary suspect role in later filter; get recent
    search = age_query(domain)
    log(f"FAERS openFDA domain={domain} search={search} key={'yes' if key else 'no'}")

    while pages_done < max_pages:
        if max_reports is not None and len(reports) >= max_reports:
            break
        params = {
            "search": search.replace("+", " "),
            "limit": FAERS_PAGE_LIMIT,
            "skip": skip,
        }
        if key:
            params["api_key"] = key
        delay = 1.0
        for attempt in range(6):
            try:
                r = session.get(OPENFDA_BASE, params=params, timeout=90)
                if r.status_code == 404:
                    # no more results
                    progress["done"] = True
                    atomic_write_json(progress_path, progress)
                    log(f"FAERS {domain} 404 at skip={skip}; done")
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
                    log_exc("faers_fetch", f"{domain}_skip{skip}", e)
                    atomic_write_json(progress_path, {**progress, "skip": skip, "pages_done": pages_done, "done": False})
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120)
        else:
            break

        results = data.get("results") or []
        if not results:
            progress["done"] = True
            atomic_write_json(progress_path, {**progress, "skip": skip, "pages_done": pages_done, "done": True})
            break
        pages_done += 1
        batch_path = out_dir / f"batch_{pages_done:05d}.json"
        atomic_write_json(batch_path, {"results": results, "skip": skip})
        reports.extend(results)
        skip += len(results)
        progress = {"skip": skip, "pages_done": pages_done, "done": False,
                    "search": search, "total_meta": (data.get("meta") or {}).get("results")}
        atomic_write_json(progress_path, progress)
        if pages_done % 5 == 0:
            log(f"FAERS {domain} pages={pages_done} reports={len(reports)}")
        time.sleep(0.25 if key else 0.85)

    progress["done"] = True
    atomic_write_json(progress_path, progress)
    log(f"FAERS {domain} complete reports={len(reports)}")
    return reports


def _load_all(out_dir: Path) -> List[dict]:
    acc = []
    for p in sorted(out_dir.glob("batch_*.json")):
        acc.extend(load_json(p, {}).get("results") or [])
    return acc
