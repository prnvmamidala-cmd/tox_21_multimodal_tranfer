"""ClinicalTrials.gov API v2 fetch with atomic page checkpoints. Stream-friendly (no full in-RAM accumulate)."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

from ..utils.config import CTGOV_BASE, CTGOV_PAGE_SIZE
from ..utils.io_utils import atomic_write_json, load_json
from ..utils.logging_utils import log, log_exc
from ..utils.paths import RAW


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "pediatric-transfer-aligned-v2/1.0 (research; academic)"})
    return s


def build_query(tag: str) -> Dict:
    return {
        "format": "json",
        "pageSize": CTGOV_PAGE_SIZE,
        "query.term": "AREA[StudyType]INTERVENTIONAL AND AREA[HasResults]true",
        "countTotal": "true",
    }


def fetch_studies_page(session: requests.Session, page_token: Optional[str], params: Dict,
                       max_retries: int = 8) -> dict:
    p = dict(params)
    if page_token:
        p["pageToken"] = page_token
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = session.get(CTGOV_BASE, params=p, timeout=120)
            if r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def checkpoint_dir(tag: str) -> Path:
    d = RAW / "ctgov" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_all_studies(tag: str = "all_results", max_pages: Optional[int] = None,
                      resume: bool = True) -> int:
    """Paginate CT.gov; checkpoint every page. Returns pages_done. Does NOT hold all studies in RAM."""
    out_dir = checkpoint_dir(tag)
    progress_path = out_dir / "progress.json"
    progress = load_json(progress_path, {"next_page_token": None, "pages_done": 0, "total_count": None, "done": False})
    if progress.get("done") and resume:
        log(f"CT.gov [{tag}] already complete ({progress['pages_done']} pages)")
        return int(progress["pages_done"])

    session = _session()
    params = build_query(tag)
    page_token = progress.get("next_page_token") if resume else None
    pages_done = progress.get("pages_done", 0) if resume else 0

    log(f"CT.gov fetch start tag={tag} resume_pages={pages_done} token={bool(page_token)}")
    log(f"CT.gov query: {params}")

    while True:
        if max_pages is not None and pages_done >= max_pages:
            log(f"CT.gov hit max_pages={max_pages}")
            break
        try:
            data = fetch_studies_page(session, page_token, params)
        except Exception as e:
            log_exc("ctgov_fetch", f"page_{pages_done+1}", e)
            atomic_write_json(progress_path, {
                "next_page_token": page_token,
                "pages_done": pages_done,
                "total_count": progress.get("total_count"),
                "done": False,
                "last_error": str(e),
            })
            raise

        if progress.get("total_count") is None and "totalCount" in data:
            progress["total_count"] = data["totalCount"]
            log(f"CT.gov totalCount={data['totalCount']}")

        studies = data.get("studies") or []
        pages_done += 1
        page_path = out_dir / f"page_{pages_done:05d}.json"
        atomic_write_json(page_path, {"studies": studies, "page": pages_done})
        # intentionally do NOT accumulate studies in memory

        page_token = data.get("nextPageToken")
        atomic_write_json(progress_path, {
            "next_page_token": page_token,
            "pages_done": pages_done,
            "total_count": progress.get("total_count"),
            "done": not page_token,
            "query": params,
        })
        if pages_done % 5 == 0:
            log(f"CT.gov [{tag}] pages={pages_done} next={bool(page_token)}")
        if not page_token:
            log(f"CT.gov [{tag}] complete pages={pages_done}")
            break
        time.sleep(0.25)
        # free page payload
        del studies, data

    return pages_done


def iter_checkpointed_studies(tag: str = "all_results") -> Iterator[dict]:
    """Yield studies one-at-a-time from checkpoint pages."""
    out_dir = checkpoint_dir(tag)
    for p in sorted(out_dir.glob("page_*.json")):
        data = load_json(p, {})
        for st in data.get("studies") or []:
            yield st
        del data


def _load_checkpointed(out_dir: Path) -> List[dict]:
    """Load all — use only for small smoke tags."""
    studies = []
    for p in sorted(out_dir.glob("page_*.json")):
        data = load_json(p, {})
        studies.extend(data.get("studies") or [])
    return studies


def fetch_smoke(n_adult_target: int = 20, n_ped_target: int = 20) -> Dict[str, List[dict]]:
    from .age import classify_age
    from .parse import extract_eligibility_ages

    session = _session()
    params = build_query("smoke")
    out_dir = checkpoint_dir("smoke")
    progress_path = out_dir / "progress.json"
    progress = load_json(progress_path, {"next_page_token": None, "pages_done": 0, "done": False})
    adult, ped = [], []
    page_token = progress.get("next_page_token")
    pages_done = progress.get("pages_done", 0)
    max_pages = 30

    prior = _load_checkpointed(out_dir)
    for st in prior:
        bucket = _classify_study(st)
        if bucket == "adult":
            adult.append(st)
        elif bucket == "pediatric":
            ped.append(st)

    log(f"Smoke CT.gov resume adult={len(adult)} ped={len(ped)} pages={pages_done}")
    while (len(adult) < n_adult_target or len(ped) < n_ped_target) and pages_done < max_pages:
        data = fetch_studies_page(session, page_token, params)
        studies = data.get("studies") or []
        pages_done += 1
        atomic_write_json(out_dir / f"page_{pages_done:05d}.json", {"studies": studies, "page": pages_done})
        for st in studies:
            bucket = _classify_study(st)
            if bucket == "adult":
                adult.append(st)
            elif bucket == "pediatric":
                ped.append(st)
        page_token = data.get("nextPageToken")
        atomic_write_json(progress_path, {
            "next_page_token": page_token, "pages_done": pages_done,
            "done": False, "adult_n": len(adult), "ped_n": len(ped),
        })
        log(f"Smoke page {pages_done}: adult={len(adult)} ped={len(ped)}")
        if not page_token:
            break
        time.sleep(0.3)

    return {"adult": adult[: max(n_adult_target * 3, len(adult))],
            "pediatric": ped[: max(n_ped_target * 3, len(ped))]}


def _classify_study(study: dict) -> Optional[str]:
    from .parse import extract_eligibility_ages
    from .age import classify_age
    amin, amax = extract_eligibility_ages(study)
    return classify_age(amin, amax)
