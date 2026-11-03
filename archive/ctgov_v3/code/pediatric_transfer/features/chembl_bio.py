"""chembl_target_profile biological features via official ChEMBL web services.

NOT Chemical Checker / BioFeat / signaturizer. Never call those remotes.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from ..utils.config import CHEMBL_BASE, CHEMBL_TARGET_CAP, CHEMBL_MIN_COVERAGE
from ..utils.io_utils import atomic_write_df, atomic_write_json, load_json
from ..utils.logging_utils import log, log_exc
from ..utils.paths import CACHE, FEATURES

MOL_CACHE = CACHE / "chembl_molecules"
ACT_CACHE = CACHE / "chembl_activities"
MOL_CACHE.mkdir(parents=True, exist_ok=True)
ACT_CACHE.mkdir(parents=True, exist_ok=True)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "pediatric-transfer-aligned-v2/1.0", "Accept": "application/json"})
    return s


def inchikey_to_chembl_id(session: requests.Session, inchikey: str) -> Optional[str]:
    cp = MOL_CACHE / f"{inchikey}.json"
    if cp.exists():
        data = load_json(cp, {})
        return data.get("molecule_chembl_id")
    delay = 1.0
    for attempt in range(6):
        try:
            url = f"{CHEMBL_BASE}/molecule/{inchikey}.json"
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                atomic_write_json(cp, {"inchikey": inchikey, "molecule_chembl_id": None})
                return None
            if r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            r.raise_for_status()
            data = r.json()
            mid = data.get("molecule_chembl_id")
            atomic_write_json(cp, {"inchikey": inchikey, "molecule_chembl_id": mid, "pref_name": data.get("pref_name")})
            return mid
        except Exception as e:
            if attempt == 5:
                log_exc("chembl_molecule", inchikey, e)
                return None
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def fetch_activities(session: requests.Session, chembl_id: str) -> List[dict]:
    cp = ACT_CACHE / f"{chembl_id}.json"
    if cp.exists():
        return load_json(cp, {}).get("activities") or []
    activities = []
    offset = 0
    limit = 100
    delay = 1.0
    while True:
        try:
            url = f"{CHEMBL_BASE}/activity.json"
            r = session.get(url, params={
                "molecule_chembl_id": chembl_id,
                "limit": limit,
                "offset": offset,
                "pchembl_value__isnull": "false",
            }, timeout=90)
            if r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 404:
                break
            r.raise_for_status()
            data = r.json()
            page = data.get("activities") or []
            activities.extend(page)
            total = (data.get("page_meta") or {}).get("total_count") or len(activities)
            offset += limit
            if offset >= total or not page:
                break
            time.sleep(0.2)
            delay = 1.0
        except Exception as e:
            log_exc("chembl_activity", chembl_id, e)
            break
    atomic_write_json(cp, {"molecule_chembl_id": chembl_id, "activities": activities})
    return activities


def activities_to_target_profile(activities: List[dict]) -> Dict[str, float]:
    """Prefer pChEMBL; median aggregate per target."""
    by_target: Dict[str, List[float]] = {}
    for a in activities:
        tid = a.get("target_chembl_id")
        if not tid:
            continue
        p = a.get("pchembl_value")
        if p is None:
            continue
        try:
            by_target.setdefault(tid, []).append(float(p))
        except Exception:
            continue
    return {t: float(np.median(vs)) for t, vs in by_target.items()}


def build_chembl_target_profiles(inchikeys: List[str], adult_inchikeys: List[str]) -> pd.DataFrame:
    session = _session()
    profiles = {}
    retry_queue = []
    for i, ik in enumerate(inchikeys):
        mid = inchikey_to_chembl_id(session, ik)
        if not mid:
            profiles[ik] = {}
            continue
        try:
            acts = fetch_activities(session, mid)
            profiles[ik] = activities_to_target_profile(acts)
        except Exception as e:
            log_exc("chembl_profile", ik, e)
            retry_queue.append(ik)
            profiles[ik] = {}
        if (i + 1) % 10 == 0:
            log(f"chembl_target_profile molecules={i+1}/{len(inchikeys)}")
            # checkpoint block
            _save_block(profiles, i)
        time.sleep(0.15)

    # retry queue once
    for ik in retry_queue:
        mid = inchikey_to_chembl_id(session, ik)
        if not mid:
            continue
        try:
            acts = fetch_activities(session, mid)
            profiles[ik] = activities_to_target_profile(acts)
        except Exception as e:
            log_exc("chembl_profile_retry", ik, e)

    # Target vocab from adult-only coverage
    adult_set = set(adult_inchikeys)
    target_counts: Dict[str, int] = {}
    for ik, prof in profiles.items():
        if ik not in adult_set:
            continue
        for t in prof:
            target_counts[t] = target_counts.get(t, 0) + 1
    n_adult = max(len(adult_set), 1)
    eligible = [(t, c) for t, c in target_counts.items() if c / n_adult >= CHEMBL_MIN_COVERAGE]
    eligible.sort(key=lambda x: -x[1])
    vocab = [t for t, _ in eligible[:CHEMBL_TARGET_CAP]]
    atomic_write_json(FEATURES / "chembl_target_vocab.json", {
        "vocab": vocab,
        "n_adult": n_adult,
        "min_coverage": CHEMBL_MIN_COVERAGE,
        "cap": CHEMBL_TARGET_CAP,
        "representation_name": "chembl_target_profile",
        "note": "NOT Chemical Checker BioFeat",
    })

    # Encode: value = pChEMBL; missing vs inactivity: use 0 for missing + missing indicator
    rows = []
    for ik in inchikeys:
        prof = profiles.get(ik, {})
        row = {"inchikey": ik}
        for t in vocab:
            if t in prof:
                row[f"tgt_{t}"] = prof[t]
                row[f"miss_{t}"] = 0.0
            else:
                row[f"tgt_{t}"] = 0.0
                row[f"miss_{t}"] = 1.0
        rows.append(row)
    df = pd.DataFrame(rows).set_index("inchikey")
    atomic_write_df(FEATURES / "chembl_target_profile.csv", df.reset_index())
    log(f"chembl_target_profile shape={df.shape} vocab={len(vocab)}")
    return df


def _save_block(profiles: dict, i: int):
    atomic_write_json(CACHE / f"chembl_profiles_block_{i}.json",
                      {k: v for k, v in list(profiles.items())})
