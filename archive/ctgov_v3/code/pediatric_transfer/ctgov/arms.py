"""Identify single-drug interventional arms; exclude controls/nondrugs/combos."""
from __future__ import annotations
import re
from typing import List, Optional, Tuple

from ..utils.config import CONTROL_KEYWORDS, NONDRUG_KEYWORDS

COMBO_SEP = re.compile(r"\s*(?:\+|/\s*| and |&| plus | with )\s*", re.I)


def normalize_intervention_name(name: str) -> str:
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    for pref in ("drug:", "biological:", "dietary supplement:", "procedure:", "device:"):
        if s.startswith(pref):
            s = s[len(pref):].strip()
            break
    s = re.sub(r"\([^)]*\)", "", s).strip()
    s = re.sub(r"\d+(\.\d+)?\s*(mg|mcg|g|ml|mg/kg|iu|units?).*$", "", s).strip()
    return s


def is_control(name: str) -> bool:
    n = normalize_intervention_name(name)
    return any(k in n for k in CONTROL_KEYWORDS)


def is_nondrug(name: str, itype: Optional[str] = None) -> bool:
    it = (itype or "").lower()
    if it in {"behavioral", "device", "dietary supplement", "procedure", "radiation", "other"}:
        # "other" may still be drug-like; check name
        if it != "other":
            return True
    n = normalize_intervention_name(name)
    return any(k in n for k in NONDRUG_KEYWORDS)


def split_combo(name: str) -> List[str]:
    parts = [p.strip() for p in COMBO_SEP.split(name) if p and p.strip()]
    return parts if parts else [name]


def extract_single_drug(interventions: List[dict]) -> Tuple[Optional[str], str]:
    """Return (drug_name, status).

    status: ok | control | nondrug | combo | ambiguous | empty
    Only arms with exactly one identifiable non-control drug.
    """
    if not interventions:
        return None, "empty"
    drug_like = []
    for iv in interventions:
        name = iv.get("name") or iv.get("interventionName") or ""
        itype = iv.get("type") or iv.get("interventionType") or ""
        if is_control(name):
            continue
        if is_nondrug(name, itype):
            continue
        if itype and itype.lower() not in {"drug", "biological", "combination product", ""}:
            # skip clearly non-drug types already handled; combination product → combo
            if itype.lower() == "combination product":
                return None, "combo"
            continue
        parts = split_combo(name)
        # filter control fragments
        parts = [p for p in parts if not is_control(p)]
        if len(parts) > 1:
            return None, "combo"
        if len(parts) == 1:
            drug_like.append(parts[0])
        elif not parts:
            continue
        else:
            drug_like.append(name)
    # unique normalized
    uniq = []
    seen = set()
    for d in drug_like:
        nd = normalize_intervention_name(d)
        if nd and nd not in seen:
            seen.add(nd)
            uniq.append(d.strip())
    if len(uniq) == 0:
        return None, "control"
    if len(uniq) > 1:
        return None, "combo"
    return uniq[0], "ok"
