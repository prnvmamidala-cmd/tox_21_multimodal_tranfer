"""Map AE terms/SOCs to harmonized outcomes. Within arm/category use MAX affected."""
from __future__ import annotations
from typing import Optional

from ..utils.config import SOC_TO_OUTCOME, KEYWORD_MAP, OUTCOMES


def map_soc_to_outcome(soc: Optional[str]) -> Optional[str]:
    if not soc:
        return None
    key = soc.strip().lower()
    return SOC_TO_OUTCOME.get(key)


def map_term_to_outcome(term: Optional[str], soc: Optional[str] = None) -> Optional[str]:
    out = map_soc_to_outcome(soc)
    if out:
        return out
    if not term:
        return None
    t = term.strip().lower()
    # exact keyword containment (longest first for specificity)
    for kw, outcome in sorted(KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
        if kw in t:
            return outcome
    return None


def is_valid_outcome(name: str) -> bool:
    return name in OUTCOMES
