"""Age parsing for adult/pediatric trial classification."""
from __future__ import annotations
import re
from typing import Optional, Tuple

_UNIT_DAYS = {
    "year": 365.25, "years": 365.25, "yr": 365.25, "yrs": 365.25, "y": 365.25,
    "month": 30.4375, "months": 30.4375, "mo": 30.4375, "mos": 30.4375,
    "week": 7.0, "weeks": 7.0, "wk": 7.0, "wks": 7.0, "w": 7.0,
    "day": 1.0, "days": 1.0, "d": 1.0,
    "hour": 1/24, "hours": 1/24, "hr": 1/24, "hrs": 1/24, "h": 1/24,
}


def parse_age_to_years(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s or s in {"n/a", "na", "none", "null", "-"}:
        return None
    if "adult" in s and not any(c.isdigit() for c in s):
        return 18.0
    if s in {"child", "children", "pediatric", "infant", "neonate", "newborn"}:
        return None  # ambiguous without number
    m = re.search(r"([\d]+(?:\.\d+)?)\s*([a-z]+)?", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "years").rstrip("s")
    # normalize plural
    for k, days in _UNIT_DAYS.items():
        if unit == k or unit == k.rstrip("s") or k.startswith(unit):
            return (val * days) / 365.25
    # fallback assume years
    return val


def classify_age(min_age_text: Optional[str], max_age_text: Optional[str]) -> Optional[str]:
    """Return 'adult', 'pediatric', or None (mixed/unparseable).

    Pediatric-only: max age ≤17 AND min+max parseable.
    Adult-only: min age ≥18 AND min+max parseable.
    Exclude mixed child+adult.
    """
    amin = parse_age_to_years(min_age_text)
    amax = parse_age_to_years(max_age_text)
    if amin is None or amax is None:
        return None
    if amax < amin:
        return None
    if amax <= 17.0 and amin >= 0:
        # ensure no adult inclusion
        if amax <= 17.0:
            return "pediatric"
    if amin >= 18.0:
        return "adult"
    # mixed (e.g. 12-65) or other
    return None
