"""Search for EudraVigilance exports; never invent an API."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from ..utils.io_utils import atomic_write_text
from ..utils.logging_utils import log
from ..utils.paths import RAW, RUN_DIR


SEARCH_ROOTS = [
    Path("/workspace"),
    Path("/home/box/agent-data"),
    Path("/home/box/reference"),
    Path("/home/box/deps"),
    Path("/home/box/sand-data"),
]


def find_eudra_files() -> List[Path]:
    hits = []
    patterns = ["*eudra*", "*Eudra*", "*EUDRA*", "*eudravigilance*", "*EVDAS*"]
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.rglob(pat):
                if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".xls", ".xlsx", ".xml"}:
                    # skip our own notice
                    if "EUDRAVIGILANCE_NOT_AVAILABLE" in p.name:
                        continue
                    if "pediatric_transfer" in str(p) and p.suffix == ".txt":
                        continue
                    hits.append(p)
    return hits


def ensure_eudra_status() -> dict:
    hits = find_eudra_files()
    notice = RAW / "EUDRAVIGILANCE_NOT_AVAILABLE.txt"
    if not hits:
        msg = (
            "EudraVigilance export not available.\n"
            "No csv/tsv/xls/xlsx/xml EudraVigilance files found under workspace search roots.\n"
            "There is no public unauthenticated EudraVigilance bulk API; an authorized EVDAS export is required.\n"
            "EV validation: PENDING/UNAVAILABLE. Continuing with CT.gov + FAERS only.\n"
            "FAERS and EV cases must never be merged.\n"
        )
        atomic_write_text(notice, msg)
        log("EudraVigilance UNAVAILABLE — notice written")
        return {"available": False, "files": [], "notice": str(notice)}
    log(f"EudraVigilance files found: {hits}")
    return {"available": True, "files": [str(h) for h in hits], "notice": None}
