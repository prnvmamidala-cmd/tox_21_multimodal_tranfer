"""Drug name → structure harmonization with checkpoints every 25 mappings."""
from __future__ import annotations
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")

from ..utils.config import PUBCHEM_BASE, CHEMBL_BASE, CONTROL_KEYWORDS, NONDRUG_KEYWORDS
from ..utils.io_utils import atomic_write_df, atomic_write_json, load_json
from ..utils.logging_utils import log, log_exc
from ..utils.paths import CACHE, PROCESSED

CACHE_MAP = CACHE / "drug_name_cache.json"
_PUBCHEM_SKIP_UNTIL = time.time() + 900  # start with CIR during known 503 outage
_PUBCHEM_503_STREAK = 0
_CHEMBL_SKIP_UNTIL = 0.0  # re-enabled; was timing out earlier
_CHEMBL_FAIL_STREAK = 0
_CIR_SKIP_UNTIL = time.time() + 1200  # SSL EOF storm; cool down
_CIR_FAIL_STREAK = 0

RETRY_QUEUE = CACHE / "mapping_retry_queue.json"

# Prefixes / role labels often appearing before the active substance
_ROLE_PREFIX = re.compile(
    r"^(?:comparator|reference|active|investigational|test|control|treatment|"
    r"study\s*drug|study\s*medication|arm|group|cohort|part\s*[a-z0-9]+|"
    r"combination\s*product|drug|medication|intervention)\s*[:\-–]\s*",
    re.I,
)
_ROUTE_FORM = re.compile(
    r"\b(?:oral|iv|i\.v\.|intravenous|intramuscular|im|subcutaneous|sc|s\.c\.|"
    r"topical|inhaled|inhalation|nasal|ophthalmic|otic|rectal|vaginal|"
    r"transdermal|sublingual|buccal|tablet|capsule|injection|depot|"
    r"solution|suspension|cream|ointment|gel|patch|syrup|powder|"
    r"extended[\s\-]?release|immediate[\s\-]?release|delayed[\s\-]?release|"
    r"er|xr|sr|cr|mr|odt)\b",
    re.I,
)
_DOSE_PAT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|iu|units?|%|mg/kg|mg/m2)\b",
    re.I,
)
_TRAILING_JUNK = re.compile(
    r"[\s,;]+(?:for\s+\w+|fluidcrystal®?|injection\s+depot|mmx®?|"
    r"without\s+api|system|device).*$",
    re.I,
)


def normalize_name(name: str) -> str:
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    s = re.sub(r"\([^)]*\)", "", s).strip()
    return s


def extract_active_substance(name: str) -> List[str]:
    """Generate candidate query strings from an arm/drug label.

    Order: stripped active substance → original normalized → raw fragments.
    """
    raw = (name or "").strip()
    n = normalize_name(raw)
    cands: List[str] = []

    def _add(x: str):
        x = normalize_name(x)
        if x and x not in cands and len(x) >= 2:
            cands.append(x)

    _add(n)
    # strip role prefixes like "comparator: rosuvastatin"
    stripped = _ROLE_PREFIX.sub("", n)
    _add(stripped)
    # drop dose / route / form tokens
    no_dose = _DOSE_PAT.sub(" ", stripped)
    no_route = _ROUTE_FORM.sub(" ", no_dose)
    no_junk = _TRAILING_JUNK.sub("", no_route)
    cleaned = re.sub(r"[\s,;:/]+", " ", no_junk).strip(" -–,;")
    _add(cleaned)
    # first token-ish phrase (up to 4 words) often is the INN
    words = cleaned.split()
    if len(words) >= 2:
        _add(" ".join(words[:2]))
        _add(words[0])
    elif words:
        _add(words[0])
    # if colon-split, prefer RHS
    if ":" in n:
        rhs = n.split(":", 1)[1].strip()
        _add(rhs)
        rhs2 = _ROUTE_FORM.sub(" ", _DOSE_PAT.sub(" ", rhs)).strip()
        _add(rhs2)
    return cands or [n]


def is_excluded_name(name: str) -> bool:
    n = normalize_name(name)
    if any(k in n for k in CONTROL_KEYWORDS):
        return True
    if any(k in n for k in NONDRUG_KEYWORDS):
        return True
    # explicit combination/device-ish arm labels
    if re.search(r"\b(?:combination\s+of|plus|/\s*\+|&\s*)\b", n) and "combination product" not in n:
        # keep single-agent "combination product: X" for extraction; multi-drug "X + Y" exclude later
        if "+" in n or " and " in n:
            return True
    return False


def looks_like_nondrug_arm(name: str) -> bool:
    n = (name or "").lower().strip()
    bad = [
        "dose ", "after dose", "informed consent", "post infant", "mdv:", "sds:",
        "arm ", "group ", "cohort ", "week ", "cycle ", "placebo", "vaccine schedule",
        "spinal cord stimulator", "without api", " needle", "hz", "ppm ",
        "support assembly", "personal lubricant", "living donor", "oatmeal",
        "vaccine", "pneumococcal", "influenza virus", "conjugate", "balm",
        "cream", "ointment", "russian cohort", "cetaphil", "contrast reagent",
        "mri ", "colloidal", "age group", "age cohort", "months age", "years age",
        "allergen extract", "pollen allergen", "house dust mites", "experimental",
        "myobloc", "albumin", "corifollitropin",  # biologics/large molecules: no small-mol SMILES
    ]
    if any(b in n for b in bad):
        return True
    # bare "group"/"arm"/"cohort" tokens (including end of string)
    if re.search(r"\b(?:group|arm|cohort|part)\b", n):
        return True
    if len(n) > 80 or len(n) < 2:
        return True
    if n.count(":") >= 2:
        return True
    if re.fullmatch(r"(?:group|arm|cohort|part)\s+[a-z0-9]+", n):
        return True
    if re.fullmatch(r"[\d\s\.,µu%]+(?:mg|mcg|µg|ml|mm|hz|%|u|iu|ppm|vpnc)?", n):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?(?:mg|mcg|ml|mm|hz|ppm|u|vpnc)", n):
        return True
    # short opaque codes / dose+code labels unlikely to be INNs
    if re.fullmatch(r"[a-z]{0,3}\d+[a-z0-9._\-]{0,8}", n) and len(n) <= 12:
        return True
    if re.fullmatch(r"\d+\s*(?:ug|µg|u|iu|ir|mg)?\s*[a-z0-9_\-]{2,20}", n) and " " in n:
        # e.g. "21ug map0020", "1500u myobloc" already covered; keep generic dose+code
        if not re.search(r"[aeiou]{2}|olol|pril|sartan|statin|mab|nib|vir|cillin", n):
            # still allow if looks like dose + drug word with vowels — handled below
            pass
    # cohort / vpnc codes
    if re.search(r"\b(?:vpnc|cohort)\b", n):
        return True
    # large-molecule / biologic naming patterns (no small-mol SMILES expected)
    if re.search(r"(?:mab|cept|kinra|vectin|botulinum|toxina|vaccine|virus|stem cell|cellular)\b", n):
        return True
    if re.search(r"(?:ad\d+|acwy|rsv|menb|hpv|dtap|tdap|4cmenb)", n):
        return True
    # procedure / device / food / brand-biologic / non-actionable arm labels
    if re.search(r"\b(?:ablat(?:ed|ion)|participants?|management|omission|juice|hip stem|stem\b|matrix|device|catheter|bandage|dressing|stimulator|toxoid|deltoid|forearm|suprascapular|tetanus|diphtheria)\b", n):
        return True
    if re.search(r"\b(?:years?|months?|weeks?)\b", n) and re.search(r"\d", n):
        return True
    if re.search(r"\b(?:actemra|adacel|acam2000|humira|enbrel|remicade|keytruda|opdivo|rituxan|herceptin|avastin|lucentis|eylea|botox|xeomin|dysport|myobloc)\b", n):
        return True
    # dose+code / opaque catalog codes
    if re.search(r"^\d+\s*(?:ug|µg|u|iu|ir|mg|mcg)\s+[a-z0-9_\-]{2,}$", n):
        return True
    if re.fullmatch(r"[a-z]{1,6}\d{2,}[a-z0-9_\-]*", n):
        return True
    return False


def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_to_inchikey(mol: Chem.Mol) -> Optional[str]:
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def smiles_to_record(smiles: str) -> Optional[dict]:
    mol = Chem.MolFromSmiles(smiles)
    mol = standardize_mol(mol)
    if mol is None:
        return None
    can = Chem.MolToSmiles(mol, canonical=True)
    if "." in can:
        return None  # still a mixture
    ik = mol_to_inchikey(mol)
    if not ik:
        return None
    return {"canonical_smiles": can, "inchikey": ik, "inchikey_prefix": ik.split("-")[0]}


def pubchem_name_to_smiles(session: requests.Session, name: str) -> Optional[str]:
    """PubChem PUG REST. Circuit-breaks on 503 storms so CIR can take over quickly."""
    global _PUBCHEM_SKIP_UNTIL, _PUBCHEM_503_STREAK
    if time.time() < _PUBCHEM_SKIP_UNTIL:
        return None
    url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(name)}/property/CanonicalSMILES,IsomericSMILES,SMILES,ConnectivitySMILES/JSON"
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 404:
            _PUBCHEM_503_STREAK = 0
            return None
        if r.status_code in (429, 503):
            _PUBCHEM_503_STREAK += 1
            if _PUBCHEM_503_STREAK >= 2:
                _PUBCHEM_SKIP_UNTIL = time.time() + 600  # skip 10 min
            return None
        r.raise_for_status()
        _PUBCHEM_503_STREAK = 0
        props = (r.json().get("PropertyTable") or {}).get("Properties") or []
        if not props:
            return None
        return (props[0].get("CanonicalSMILES") or props[0].get("IsomericSMILES")
                or props[0].get("SMILES") or props[0].get("ConnectivitySMILES"))
    except Exception:
        return None


def chembl_synonym_to_smiles(session: requests.Session, name: str) -> Optional[str]:
    """ChEMBL search — no circuit-break (need complete deferred coverage)."""
    try:
        url = f"{CHEMBL_BASE}/molecule/search.json"
        r = session.get(url, params={"q": name, "limit": 3}, timeout=22)
        if r.status_code != 200:
            return None
        molecules = r.json().get("molecules") or []
        for mol in molecules:
            structs = mol.get("molecule_structures") or {}
            smi = structs.get("canonical_smiles")
            if smi:
                return smi
        return None
    except Exception:
        return None


def cir_name_to_smiles(session: requests.Session, name: str) -> Optional[str]:
    """NIH NCI CIR — circuit-breaks on SSL/EOF storms."""
    global _CIR_SKIP_UNTIL, _CIR_FAIL_STREAK
    if time.time() < _CIR_SKIP_UNTIL:
        return None
    path = f"/chemical/structure/{requests.utils.quote(name)}/smiles"
    urls = [
        f"https://cactus.nci.nih.gov{path}",
        f"http://cactus.nci.nih.gov{path}",
    ]
    for url in urls:
        try:
            r = session.get(url, timeout=8, allow_redirects=True)
            if r.status_code == 404:
                _CIR_FAIL_STREAK = 0
                return None
            if r.status_code != 200:
                continue
            smi = (r.text or "").strip()
            if not smi or smi.lower().startswith("<!") or "page not found" in smi.lower():
                continue
            _CIR_FAIL_STREAK = 0
            return smi.splitlines()[0].strip()
        except Exception:
            _CIR_FAIL_STREAK += 1
            if _CIR_FAIL_STREAK >= 3:
                _CIR_SKIP_UNTIL = time.time() + 1200
            continue
    return None


def map_one(session: requests.Session, name_norm: str, name_raw: str, cache: dict) -> dict:
    if name_norm in cache:
        return cache[name_norm]
    if is_excluded_name(name_norm) or looks_like_nondrug_arm(name_norm):
        status = "excluded_control_or_nondrug"
        rec = {"drug_name_norm": name_norm, "drug_name_raw": name_raw, "status": status,
               "source": None, "confidence": 0, "canonical_smiles": None, "inchikey": None,
               "inchikey_prefix": None, "query_used": None}
        cache[name_norm] = rec
        return rec

    # try exact as smiles only if string looks chemical
    # Only attempt SMILES parse on strings that look chemical (not drug names like "(-)-Epicatechin")
    _smi_like = name_raw and (" " not in name_raw.strip()) and any(ch in name_raw for ch in ("=", "#", "[", "]")) and any(c.isdigit() for c in name_raw)
    if _smi_like:
        if Chem.MolFromSmiles(name_raw) is not None:
            rec_s = smiles_to_record(name_raw)
            if rec_s:
                out = {"drug_name_norm": name_norm, "drug_name_raw": name_raw, "status": "ok",
                       "source": "raw_smiles", "confidence": 0.99, "query_used": name_raw, **rec_s}
                cache[name_norm] = out
                return out

    candidates = extract_active_substance(name_raw) or extract_active_substance(name_norm)
    def _bad_query(q: str) -> bool:
        q = (q or "").strip().lower()
        if len(q) < 4:
            return True
        if re.fullmatch(r"[\d\s\.,%µu/-]+", q):
            return True
        if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|iu|ir|u|mm|%)", q):
            return True
        if q in {"device", "drug", "arm", "group", "cohort", "control", "placebo", "treatment",
                 "solution", "tablet", "capsule", "injection", "ointment", "cream", "gel",
                 "vaccine", "extract", "product", "system", "therapy", "management"}:
            return True
        if re.fullmatch(r"(?:arm|group|cohort|part)\s*[a-z0-9]*", q):
            return True
        return False
    # Prefer original normalized / longer substance-like strings; drop junk queries
    seen_q = set()
    queries = []
    for q in candidates:
        if q in seen_q or _bad_query(q):
            continue
        seen_q.add(q)
        queries.append(q)
    # Cap; keep order from extract_active_substance (original first, then cleaned)
    queries = queries[:2]

    smi = None
    source = None
    query_used = None
    # ChEMBL currently reachable (~10-15s); CIR SSL / PubChem 503 circuit-broken.
    for q in queries[:2]:
        smi = chembl_synonym_to_smiles(session, q)
        if smi:
            source = "chembl_synonym"
            query_used = q
            break
    if not smi:
        for q in queries[:2]:
            smi = cir_name_to_smiles(session, q)
            if smi:
                source = "cir"
                query_used = q
                break
    if not smi:
        for q in queries[:1]:
            smi = pubchem_name_to_smiles(session, q)
            if smi:
                source = "pubchem"
                query_used = q
                break

    if not smi:
        out = {"drug_name_norm": name_norm, "drug_name_raw": name_raw, "status": "unresolved",
               "source": None, "confidence": 0, "canonical_smiles": None, "inchikey": None,
               "inchikey_prefix": None, "query_used": None}
        cache[name_norm] = out
        return out
    rec_s = smiles_to_record(smi)
    if not rec_s:
        out = {"drug_name_norm": name_norm, "drug_name_raw": name_raw, "status": "excluded_mixture_or_invalid",
               "source": source, "confidence": 0.3, "canonical_smiles": smi, "inchikey": None,
               "inchikey_prefix": None, "query_used": query_used}
        cache[name_norm] = out
        return out
    # Guard: ChEMBL search is fuzzy — reject weak query hits
    qn = normalize_name(query_used or "")
    if len(qn) < 5 or qn in {"device", "solution", "extract", "anesthetic", "milligrams"}:
        out = {"drug_name_norm": name_norm, "drug_name_raw": name_raw, "status": "unresolved",
               "source": None, "confidence": 0, "canonical_smiles": None, "inchikey": None,
               "inchikey_prefix": None, "query_used": query_used}
        cache[name_norm] = out
        return out
    conf = 0.85 if source == "pubchem" else (0.8 if source == "cir" else 0.7)
    if query_used and query_used != normalize_name(name_raw) and query_used != name_norm:
        conf = max(0.55, conf - 0.15)  # extracted active substance
        source = f"{source}+active_substance"
    out = {"drug_name_norm": name_norm, "drug_name_raw": name_raw, "status": "ok",
           "source": source, "confidence": conf, "query_used": query_used, **rec_s}
    cache[name_norm] = out
    return out


def map_drug_names(names: List[Tuple[str, str]], checkpoint_every: int = 25, workers: int = 4) -> pd.DataFrame:
    """names: list of (norm, raw). Checkpoint every 25. Concurrent workers for I/O.

    No early-stop / overnight_cap. Completes the full input list.
    """
    cache = load_json(CACHE_MAP, {})
    lock = threading.Lock()
    ordered = []
    seen = set()
    for norm, raw in names:
        if norm not in seen:
            seen.add(norm)
            ordered.append((norm, raw))

    def work(item):
        norm, raw = item
        with lock:
            if norm in cache and cache[norm].get("status") not in (None, "pending", "error"):
                return cache[norm]
        session = requests.Session()
        session.headers.update({"User-Agent": "pediatric-transfer-aligned-v3/1.0"})
        try:
            rec = map_one(session, norm, raw, {})
        except Exception as e:
            log_exc("drug_mapping", norm, e)
            rec = {"drug_name_norm": norm, "drug_name_raw": raw, "status": "error",
                   "source": None, "confidence": 0, "canonical_smiles": None, "inchikey": None,
                   "inchikey_prefix": None, "query_used": None, "error": str(e)}
        with lock:
            cache[norm] = rec
        time.sleep(0.08)
        return rec

    results = []
    todo = []
    for norm, raw in ordered:
        if norm in cache and cache[norm].get("status") not in (None, "pending", "error"):
            results.append(cache[norm])
        else:
            todo.append((norm, raw))
    log(f"drug mapping cached={len(results)} todo={len(todo)} workers={workers} (NO early-stop)")
    done = 0
    retry_later = []
    chunk_size = 300
    for chunk_start in range(0, len(todo), chunk_size):
        chunk = todo[chunk_start:chunk_start + chunk_size]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, item): item for item in chunk}
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                done += 1
                if rec.get("status") == "error":
                    retry_later.append(futs[fut])
                if done % checkpoint_every == 0:
                    with lock:
                        atomic_write_json(CACHE_MAP, cache)
                    ok_n = sum(1 for r in results if r.get("status") == "ok")
                    log(f"drug mapping checkpoint n={len(results)} done_new={done}/{len(todo)} ok_so_far~{ok_n}")
        # brief pause between chunks to ease PubChem pressure
        time.sleep(1.0)
        log(f"drug mapping chunk done {min(chunk_start+chunk_size, len(todo))}/{len(todo)}")
    # bounded retry once for errors
    if retry_later:
        log(f"retrying {len(retry_later)} error mappings once")
        atomic_write_json(RETRY_QUEUE, [{"norm": a, "raw": b} for a, b in retry_later])
        for item in retry_later:
            norm, raw = item
            with lock:
                cache.pop(norm, None)
            rec = work(item)
            # replace prior error in results
            results = [r for r in results if r.get("drug_name_norm") != norm]
            results.append(rec)
    atomic_write_json(CACHE_MAP, cache)

    df = pd.DataFrame(results)
    # ensure columns
    for col in ["drug_name_norm", "drug_name_raw", "status", "source", "confidence",
                "canonical_smiles", "inchikey", "inchikey_prefix", "query_used"]:
        if col not in df.columns:
            df[col] = None
    ok = df[df["status"] == "ok"].copy()
    unresolved = df[df["status"].isin(["unresolved", "error"])].copy()
    excluded = df[df["status"].isin(["excluded_control_or_nondrug", "excluded_mixture_or_invalid"])].copy()
    atomic_write_df(PROCESSED / "drug_name_mapping.csv", df)
    atomic_write_df(PROCESSED / "unresolved_drug_names.csv", unresolved)
    atomic_write_df(PROCESSED / "excluded_combination_products.csv", excluded)

    if not ok.empty:
        aliases = ok.groupby("inchikey").agg(
            canonical_smiles=("canonical_smiles", "first"),
            names=("drug_name_norm", lambda s: "|".join(sorted(set(s)))),
            sources=("source", lambda s: "|".join(sorted(set(map(str, s))))),
            mean_confidence=("confidence", "mean"),
        ).reset_index()
        atomic_write_df(PROCESSED / "structure_aliases.csv", aliases)
    else:
        atomic_write_df(PROCESSED / "structure_aliases.csv", pd.DataFrame())
    log(f"mapping done ok={len(ok)} unresolved={len(unresolved)} excluded={len(excluded)}")
    return df
