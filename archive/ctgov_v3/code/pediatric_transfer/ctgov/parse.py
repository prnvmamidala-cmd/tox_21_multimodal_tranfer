"""Parse CT.gov v2 study JSON into arm-level AE incidence rows."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from .age import classify_age
from .arms import extract_single_drug, normalize_intervention_name
from .outcomes import map_term_to_outcome
from ..utils.config import OUTCOMES
from ..utils.logging_utils import log


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def extract_eligibility_ages(study: dict) -> Tuple[Optional[str], Optional[str]]:
    elig = _get(study, "protocolSection", "eligibilityModule", default={}) or {}
    return elig.get("minimumAge"), elig.get("maximumAge")


def extract_nct(study: dict) -> str:
    return _get(study, "protocolSection", "identificationModule", "nctId", default="") or ""


def _arm_groups(study: dict) -> List[dict]:
    return list(_get(study, "protocolSection", "armsInterventionsModule", "armGroups", default=[]) or [])


def _interventions(study: dict) -> List[dict]:
    return list(_get(study, "protocolSection", "armsInterventionsModule", "interventions", default=[]) or [])


def _results_ae_groups(study: dict) -> List[dict]:
    """Return serious + other AE group lists with stats."""
    ae = _get(study, "resultsSection", "adverseEventsModule", default={}) or {}
    return ae


def _num(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_study_ae_rows(study: dict) -> List[dict]:
    """Parse one study into per-arm category rows before drug filtering.

    Aggregation rule (documented): within arm/category use MAX affected (not sum)
    across event terms mapped to the same outcome.
    """
    nct = extract_nct(study)
    amin, amax = extract_eligibility_ages(study)
    age_bucket = classify_age(amin, amax)
    if age_bucket is None:
        return []

    ae_mod = _results_ae_groups(study)
    if not ae_mod:
        return []

    # event groups: list of {title, deathsNumAffected, seriousNumAffected, ...} at group level
    # and event lists
    group_list = ae_mod.get("eventGroups") or []
    # Map group id/title → at-risk
    group_meta = {}
    for g in group_list:
        gid = g.get("id") or g.get("title") or ""
        group_meta[gid] = {
            "title": g.get("title") or gid,
            "id": gid,
            "seriousNumAtRisk": _num(g.get("seriousNumAtRisk")),
            "otherNumAtRisk": _num(g.get("otherNumAtRisk")),
            "deathsNumAffected": _num(g.get("deathsNumAffected")),
            "seriousNumAffected": _num(g.get("seriousNumAffected")),
            "otherNumAffected": _num(g.get("otherNumAffected")),
        }

    # Build title→interventions from protocol arms
    arms = _arm_groups(study)
    interventions = _interventions(study)
    # Map intervention name → type
    iv_type = {(iv.get("name") or "").lower(): iv.get("type") for iv in interventions}

    # Match AE event group titles to arm groups by title similarity
    arm_by_title = {(a.get("label") or "").strip().lower(): a for a in arms}

    rows = []
    inclusion_notes = []

    def resolve_drug_for_group(gtitle: str) -> Tuple[Optional[str], str, List[dict]]:
        key = (gtitle or "").strip().lower()
        arm = arm_by_title.get(key)
        # fuzzy: substring match either direction
        if arm is None:
            for lab, a in arm_by_title.items():
                if not lab:
                    continue
                if lab in key or key in lab:
                    arm = a
                    break
        ivs = []
        if arm:
            for iname in arm.get("interventionNames") or []:
                # strip leading "Drug: " prefixes common in CT.gov
                clean = iname
                for pref in ("Drug: ", "Biological: ", "Dietary Supplement: ", "Procedure: "):
                    if clean.startswith(pref):
                        clean = clean[len(pref):]
                ivs.append({"name": clean, "type": iv_type.get(iname.lower(), iv_type.get(clean.lower(), "DRUG"))})
        if not ivs:
            # match intervention by name contained in group title
            for iv in interventions:
                iname = iv.get("name") or ""
                if iname and iname.lower() in key:
                    ivs.append({"name": iname, "type": iv.get("type") or "DRUG"})
        if not ivs:
            # fallback: use group title as intervention name candidate
            ivs = [{"name": gtitle, "type": "DRUG"}]
        drug, status = extract_single_drug(ivs)
        return drug, status, ivs

    # Aggregate dict: (group_id, outcome) → max affected, at_risk, source
    agg: Dict[Tuple[str, str], dict] = {}

    def upd(gid: str, outcome: str, affected: Optional[float], at_risk: Optional[float], how: str):
        if affected is None or at_risk is None or at_risk <= 0:
            return
        key = (gid, outcome)
        if key not in agg:
            agg[key] = {"affected": affected, "at_risk": at_risk, "how": how}
        else:
            # MAX affected within arm/category (document: not sum)
            if affected > agg[key]["affected"]:
                agg[key]["affected"] = affected
                agg[key]["how"] = how + "|max"

    for gid, meta in group_meta.items():
        # ANY_SERIOUS / ANY_OTHER_AE / MORTALITY from group totals
        upd(gid, "ANY_SERIOUS", meta["seriousNumAffected"], meta["seriousNumAtRisk"], "group_serious")
        upd(gid, "ANY_OTHER_AE", meta["otherNumAffected"], meta["otherNumAtRisk"], "group_other")
        # deaths often under serious at-risk
        deaths_at_risk = meta["seriousNumAtRisk"] or meta["otherNumAtRisk"]
        upd(gid, "MORTALITY", meta["deathsNumAffected"], deaths_at_risk, "group_deaths")

    # Per-event lists
    for list_key, default_at_risk_key in (
        ("seriousEvents", "seriousNumAtRisk"),
        ("otherEvents", "otherNumAtRisk"),
    ):
        for ev in ae_mod.get(list_key) or []:
            term = ev.get("term") or ""
            soc = ev.get("organSystem") or ev.get("sourceVocabulary") or ""
            # organSystem is MedDRA SOC in CT.gov
            outcome = map_term_to_outcome(term, soc if "disorder" in (soc or "").lower() or "infection" in (soc or "").lower() or len(soc or "") > 3 else None)
            # Prefer organSystem as SOC
            from .outcomes import map_soc_to_outcome
            soc_out = map_soc_to_outcome(ev.get("organSystem"))
            outcome = soc_out or map_term_to_outcome(term, None)
            if not outcome:
                continue
            for st in ev.get("stats") or []:
                gid = st.get("groupId") or ""
                affected = _num(st.get("numAffected"))
                at_risk = _num(st.get("numAtRisk"))
                if at_risk is None and gid in group_meta:
                    at_risk = group_meta[gid].get(default_at_risk_key)
                upd(gid, outcome, affected, at_risk, f"event:{term[:40]}")

    # Emit rows with drug resolution
    for (gid, outcome), v in agg.items():
        meta = group_meta.get(gid, {})
        gtitle = meta.get("title") or gid
        drug, status, ivs = resolve_drug_for_group(gtitle)
        if status != "ok" or not drug:
            inclusion_notes.append({
                "nct_id": nct, "group": gtitle, "status": status,
                "interventions": [i.get("name") for i in ivs],
            })
            continue
        rows.append({
            "nct_id": nct,
            "age_bucket": age_bucket,
            "min_age": amin,
            "max_age": amax,
            "arm_title": gtitle,
            "group_id": gid,
            "drug_name_raw": drug,
            "drug_name_norm": normalize_intervention_name(drug),
            "outcome": outcome,
            "num_affected": v["affected"],
            "num_at_risk": v["at_risk"],
            "agg_rule": "max_within_arm_category",
            "agg_source": v["how"],
        })
    return rows


def studies_to_frame(studies: List[dict]):
    import pandas as pd
    all_rows = []
    for st in studies:
        try:
            all_rows.extend(parse_study_ae_rows(st))
        except Exception:
            continue
    return pd.DataFrame(all_rows)
