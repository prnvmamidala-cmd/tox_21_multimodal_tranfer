"""Pediatric-vs-adult ROR with 0.5 Haldane. NOT incidence."""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.io_utils import atomic_write_df
from ..utils.logging_utils import log
from ..utils.paths import PROCESSED
from ..ctgov.outcomes import map_term_to_outcome, map_soc_to_outcome
from ..ctgov.arms import normalize_intervention_name


def _is_primary_suspect(drug: dict) -> bool:
    role = (drug.get("drugcharacterization") or "").strip()
    # 1 = Suspect (primary often characterization)
    return role == "1"


def reports_to_pairs(reports: List[dict], domain: str) -> pd.DataFrame:
    """Dedupe report–drug–outcome; keep all suspect reports for denominators."""
    rows = []
    for rep in reports:
        safetyreportid = str(rep.get("safetyreportid") or rep.get("receiptdate") or "")
        patient = rep.get("patient") or {}
        drugs = patient.get("drug") or []
        reactions = patient.get("reaction") or []
        suspect_drugs = []
        all_drugs = []
        for d in drugs:
            names = []
            if d.get("medicinalproduct"):
                names.append(d["medicinalproduct"])
            for a in d.get("activesubstance") or []:
                if isinstance(a, dict) and a.get("activesubstancename"):
                    names.append(a["activesubstancename"])
                elif isinstance(a, str):
                    names.append(a)
            openfda = (d.get("openfda") or {})
            for gn in openfda.get("generic_name") or []:
                names.append(gn)
            for bn in openfda.get("substance_name") or []:
                names.append(bn)
            entry = {"names": names, "primary": _is_primary_suspect(d)}
            all_drugs.append(entry)
            if entry["primary"]:
                suspect_drugs.append(entry)
        # outcomes from reactions
        outcomes = set()
        for rxn in reactions:
            pt = rxn.get("reactionmeddrapt") or ""
            soc = rxn.get("reactionmeddrasoc") or ""
            out = map_soc_to_outcome(soc) or map_term_to_outcome(pt, soc)
            if out:
                outcomes.add(out)
        # also ANY_SERIOUS if serious
        serious = str(rep.get("serious") or "") == "1"
        if serious:
            outcomes.add("ANY_SERIOUS")

        for sd in suspect_drugs:
            for name in sd["names"]:
                nn = normalize_intervention_name(name)
                if not nn:
                    continue
                for out in outcomes:
                    rows.append({
                        "safetyreportid": safetyreportid,
                        "drug_name_norm": nn,
                        "drug_name_raw": name,
                        "outcome": out,
                        "domain": domain,
                        "primary_suspect": True,
                    })
        # denominator tracking: all primary-suspect report-drug pairs
        for sd in suspect_drugs:
            for name in sd["names"]:
                nn = normalize_intervention_name(name)
                if nn:
                    rows.append({
                        "safetyreportid": safetyreportid,
                        "drug_name_norm": nn,
                        "drug_name_raw": name,
                        "outcome": "__ANY_REPORT__",
                        "domain": domain,
                        "primary_suspect": True,
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["safetyreportid", "drug_name_norm", "outcome"])
    return df


def compute_ror(ped_df: pd.DataFrame, adult_df: pd.DataFrame) -> pd.DataFrame:
    """ROR pediatric vs adult with Haldane 0.5 correction. Vectorized."""
    def counts(df):
        if df.empty:
            return pd.DataFrame(columns=["drug_name_norm", "outcome", "n"])
        return (df[df["outcome"] != "__ANY_REPORT__"]
                .groupby(["drug_name_norm", "outcome"]).size().reset_index(name="n"))

    def denom(df):
        if df.empty:
            return pd.Series(dtype=float)
        return (df[df["outcome"] == "__ANY_REPORT__"]
                .groupby("drug_name_norm").size())

    pc = counts(ped_df).rename(columns={"n": "ped_a"})
    ac = counts(adult_df).rename(columns={"n": "adult_c"})
    pd_den = denom(ped_df).rename("ped_drug_reports")
    ad_den = denom(adult_df).rename("adult_drug_reports")

    keys = pd.concat([pc[["drug_name_norm", "outcome"]], ac[["drug_name_norm", "outcome"]]]).drop_duplicates()
    m = keys.merge(pc, on=["drug_name_norm", "outcome"], how="left")
    m = m.merge(ac, on=["drug_name_norm", "outcome"], how="left")
    m = m.merge(pd_den, left_on="drug_name_norm", right_index=True, how="left")
    m = m.merge(ad_den, left_on="drug_name_norm", right_index=True, how="left")
    m = m.fillna(0)
    a = m["ped_a"].astype(float)
    c = m["adult_c"].astype(float)
    n_ped = m["ped_drug_reports"].astype(float)
    n_ad = m["adult_drug_reports"].astype(float)
    b = (n_ped - a).clip(lower=0)
    d = (n_ad - c).clip(lower=0)
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ror = (a_ / b_) / (c_ / d_)
    log_ror = np.log(ror)
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    ci_lo = log_ror - 1.96 * se
    ci_hi = log_ror + 1.96 * se
    out_df = pd.DataFrame({
        "drug_name_norm": m["drug_name_norm"],
        "outcome": m["outcome"],
        "ped_a": a, "ped_b": b, "adult_c": c, "adult_d": d,
        "ped_drug_reports": n_ped, "adult_drug_reports": n_ad,
        "ROR": ror, "logROR": log_ror, "SE_logROR": se,
        "logROR_ci95_lo": ci_lo, "logROR_ci95_hi": ci_hi,
        "ROR_ci95_lo": np.exp(ci_lo), "ROR_ci95_hi": np.exp(ci_hi),
        "estimate_type": "faers_ror_not_incidence",
        "note": "Spontaneous reporting ROR; not confirmed causal toxicity; not incidence",
    })
    # NOTE: callers write faers_ror.csv explicitly; do not side-effect here (breaks unit tests vs production)
    log(f"FAERS ROR rows={len(out_df)}")
    return out_df
