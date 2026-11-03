"""Aggregate arm-level rows to structure–outcome smoothed incidence."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..utils.io_utils import atomic_write_df
from ..utils.logging_utils import log
from ..utils.paths import PROCESSED


def smoothed_incidence(affected: float, at_risk: float) -> float:
    """(a+0.5)/(n+1)"""
    return (affected + 0.5) / (at_risk + 1.0)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def aggregate_by_structure_outcome(df: pd.DataFrame, structure_col: str = "inchikey") -> pd.DataFrame:
    """Across trials: sum affected and at_risk by standardized structure+outcome.

    Within arm/category MAX already applied upstream.
    """
    if df.empty:
        return df
    g = df.groupby([structure_col, "outcome"], as_index=False).agg(
        num_affected=("num_affected", "sum"),
        num_at_risk=("num_at_risk", "sum"),
        n_trials=("nct_id", "nunique"),
        n_arms=("nct_id", "count"),
        drug_names=("drug_name_raw", lambda s: "|".join(sorted(set(map(str, s)))[:20])),
    )
    g["incidence_raw"] = g["num_affected"] / g["num_at_risk"].clip(lower=1)
    g["incidence_smooth"] = g.apply(lambda r: smoothed_incidence(r["num_affected"], r["num_at_risk"]), axis=1)
    g["logit_incidence"] = g["incidence_smooth"].map(logit)
    g["estimate_type"] = "clinical_trial_smoothed_incidence"
    return g


def split_and_save(arm_df: pd.DataFrame, mapped: pd.DataFrame) -> dict:
    """Merge mapping, aggregate adult/pediatric separately."""
    if arm_df.empty:
        log("aggregate: empty arm_df")
        return {"adult": pd.DataFrame(), "pediatric": pd.DataFrame()}
    m = arm_df.merge(mapped, on="drug_name_norm", how="inner")
    m = m[m["inchikey"].notna() & (m["inchikey"] != "")]
    out = {}
    for bucket in ("adult", "pediatric"):
        sub = m[m["age_bucket"] == bucket].copy()
        agg = aggregate_by_structure_outcome(sub)
        agg["age_bucket"] = bucket
        path = PROCESSED / f"ctgov_{bucket}_structure_outcome.csv"
        atomic_write_df(path, agg)
        log(f"Wrote {path} rows={len(agg)} structures={agg['inchikey'].nunique() if len(agg) else 0}")
        out[bucket] = agg
    atomic_write_df(PROCESSED / "ctgov_arm_level_mapped.csv", m)
    return out
