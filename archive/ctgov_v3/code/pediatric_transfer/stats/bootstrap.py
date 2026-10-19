"""Paired compound bootstrap ≥2000 + BH correction."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..utils.config import BOOTSTRAP_N, SEEDS, GLOBAL_SEED
from ..utils.io_utils import atomic_write_df, atomic_write_json
from ..utils.logging_utils import log
from ..utils.paths import RESULTS, CACHE


def spearman_safe(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3 or np.std(y[m]) == 0 or np.std(p[m]) == 0:
        return np.nan
    r = spearmanr(y[m], p[m]).correlation
    return float(r) if r == r else np.nan


def paired_bootstrap_delta(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = GLOBAL_SEED,
) -> dict:
    """Bootstrap compounds; delta = spearman(a) - spearman(b)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    if n < 3:
        return {"delta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan, "n": n}
    base_a = spearman_safe(y, pred_a)
    base_b = spearman_safe(y, pred_b)
    delta0 = base_a - base_b
    deltas = []
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        da = spearman_safe(y[idx], pred_a[idx])
        db = spearman_safe(y[idx], pred_b[idx])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(da - db)
        if (i + 1) % 500 == 0:
            # checkpoint every 500
            atomic_write_json(CACHE / f"boot_partial_{seed}_{i+1}.json", {"deltas_so_far": len(deltas)})
    deltas = np.array(deltas)
    if len(deltas) == 0:
        return {"delta": delta0, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan, "n": n,
                "spearman_a": base_a, "spearman_b": base_b}
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided p: fraction of bootstrap deltas with opposite sign or zero relative to observed
    p = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    p = float(min(max(p, 0.0), 1.0))
    return {
        "delta": float(delta0),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": p,
        "n": n,
        "n_boot_valid": int(len(deltas)),
        "spearman_a": base_a,
        "spearman_b": base_b,
    }


def bh_correct(pvals: List[float]) -> List[float]:
    p = np.array([np.nan if v is None else v for v in pvals], float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return out.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    adj = np.clip(adj, 0, 1)
    # unsort
    inv = np.empty(m, int)
    inv[order] = np.arange(m)
    out[valid] = adj[inv]
    return out.tolist()


def run_all_comparisons(predictions_dir, outcomes: List[str], model: str = "xgb") -> pd.DataFrame:
    from pathlib import Path
    predictions_dir = Path(predictions_dir)
    rows = []
    for outcome in outcomes:
        # load chem and chem_true_bio and perms
        def load(rep):
            p = predictions_dir / f"pred_{outcome}_{rep}_{model}.csv"
            if not p.exists():
                return None
            return pd.read_csv(p)

        chem = load("chem")
        true = load("chem_true_bio")
        bio = load("bio_only")
        if chem is None or true is None:
            continue
        # align on inchikey
        m = chem.merge(true, on="inchikey", suffixes=("_chem", "_true"))
        y = m["y_true_chem"].values
        comparisons = [
            ("chem_true_bio", "chem", m["y_pred_true"].values, m["y_pred_chem"].values),
        ]
        if bio is not None:
            mb = chem.merge(bio, on="inchikey", suffixes=("_chem", "_bio"))
            comparisons.append(("chem_true_bio", "bio_only",
                                true.set_index("inchikey").loc[mb["inchikey"], "y_pred"].values,
                                mb["y_pred_bio"].values))
        perm_preds = []
        for seed in SEEDS:
            pr = load(f"chem_permuted_bio_seed_{seed}")
            if pr is not None:
                mp = chem.merge(pr, on="inchikey", suffixes=("_chem", "_perm"))
                comparisons.append((f"chem_true_bio", f"chem_permuted_bio_seed_{seed}",
                                    true.set_index("inchikey").reindex(mp["inchikey"])["y_pred"].values,
                                    mp["y_pred_perm"].values))
                perm_preds.append(pr.set_index("inchikey").reindex(chem["inchikey"])["y_pred"].values)
        if perm_preds:
            mean_perm = np.nanmean(np.vstack(perm_preds), axis=0)
            comparisons.append(("chem_true_bio", "mean_permuted_bio",
                                true.set_index("inchikey").reindex(chem["inchikey"])["y_pred"].values,
                                mean_perm))

        for a_name, b_name, pa, pb in comparisons:
            # drop nan pairs
            mask = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
            # y may need reindex if lengths differ
            if len(pa) != len(y):
                continue
            stats = paired_bootstrap_delta(y[mask], pa[mask], pb[mask])
            rows.append({
                "outcome": outcome,
                "model": model,
                "contrast": f"{a_name}_minus_{b_name}",
                "representation_a": a_name,
                "representation_b": b_name,
                **stats,
            })
            log(f"bootstrap {outcome} {a_name}-{b_name} delta={stats['delta']} ci=[{stats['ci_lo']},{stats['ci_hi']}]")
            atomic_write_df(RESULTS / "bootstrap_comparisons_partial.csv", pd.DataFrame(rows))

    df = pd.DataFrame(rows)
    if not df.empty:
        # BH across endpoint primary comparisons: chem_true_bio_minus_chem
        primary = df["contrast"] == "chem_true_bio_minus_chem"
        adj = bh_correct(df.loc[primary, "p_value"].tolist())
        df.loc[primary, "p_bh"] = adj
        # also BH all contrasts
        df["p_bh_all"] = bh_correct(df["p_value"].tolist())
    atomic_write_df(RESULTS / "bootstrap_comparisons.csv", df)
    return df


def faers_validation(ped_pred: pd.DataFrame, faers_ror: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Spearman vs FAERS logROR; never merge with EV."""
    if ped_pred.empty or faers_ror.empty:
        return pd.DataFrame()
    # map inchikey via mapping
    mp = mapping[mapping.status == "ok"][["drug_name_norm", "inchikey"]]
    fr = faers_ror.merge(mp, on="drug_name_norm", how="inner")
    rows = []
    for (outcome, rep, model), g in ped_pred.groupby(["outcome", "representation", "model"]):
        sub = g.merge(fr[fr.outcome == outcome][["inchikey", "logROR", "ped_drug_reports", "adult_drug_reports"]],
                      on="inchikey", how="inner")
        if len(sub) < 5:
            continue
        sp = spearman_safe(sub["y_pred"].values, sub["logROR"].values)
        # sensitivity min reports
        for min_rep in [1, 3, 5, 10]:
            ss = sub[(sub.ped_drug_reports >= min_rep) & (sub.adult_drug_reports >= min_rep)]
            if len(ss) < 5:
                continue
            rows.append({
                "outcome": outcome, "representation": rep, "model": model,
                "min_reports": min_rep, "n": len(ss),
                "spearman_pred_vs_faers_logROR": spearman_safe(ss["y_pred"].values, ss["logROR"].values),
                "note": "FAERS spontaneous ROR is not incidence; external PV evidence only",
            })
    out = pd.DataFrame(rows)
    atomic_write_df(RESULTS / "faers_validation.csv", out)
    return out
