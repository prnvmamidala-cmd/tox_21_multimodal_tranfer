"""Metrics, fusion comparisons, and claim files computed only from saved predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from toxtransfer.evaluation.metrics import calibration_slope, regression_errors, spearman_rho
from toxtransfer.models.fusion import (
    deterministic_uniform_gate,
    fuse,
    oracle_gate,
    select_global_weight,
)
from toxtransfer.reporting.tables import write_table
from toxtransfer.statistics.inference import (
    benjamini_hochberg,
    bootstrap_statistic,
    paired_bootstrap_delta,
    wilcoxon_signed_rank,
)
from toxtransfer.visualization.figures import make_figures


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def _pair(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    a = frame.loc[frame["representation"] == left].copy()
    b = frame.loc[frame["representation"] == right, ["inchikey", "y_pred"]].rename(
        columns={"y_pred": "y_other"}
    )
    merged = a.merge(b, on="inchikey", how="inner")
    return merged


def _score_rows(frame: pd.DataFrame, slice_name: str) -> dict:
    y = frame["y_true"].to_numpy()
    pred = frame["y_pred"].to_numpy()
    err = regression_errors(y, pred)
    return {
        "slice": slice_name,
        "spearman": spearman_rho(y, pred),
        "rmse": err["rmse"],
        "mae": err["mae"],
        "calibration_slope": calibration_slope(y, pred),
        "n": err["n"],
        "mean_y": float(np.mean(y)) if len(y) else float("nan"),
        "mean_coverage": float(frame["bio_coverage"].mean()) if len(frame) else float("nan"),
        "frac_scaffold_unseen": float(1.0 - frame["scaffold_seen"].mean()) if len(frame) else float("nan"),
    }


def _slice_frame(frame: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    if slice_name == "all_pediatric":
        return frame
    if slice_name == "structure_unseen":
        return frame.loc[frame["structure_seen"] == 0]
    if slice_name == "structure_seen":
        return frame.loc[frame["structure_seen"] == 1]
    if slice_name == "scaffold_unseen":
        return frame.loc[frame["scaffold_seen"] == 0]
    if slice_name == "unseen_coverage_positive":
        return frame.loc[(frame["structure_seen"] == 0) & (frame["bio_coverage"] > 0)]
    if slice_name == "unseen_coverage_zero":
        return frame.loc[(frame["structure_seen"] == 0) & (frame["bio_coverage"] <= 0)]
    raise KeyError(slice_name)


def _add_postprocess(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Add fusion rows. Weights are chosen on adult out-of-fold predictions only."""
    grid = [float(v) for v in cfg["transfer"]["late_fusion_grid"]]
    adult = frame.loc[(frame["phase"] == "adult_oof") & (frame["model"] == "ridge")]
    weight_table = []
    endpoint_weights = {}
    for endpoint, sub in adult.groupby("endpoint"):
        merged = _pair(sub, "chem", "early")
        if len(merged) < 3:
            continue
        weight = select_global_weight(
            merged["y_true"].to_numpy(),
            merged["y_pred"].to_numpy(),
            merged["y_other"].to_numpy(),
            grid,
            spearman_rho,
        )
        endpoint_weights[endpoint] = weight
        weight_table.append({"endpoint": endpoint, "late_endpoint_weight_chem_early": weight})
    macro_weight, macro_scores = _macro_weight(adult, grid)
    extra = []
    for phase in ("adult_oof", "transfer"):
        phase_rows = frame.loc[(frame["phase"] == phase) & (frame["model"] == "ridge")]
        for endpoint, sub in phase_rows.groupby("endpoint"):
            chem = _pair(sub, "chem", "early")
            if chem.empty:
                continue
            bio = _pair(sub, "chem", "bio")
            specs = {
                "fixed_50_50_chem_early": (chem, np.full(len(chem), 0.5)),
                "gate_coverage": (chem, chem["bio_coverage"].to_numpy()),
                "gate_binary_coverage": (chem, (chem["bio_coverage"].to_numpy() > 0).astype(float)),
                "chem_fallback": (chem, np.zeros(len(chem))),
                "random_reliability": (
                    chem,
                    deterministic_uniform_gate(chem["inchikey"].to_numpy(), 101),
                ),
                "permuted_reliability": (chem, _permute_coverage(chem, 202)),
                "oracle_nondeployable": (
                    chem,
                    oracle_gate(
                        chem["y_true"].to_numpy(),
                        chem["y_pred"].to_numpy(),
                        chem["y_other"].to_numpy(),
                    ),
                ),
                "late_global_chem_early": (chem, np.full(len(chem), macro_weight)),
                "late_endpoint_chem_early": (
                    chem,
                    np.full(len(chem), endpoint_weights.get(endpoint, macro_weight)),
                ),
            }
            if not bio.empty and len(bio) == len(chem):
                specs["fixed_50_50_chem_bio"] = (bio, np.full(len(bio), 0.5))
            for name, (base, gate) in specs.items():
                pred = fuse(base["y_pred"].to_numpy(), base["y_other"].to_numpy(), gate)
                block = base.copy()
                block["y_pred"] = pred
                block["representation"] = name
                block["model"] = "postprocess"
                block["phase"] = phase
                block["endpoint"] = endpoint
                block["dataset"] = sub["dataset"].iloc[0]
                block["fold"] = -1
                extra.append(block[frame.columns])
            if phase == "transfer" and cfg["ablations"].get("isotonic_calibration", True):
                extra.extend(_isotonic_rows(adult, sub, endpoint, frame.columns))
    fused = pd.concat([frame] + extra, ignore_index=True) if extra else frame
    info = {
        "late_global_weight": macro_weight,
        "late_global_adult_scores": {str(k): float(v) for k, v in macro_scores.items()},
        "late_endpoint_weights": weight_table,
    }
    return fused, info


def _macro_weight(adult: pd.DataFrame, grid: list[float]) -> tuple[float, dict]:
    scores = {}
    best_w = float(grid[0])
    best = -np.inf
    for weight in grid:
        rhos = []
        for _, sub in adult.groupby("endpoint"):
            merged = _pair(sub, "chem", "early")
            if len(merged) < 3:
                continue
            pred = fuse(merged["y_pred"].to_numpy(), merged["y_other"].to_numpy(), weight)
            rhos.append(spearman_rho(merged["y_true"].to_numpy(), pred))
        score = float(np.nanmean(rhos)) if rhos else float("nan")
        scores[weight] = score
        if np.isfinite(score) and score > best:
            best = score
            best_w = float(weight)
    return best_w, scores


def _permute_coverage(frame: pd.DataFrame, seed: int) -> np.ndarray:
    coverage = frame["bio_coverage"].to_numpy().copy()
    order = np.argsort(frame["inchikey"].to_numpy(), kind="mergesort")
    rng = np.random.default_rng(seed)
    shuffled = coverage[order].copy()
    perm = rng.permutation(len(shuffled))
    out = np.empty_like(coverage)
    out[order] = shuffled[perm]
    return out


def _isotonic_rows(adult: pd.DataFrame, transfer_endpoint: pd.DataFrame, endpoint: str, columns) -> list[pd.DataFrame]:
    rows = []
    adult_ep = adult.loc[adult["endpoint"] == endpoint]
    for representation in ("chem", "early"):
        src = adult_ep.loc[adult_ep["representation"] == representation]
        dst = transfer_endpoint.loc[transfer_endpoint["representation"] == representation]
        if len(src) < 10 or dst.empty:
            continue
        if np.nanstd(src["y_pred"]) < 1e-8:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(src["y_pred"].to_numpy(), src["y_true"].to_numpy())
        block = dst.copy()
        block["y_pred"] = iso.predict(dst["y_pred"].to_numpy())
        block["representation"] = f"{representation}_isotonic_adult_oof"
        block["model"] = "postprocess"
        rows.append(block[list(columns)])
    return rows


SLICES = (
    "all_pediatric",
    "structure_unseen",
    "structure_seen",
    "scaffold_unseen",
    "unseen_coverage_positive",
    "unseen_coverage_zero",
)


def _metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (phase, model, representation, endpoint), sub in frame.groupby(
        ["phase", "model", "representation", "endpoint"]
    ):
        slices = ("adult_scaffold_oof",) if phase == "adult_oof" else SLICES
        if phase == "target_reference":
            slices = ("target_reference_holdout",)
        for slice_name in slices:
            piece = sub if slice_name in {"adult_scaffold_oof", "target_reference_holdout"} else _slice_frame(sub, slice_name)
            if piece.empty:
                continue
            rec = _score_rows(piece, slice_name)
            rec.update(
                {
                    "phase": phase,
                    "model": model,
                    "representation": representation,
                    "endpoint": endpoint,
                }
            )
            records.append(rec)
    return pd.DataFrame(records)


def _eligible_endpoints(metrics: pd.DataFrame, min_adult: int, min_unseen: int) -> list[str]:
    adult = metrics.loc[
        (metrics["phase"] == "adult_oof")
        & (metrics["model"] == "ridge")
        & (metrics["representation"] == "chem")
    ]
    unseen = metrics.loc[
        (metrics["phase"] == "transfer")
        & (metrics["model"] == "ridge")
        & (metrics["representation"] == "chem")
        & (metrics["slice"] == "structure_unseen")
    ]
    keep = []
    for endpoint in sorted(unseen["endpoint"].unique()):
        n_adult = int(adult.loc[adult["endpoint"] == endpoint, "n"].iloc[0]) if endpoint in set(adult["endpoint"]) else 0
        n_unseen = int(unseen.loc[unseen["endpoint"] == endpoint, "n"].iloc[0])
        if n_adult >= min_adult and n_unseen >= min_unseen:
            keep.append(endpoint)
    return keep


def _lookup(metrics, endpoint, representation, slice_name, model="ridge", phase="transfer") -> float:
    hit = metrics.loc[
        (metrics["endpoint"] == endpoint)
        & (metrics["representation"] == representation)
        & (metrics["slice"] == slice_name)
        & (metrics["model"] == model)
        & (metrics["phase"] == phase)
    ]
    if hit.empty:
        return float("nan")
    return float(hit["spearman"].iloc[0])


def _bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> dict:
    base = bootstrap_statistic(values, np.mean, n_boot=n_boot, seed=seed)
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        base["p_bootstrap"] = float("nan")
        return base
    rng = np.random.default_rng(seed)
    means = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    base["p_bootstrap"] = float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0))))
    return base


def _endpoint_bootstrap(transfer: pd.DataFrame, endpoint: str, other: str, n_boot: int, seed: int) -> dict:
    sub = transfer.loc[(transfer["endpoint"] == endpoint) & (transfer["structure_seen"] == 0)]
    chem = sub.loc[(sub["representation"] == "chem") & (sub["model"] == "ridge")]
    other_model = "postprocess" if _is_post(other) else "ridge"
    alt = sub.loc[(sub["representation"] == other) & (sub["model"] == other_model)]
    merged = chem.merge(alt[["inchikey", "y_pred"]], on="inchikey", suffixes=("", "_other"))
    if len(merged) < 8:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_bootstrap": float("nan"), "n": int(len(merged))}
    return paired_bootstrap_delta(
        merged["y_true"].to_numpy(),
        merged["y_pred_other"].to_numpy(),
        merged["y_pred"].to_numpy(),
        spearman_rho,
        n_boot=n_boot,
        seed=seed,
    )


def summarize(pred_path: Path, cfg: dict, output_dir: Path, report_path: Path | None = None) -> dict:
    frame = pd.read_csv(pred_path)
    char_path = pred_path.parents[1] / "data_characterization.json"
    if not char_path.exists():
        char_path = output_dir / "data_characterization.json"
    characterization = json.loads(char_path.read_text()) if char_path.exists() else {}
    fused, fusion_info = _add_postprocess(frame, cfg)
    fused_path = output_dir / "predictions" / "predictions_with_fusion.csv.gz"
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(fused_path, index=False, compression="gzip")
    metrics = _metrics_table(fused)
    min_adult = int(cfg["data"]["min_adult_n"])
    min_unseen = int(cfg["data"]["min_unseen_n"])
    eligible = _eligible_endpoints(metrics, min_adult, min_unseen)
    metrics["eligible_primary"] = metrics["endpoint"].isin(eligible) & (metrics["slice"] == "structure_unseen")
    metrics_dir = output_dir / "metrics"
    stats_dir = output_dir / "statistics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "endpoint_metrics.csv", index=False)

    n_boot = int(cfg["transfer"]["bootstrap_n"])
    seed0 = int(cfg["transfer"]["bootstrap_seed"])
    transfer = fused.loc[fused["phase"] == "transfer"]
    contrasts = [
        "early",
        "bio",
        "missingness",
        "early_imputed_no_indicator",
        "gate_coverage",
        "gate_binary_coverage",
        "fixed_50_50_chem_early",
        "fixed_50_50_chem_bio",
        "late_global_chem_early",
        "late_endpoint_chem_early",
        "random_reliability",
        "permuted_reliability",
        "oracle_nondeployable",
        "chem_fallback",
        "coral_chem_transductive",
        "importance_weighted_chem_transductive",
        "chem_random_matched",
        "chem_ylabel_permuted_seed_101",
        "chem_isotonic_adult_oof",
        "early_isotonic_adult_oof",
    ]
    perm_names = [f"chem_permuted_bio_seed_{int(s)}" for s in cfg["transfer"]["permutation_seeds"]]
    endpoint_rows = []
    for i, endpoint in enumerate(sorted(transfer["endpoint"].unique())):
        row = {
            "endpoint": endpoint,
            "eligible": endpoint in eligible,
            "spearman_chem": _lookup(metrics, endpoint, "chem", "structure_unseen"),
            "n_unseen": _lookup_n(metrics, endpoint),
            "n_adult": _lookup_n(metrics, endpoint, phase="adult_oof", slice_name="adult_scaffold_oof"),
            "mean_coverage_unseen": _metric_field(metrics, endpoint, "mean_coverage"),
            "frac_scaffold_novel": _metric_field(metrics, endpoint, "frac_scaffold_unseen"),
            "mean_y_unseen": _metric_field(metrics, endpoint, "mean_y"),
            "adult_chem_spearman": _lookup(
                metrics, endpoint, "chem", "adult_scaffold_oof", phase="adult_oof"
            ),
        }
        for name in contrasts + perm_names:
            row[f"spearman_{name}"] = _lookup(
                metrics,
                endpoint,
                name,
                "structure_unseen",
                model="postprocess" if _is_post(name) else "ridge",
            )
            row[f"delta_{name}"] = row[f"spearman_{name}"] - row["spearman_chem"]
        if endpoint in eligible:
            boot = _endpoint_bootstrap(transfer, endpoint, "early", n_boot, seed0 + 100 + i)
            row["delta_early_ci_low"] = boot["ci_low"]
            row["delta_early_ci_high"] = boot["ci_high"]
            row["delta_early_p"] = boot["p_bootstrap"]
            gate = _endpoint_bootstrap(transfer, endpoint, "gate_coverage", n_boot, seed0 + 200 + i)
            row["delta_gate_ci_low"] = gate["ci_low"]
            row["delta_gate_ci_high"] = gate["ci_high"]
        endpoint_rows.append(row)
    endpoint_df = pd.DataFrame(endpoint_rows)
    endpoint_df.to_csv(stats_dir / "endpoint_contrasts.csv", index=False)

    eligible_df = endpoint_df.loc[endpoint_df["eligible"]].copy()
    perm_delta = eligible_df[[f"delta_{name}" for name in perm_names]].mean(axis=1)
    eligible_df["delta_perm_mean"] = eligible_df["spearman_early"] - (
        eligible_df[[f"spearman_{name}" for name in perm_names]].mean(axis=1)
    )
    claims = _claims(eligible_df, endpoint_df, metrics, cfg, fusion_info, characterization, n_boot, seed0)
    (metrics_dir / "primary_claims.json").write_text(json.dumps(claims, indent=2) + "\n")
    _write_supporting_tables(output_dir, metrics, endpoint_df, eligible_df, claims, characterization, fusion_info, cfg)
    make_figures(output_dir / "figures", metrics, endpoint_df, eligible_df, fused, characterization, claims)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_report(claims, eligible_df, characterization) + "\n")
    return claims


def _is_post(name: str) -> bool:
    return name in {
        "gate_coverage",
        "gate_binary_coverage",
        "fixed_50_50_chem_early",
        "fixed_50_50_chem_bio",
        "late_global_chem_early",
        "late_endpoint_chem_early",
        "random_reliability",
        "permuted_reliability",
        "oracle_nondeployable",
        "chem_fallback",
        "chem_isotonic_adult_oof",
        "early_isotonic_adult_oof",
    }


def _metric_field(metrics, endpoint, field: str) -> float:
    hit = metrics.loc[
        (metrics["endpoint"] == endpoint)
        & (metrics["representation"] == "chem")
        & (metrics["model"] == "ridge")
        & (metrics["phase"] == "transfer")
        & (metrics["slice"] == "structure_unseen")
    ]
    if hit.empty or field not in hit.columns:
        return float("nan")
    return float(hit[field].iloc[0])


def _lookup_n(metrics, endpoint, phase="transfer", slice_name="structure_unseen") -> int:
    hit = metrics.loc[
        (metrics["endpoint"] == endpoint)
        & (metrics["representation"] == "chem")
        & (metrics["model"] == "ridge")
        & (metrics["phase"] == phase)
        & (metrics["slice"] == slice_name)
    ]
    if hit.empty:
        return 0
    return int(hit["n"].iloc[0])


def _claims(eligible_df, endpoint_df, metrics, cfg, fusion_info, characterization, n_boot, seed0) -> dict:
    def pack(series, seed):
        stats = _bootstrap_mean(series.to_numpy(), n_boot, seed)
        return stats

    early = pack(eligible_df["delta_early"], seed0 + 1) if len(eligible_df) else _empty()
    gate = pack(eligible_df["delta_gate_coverage"], seed0 + 2) if len(eligible_df) else _empty()
    perm = pack(eligible_df["delta_perm_mean"], seed0 + 3) if len(eligible_df) else _empty()
    pvals = np.array([early["p_bootstrap"], perm["p_bootstrap"], gate["p_bootstrap"]], dtype=float)
    qvals = benjamini_hochberg(pvals)
    nt_early = int(np.sum(eligible_df["delta_early"] < 0)) if len(eligible_df) else 0
    nt_gate = int(np.sum(eligible_df["delta_gate_coverage"] < 0)) if len(eligible_df) else 0
    chem_macro = float(eligible_df["spearman_chem"].mean()) if len(eligible_df) else float("nan")
    early_macro = float(eligible_df["spearman_early"].mean()) if len(eligible_df) else float("nan")
    gate_macro = float(eligible_df["spearman_gate_coverage"].mean()) if len(eligible_df) else float("nan")
    ci_pos = lambda block: bool(np.isfinite(block["ci_low"]) and block["ci_low"] > 0)
    multimodal = bool(ci_pos(early) and ci_pos(perm))
    gate_gain = ci_pos(gate)
    gate_nt = bool(
        nt_gate < nt_early and np.isfinite(gate_macro) and np.isfinite(chem_macro) and gate_macro >= chem_macro - 0.005
    )
    wilcox = wilcoxon_signed_rank(eligible_df["delta_early"].to_numpy()) if len(eligible_df) else {"p": float("nan"), "note": "", "n": 0}
    sensitivity = []
    for threshold in cfg["data"].get("eligibility_sensitivity", []):
        sub = endpoint_df.loc[
            (endpoint_df["n_unseen"] >= int(threshold)) & (endpoint_df["n_adult"] >= int(cfg["data"]["min_adult_n"]))
        ]
        sensitivity.append(
            {
                "min_unseen_n": int(threshold),
                "n_endpoints": int(len(sub)),
                "macro_delta_early": float(sub["delta_early"].mean()) if len(sub) else float("nan"),
                "macro_delta_gate": float(sub["delta_gate_coverage"].mean()) if len(sub) else float("nan"),
            }
        )
    moderators = _moderators(eligible_df)
    return {
        "status": "computed_from_saved_predictions",
        "primary_slice": "structure_unseen",
        "primary_model": "ridge",
        "ridge_alpha": float(cfg["models"]["ridge"]["alpha"]),
        "alpha_mode": cfg["models"]["ridge"].get("alpha_mode", "prespecified"),
        "n_endpoints_total": int(endpoint_df["endpoint"].nunique()) if len(endpoint_df) else 0,
        "n_endpoints_eligible": int(len(eligible_df)),
        "eligible_endpoints": eligible_df["endpoint"].tolist() if len(eligible_df) else [],
        "excluded_endpoints": endpoint_df.loc[~endpoint_df["eligible"], "endpoint"].tolist() if len(endpoint_df) else [],
        "macro_spearman_chem": chem_macro,
        "macro_spearman_early": early_macro,
        "macro_spearman_gate": gate_macro,
        "delta_early_minus_chem": early,
        "delta_early_minus_perm_mean": perm,
        "delta_gate_minus_chem": gate,
        "bh_q_early_perm_gate": [None if not np.isfinite(v) else float(v) for v in qvals],
        "negative_transfer_endpoints_early": nt_early,
        "negative_transfer_endpoints_gate": nt_gate,
        "multimodal_benefit_supported": multimodal,
        "gate_macro_gain_supported": gate_gain,
        "gate_reduces_negative_transfer": gate_nt,
        "wilcoxon_early_minus_chem_p": wilcox["p"],
        "wilcoxon_note": wilcox["note"],
        "eligibility_sensitivity": sensitivity,
        "moderators_exploratory": moderators,
        "late_global_weight": fusion_info["late_global_weight"],
        "n_structure_overlap": characterization.get("n_structure_overlap"),
        "n_structure_unseen_compounds": characterization.get("n_structure_unseen"),
        "bio_mean_missingness": characterization.get("bio_mean_missingness"),
        "bio_n_targets": characterization.get("bio_n_targets"),
        "interpretation": _interpretation(early, perm, gate, multimodal, gate_gain, gate_nt, nt_early, nt_gate),
    }


def _moderators(eligible_df: pd.DataFrame) -> list[dict]:
    """Endpoint-level associations. n is the number of endpoints, not compounds."""
    if len(eligible_df) < 5:
        return []
    rows = []
    y = eligible_df["delta_early"].to_numpy()
    for name in ("mean_coverage_unseen", "frac_scaffold_novel", "adult_chem_spearman", "n_unseen", "mean_y_unseen"):
        if name not in eligible_df.columns:
            continue
        rho = spearman_rho(y, eligible_df[name].to_numpy())
        rows.append(
            {
                "moderator": name,
                "spearman_with_early_minus_chem": rho,
                "n_endpoints": int(len(eligible_df)),
                "note": "Exploratory association across endpoints. Not a causal effect.",
            }
        )
    return rows


def _empty() -> dict:
    return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_bootstrap": float("nan"), "n": 0}


def _interpretation(early, perm, gate, multimodal, gate_gain, gate_nt, nt_early, nt_gate) -> str:
    parts = [
        _sentence("Early fusion minus chemistry", early),
        _sentence("Early fusion minus the mean of the bio-permutation seeds", perm),
        _sentence("Coverage gate minus chemistry", gate),
    ]
    if multimodal:
        parts.append(
            "The pre-specified rule for a multimodal benefit is met: both contrasts have 95% intervals above 0."
        )
    else:
        parts.append(
            "The pre-specified rule for a multimodal benefit is not met. A benefit required both the early-versus-chemistry interval and the early-versus-permutation interval to lie above 0."
        )
    if gate_gain:
        parts.append("The coverage gate has a macro interval above 0 relative to chemistry.")
    elif gate_nt:
        parts.append(
            f"The coverage gate has fewer negative-transfer endpoints ({nt_gate}) than early fusion ({nt_early}) and does not fall more than 0.005 Spearman below chemistry. That is a reduction in negative-transfer count, not a demonstrated accuracy gain."
        )
    else:
        parts.append(
            f"The coverage gate does not meet the gain rule or the negative-transfer rule (negative-transfer endpoints: gate {nt_gate}, early fusion {nt_early})."
        )
    parts.append(
        "These statements are about smoothed adverse-event incidence in pediatric-only trials, on structures absent from the adult-trial training set. They are not clinical pediatric-toxicity performance."
    )
    return " ".join(parts)


def _sentence(name: str, block: dict) -> str:
    delta, lo, hi = block.get("estimate"), block.get("ci_low"), block.get("ci_high")
    if not np.isfinite(delta):
        return f"{name} is not estimable."
    where = "includes 0"
    if lo > 0:
        where = "lies above 0"
    elif hi < 0:
        where = "lies below 0"
    return f"{name}: delta {_fmt(delta)} (95% CI {_fmt(lo)} to {_fmt(hi)}; the interval {where})."


def _report(claims: dict, eligible_df: pd.DataFrame, characterization: dict) -> str:
    lines = [
        "# Results",
        "",
        "Numbers in this file are written by `src/toxtransfer/analysis/summarize.py` from `results/predictions/predictions.csv.gz`.",
        "Do not edit the numeric sentences by hand. Re-run the summary script instead.",
        "",
        "## Primary estimand",
        "",
        "Adult-only-trial to pediatric-only-trial transfer of smoothed logit adverse-event incidence.",
        "Primary rows are pediatric-trial structures whose InChIKey is absent from the adult-trial set.",
        f"Eligible endpoints: {claims['n_endpoints_eligible']} of {claims['n_endpoints_total']} "
        f"(minimum adult n and minimum structure-unseen n are in the analysis plan).",
        f"Structure overlap (all endpoints, compound level): {characterization.get('n_structure_overlap')} shared, "
        f"{characterization.get('n_structure_unseen')} pediatric-only.",
        f"ChEMBL target-profile width: {characterization.get('bio_n_targets')} targets; "
        f"mean missingness {characterization.get('bio_mean_missingness')}.",
        "",
        "## Confirmatory contrasts",
        "",
        f"Ridge alpha mode: {claims['alpha_mode']}; alpha {claims['ridge_alpha']}.",
        f"Macro Spearman, chemistry: {_fmt(claims['macro_spearman_chem'])}.",
        f"Macro Spearman, early fusion: {_fmt(claims['macro_spearman_early'])}.",
        f"Macro Spearman, coverage gate: {_fmt(claims['macro_spearman_gate'])}.",
        "",
        claims["interpretation"],
        "",
        f"Benjamini–Hochberg q-values for the three macro bootstrap tests "
        f"(early−chem, early−permutation, gate−chem): {claims['bh_q_early_perm_gate']}.",
        f"Wilcoxon signed-rank p on endpoint early−chem deltas: {_fmt(claims['wilcoxon_early_minus_chem_p'])}. "
        f"{claims['wilcoxon_note']}",
        "",
        "Eligibility sensitivity (same predictions, different pre-specified unseen-n cutoffs):",
    ]
    for row in claims.get("eligibility_sensitivity", []):
        lines.append(
            f"- min unseen n={row['min_unseen_n']}: {row['n_endpoints']} endpoints, "
            f"macro early−chem {_fmt(row['macro_delta_early'])}, "
            f"macro gate−chem {_fmt(row['macro_delta_gate'])}"
        )
    lines.extend(
        [
            "",
            "Exploratory moderator associations (Spearman across endpoints, not compounds):",
        ]
    )
    for row in claims.get("moderators_exploratory", []):
        lines.append(
            f"- {row['moderator']}: {_fmt(row['spearman_with_early_minus_chem'])} "
            f"(n={row['n_endpoints']}). {row['note']}"
        )
    lines.extend(
        [
        "",
        "Multimodal benefit supported: "
        + ("yes" if claims["multimodal_benefit_supported"] else "no")
        + ".",
        "Coverage-gate macro gain supported: "
        + ("yes" if claims["gate_macro_gain_supported"] else "no")
        + ".",
        "Coverage gate reduces the negative-transfer count under the pre-specified rule: "
        + ("yes" if claims["gate_reduces_negative_transfer"] else "no")
        + ".",
        "",
        "## Endpoint deltas (eligible, structure-unseen, early fusion minus chemistry)",
        "",
        ]
    )
    if len(eligible_df):
        show = eligible_df.sort_values("delta_early")[
            ["endpoint", "n_unseen", "spearman_chem", "spearman_early", "delta_early"]
        ]
        for _, row in show.iterrows():
            lines.append(
                f"- {row['endpoint']}: n={int(row['n_unseen'])}, chem {_fmt(row['spearman_chem'])}, "
                f"early {_fmt(row['spearman_early'])}, delta {_fmt(row['delta_early'])}"
            )
    lines.append("")
    return "\n".join(lines)


def _write_supporting_tables(output_dir, metrics, endpoint_df, eligible_df, claims, characterization, fusion_info, cfg):
    table_dir = output_dir / "tables"
    paper = output_dir.parents[0] / "paper" / "tables" if output_dir.name != "smoke" else output_dir / "paper_tables"
    # When output is results/, parents[0] is the repo if output is results. 
    # output_dir is results or results/smoke. paper tables for the real run go to repo/paper/tables.
    if output_dir.name == "smoke":
        paper = output_dir / "paper_tables"
    else:
        paper = output_dir.parent / "paper" / "tables"
    inclusion = []
    for row in characterization.get("endpoints", []):
        inclusion.append(
            {
                "endpoint": row["endpoint"],
                "n_adult": row["n_adult"],
                "n_pediatric": row["n_pediatric"],
                "n_structure_unseen": row["n_structure_unseen"],
                "included_primary": row["endpoint"] in claims["eligible_endpoints"],
                "reason": "meets n thresholds" if row["endpoint"] in claims["eligible_endpoints"] else "below pre-specified n",
            }
        )
    write_table(pd.DataFrame(inclusion), "endpoint_inclusion", table_dir, paper)
    write_table(
        pd.DataFrame(
            [
                {
                    "n_adult_structures": characterization.get("n_adult_structures"),
                    "n_pediatric_structures": characterization.get("n_pediatric_structures"),
                    "n_structure_overlap": characterization.get("n_structure_overlap"),
                    "n_structure_unseen": characterization.get("n_structure_unseen"),
                    "n_scaffold_overlap": characterization.get("n_scaffold_overlap"),
                    "bio_n_targets": characterization.get("bio_n_targets"),
                    "bio_mean_missingness": characterization.get("bio_mean_missingness"),
                    "bio_frac_any_observed": characterization.get("bio_frac_any_observed"),
                    "n_chem_features": characterization.get("n_chem_features"),
                }
            ]
        ),
        "dataset_characteristics",
        table_dir,
        paper,
    )
    primary = metrics.loc[
        (metrics["slice"] == "structure_unseen") & (metrics["endpoint"].isin(claims["eligible_endpoints"]))
    ]
    macro_rows = []
    for (model, representation), sub in primary.groupby(["model", "representation"]):
        macro_rows.append(
            {
                "model": model,
                "representation": representation,
                "macro_spearman": float(sub["spearman"].mean()),
                "median_spearman": float(sub["spearman"].median()),
                "n_endpoints": int(sub["endpoint"].nunique()),
            }
        )
    macro = pd.DataFrame(macro_rows).sort_values(["model", "macro_spearman"], ascending=[True, False])
    write_table(macro, "transfer_macro", table_dir, paper)
    write_table(endpoint_df, "endpoint_contrasts", table_dir, paper)
    write_table(pd.DataFrame(fusion_info["late_endpoint_weights"]), "late_fusion_weights", table_dir, paper)
    claim_table = pd.DataFrame(
        [
            {
                "contrast": "early_minus_chem",
                "delta": claims["delta_early_minus_chem"]["estimate"],
                "ci_low": claims["delta_early_minus_chem"]["ci_low"],
                "ci_high": claims["delta_early_minus_chem"]["ci_high"],
                "p_bootstrap": claims["delta_early_minus_chem"]["p_bootstrap"],
            },
            {
                "contrast": "early_minus_perm_mean",
                "delta": claims["delta_early_minus_perm_mean"]["estimate"],
                "ci_low": claims["delta_early_minus_perm_mean"]["ci_low"],
                "ci_high": claims["delta_early_minus_perm_mean"]["ci_high"],
                "p_bootstrap": claims["delta_early_minus_perm_mean"]["p_bootstrap"],
            },
            {
                "contrast": "gate_minus_chem",
                "delta": claims["delta_gate_minus_chem"]["estimate"],
                "ci_low": claims["delta_gate_minus_chem"]["ci_low"],
                "ci_high": claims["delta_gate_minus_chem"]["ci_high"],
                "p_bootstrap": claims["delta_gate_minus_chem"]["p_bootstrap"],
            },
        ]
    )
    write_table(claim_table, "primary_contrasts", table_dir, paper)
    write_table(pd.DataFrame(claims.get("eligibility_sensitivity", [])), "eligibility_sensitivity", table_dir, paper)
    write_table(pd.DataFrame(claims.get("moderators_exploratory", [])), "moderator_associations", table_dir, paper)
    # Silence unused in case a caller inspects cfg.
    _ = cfg
