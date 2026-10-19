"""Vector figures. Colors are the Okabe–Ito palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHEM = "#0072B2"
EARLY = "#009E73"
GATE = "#CC79A7"
PERM = "#D55E00"
BIO = "#E69F00"
NEUTRAL = "#000000"


def _save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_figures(directory, metrics, endpoint_df, eligible_df, fused, characterization, claims) -> None:
    _design(directory)
    _composition(directory, characterization)
    _coverage(directory, fused, characterization)
    _per_endpoint(directory, eligible_df)
    _forest(directory, eligible_df)
    _permutations(directory, eligible_df)
    _transfer_gap(directory, metrics, claims)
    _moderator(directory, eligible_df, fused, "scaffold", "08_gain_vs_scaffold_novelty")
    _moderator(directory, eligible_df, fused, "coverage", "09_gain_vs_bio_coverage")
    _calibration(directory, fused, claims)
    _risk(directory, fused, claims)
    _ablation(directory, metrics, claims)
    _failures(directory, eligible_df)


def _design(directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.62, "Adult-only trials\nlabels: smoothed logit AE"),
        (0.02, 0.22, "Pediatric-only trials\nstructure-unseen primary"),
        (0.38, 0.62, "Chemistry\nMorgan r=2, 2048\n+ RDKit descriptors"),
        (0.38, 0.22, "ChEMBL target profile\n+ missingness\npermutation controls"),
        (0.70, 0.42, "Ridge, trees\nearly / late / gate\nno target-label tuning"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.26, 0.28, fill=False, lw=1.2))
        ax.text(x + 0.13, y + 0.14, text, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.38, 0.76), xytext=(0.28, 0.76), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.38, 0.36), xytext=(0.28, 0.36), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.70, 0.56), xytext=(0.64, 0.76), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.70, 0.50), xytext=(0.64, 0.36), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Adult-trial to pediatric-trial incidence transfer")
    _save(fig, directory, "01_design_schematic")


def _composition(directory: Path, characterization: dict) -> None:
    rows = characterization.get("endpoints") or []
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    if not rows:
        ax.text(0.5, 0.5, "No endpoint counts", ha="center")
    else:
        names = [row["endpoint"] for row in rows]
        y = np.arange(len(names))
        ax.barh(y - 0.15, [row["n_adult"] for row in rows], height=0.3, color=CHEM, label="Adult-trial")
        ax.barh(y + 0.15, [row["n_pediatric"] for row in rows], height=0.3, color=BIO, label="Pediatric-trial")
        ax.set_yticks(y, names, fontsize=7)
        ax.set_xlabel("Structures with a label")
        ax.legend(frameon=False)
    ax.set_title("Dataset composition by endpoint")
    _save(fig, directory, "02_dataset_composition")


def _coverage(directory: Path, fused: pd.DataFrame, characterization: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ped = fused.loc[
        (fused["phase"] == "transfer")
        & (fused["representation"] == "chem")
        & (fused["model"] == "ridge")
        & (fused["endpoint"] == fused.loc[fused["phase"] == "transfer", "endpoint"].iloc[0])
    ]
    ax.hist(ped["bio_coverage"], bins=10, color=CHEM, edgecolor="white")
    ax.set_xlabel("Fraction of ChEMBL targets observed")
    ax.set_ylabel("Pediatric-trial structures")
    mean_miss = characterization.get("bio_mean_missingness")
    ax.set_title(f"Biological coverage (mean missingness {mean_miss})")
    _save(fig, directory, "03_bio_coverage")


def _per_endpoint(directory: Path, eligible: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    if len(eligible):
        ax.scatter(eligible["spearman_chem"], eligible["spearman_early"], color=EARLY, zorder=3)
        for _, row in eligible.iterrows():
            ax.annotate(row["endpoint"], (row["spearman_chem"], row["spearman_early"]), fontsize=6, alpha=0.8)
        lo = min(eligible["spearman_chem"].min(), eligible["spearman_early"].min(), 0) - 0.05
        hi = max(eligible["spearman_chem"].max(), eligible["spearman_early"].max(), 0) + 0.05
        ax.plot([lo, hi], [lo, hi], color=NEUTRAL, lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Chemistry Spearman")
    ax.set_ylabel("Early fusion Spearman")
    ax.set_title("Structure-unseen pediatric-trial endpoints")
    _save(fig, directory, "04_chem_vs_multimodal")


def _forest(directory: Path, eligible: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if len(eligible):
        ordered = eligible.sort_values("delta_early")
        y = np.arange(len(ordered))
        ax.axvline(0, color=NEUTRAL, lw=0.8)
        xerr = None
        if "delta_early_ci_low" in ordered:
            low = ordered["delta_early"] - ordered["delta_early_ci_low"]
            high = ordered["delta_early_ci_high"] - ordered["delta_early"]
            xerr = np.vstack([low, high])
        ax.errorbar(ordered["delta_early"], y, xerr=xerr, fmt="o", color=EARLY, ecolor=EARLY)
        ax.set_yticks(y, ordered["endpoint"], fontsize=7)
    ax.set_xlabel("Early fusion minus chemistry (Spearman)")
    ax.set_title("Paired endpoint differences")
    _save(fig, directory, "05_paired_differences")


def _permutations(directory: Path, eligible: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    perm_cols = [c for c in eligible.columns if c.startswith("spearman_chem_permuted_bio_seed_")]
    if len(eligible) and perm_cols:
        ax.scatter(np.zeros(len(eligible)), eligible["spearman_chem"], color=CHEM, label="Chemistry")
        ax.scatter(np.ones(len(eligible)), eligible["spearman_early"], color=EARLY, label="True bio, early")
        stacked = eligible[perm_cols].to_numpy().ravel()
        ax.scatter(np.full(stacked.shape, 2.0), stacked, color=PERM, label="Permuted bio", alpha=0.7)
        ax.set_xticks([0, 1, 2], ["Chemistry", "Early, true", "Early, permuted"])
        ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Structure-unseen Spearman")
    ax.set_title("True versus permuted biological features")
    _save(fig, directory, "06_true_vs_permuted")


def _transfer_gap(directory: Path, metrics: pd.DataFrame, claims: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    eligible = set(claims.get("eligible_endpoints") or [])
    reps = ["chem", "bio", "early", "missingness"]
    adult_vals, transfer_vals = [], []
    for rep in reps:
        adult = metrics.loc[
            (metrics["phase"] == "adult_oof")
            & (metrics["model"] == "ridge")
            & (metrics["representation"] == rep)
            & (metrics["endpoint"].isin(eligible))
        ]
        transfer = metrics.loc[
            (metrics["phase"] == "transfer")
            & (metrics["slice"] == "structure_unseen")
            & (metrics["model"] == "ridge")
            & (metrics["representation"] == rep)
            & (metrics["endpoint"].isin(eligible))
        ]
        adult_vals.append(adult["spearman"].mean() if len(adult) else np.nan)
        transfer_vals.append(transfer["spearman"].mean() if len(transfer) else np.nan)
    x = np.arange(len(reps))
    ax.bar(x - 0.15, adult_vals, width=0.3, color=CHEM, label="Adult scaffold OOF")
    ax.bar(x + 0.15, transfer_vals, width=0.3, color=PERM, label="Structure-unseen transfer")
    ax.set_xticks(x, reps, rotation=15)
    ax.set_ylabel("Macro Spearman")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("In-domain versus transfer")
    _save(fig, directory, "07_transfer_gap")


def _moderator(directory, eligible, fused, kind: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if len(eligible):
        if kind == "coverage":
            x = []
            for endpoint in eligible["endpoint"]:
                sub = fused.loc[
                    (fused["phase"] == "transfer")
                    & (fused["endpoint"] == endpoint)
                    & (fused["representation"] == "chem")
                    & (fused["model"] == "ridge")
                    & (fused["structure_seen"] == 0)
                ]
                x.append(sub["bio_coverage"].mean() if len(sub) else np.nan)
            xlabel = "Mean bio coverage on structure-unseen compounds"
        else:
            x = []
            for endpoint in eligible["endpoint"]:
                sub = fused.loc[
                    (fused["phase"] == "transfer")
                    & (fused["endpoint"] == endpoint)
                    & (fused["representation"] == "chem")
                    & (fused["model"] == "ridge")
                    & (fused["structure_seen"] == 0)
                ]
                x.append(1.0 - sub["scaffold_seen"].mean() if len(sub) else np.nan)
            xlabel = "Fraction of structure-unseen compounds with a novel scaffold"
        ax.scatter(x, eligible["delta_early"], color=EARLY)
        ax.axhline(0, color=NEUTRAL, lw=0.8)
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Early fusion minus chemistry")
    ax.set_title("Association, not a causal effect")
    _save(fig, directory, stem)


def _calibration(directory, fused, claims) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    eligible = claims.get("eligible_endpoints") or []
    if not eligible:
        ax.set_title("Calibration")
        _save(fig, directory, "10_calibration")
        return
    endpoint = eligible[0]
    for rep, color, label in (("chem", CHEM, "Chemistry"), ("early", EARLY, "Early fusion")):
        sub = fused.loc[
            (fused["phase"] == "transfer")
            & (fused["endpoint"] == endpoint)
            & (fused["model"] == "ridge")
            & (fused["representation"] == rep)
            & (fused["structure_seen"] == 0)
        ].sort_values("y_pred")
        if len(sub) < 10:
            continue
        n_bins = min(8, max(2, len(sub) // 5))
        edges = np.linspace(0, len(sub), n_bins + 1).astype(int)
        px, py = [], []
        for start, stop in zip(edges[:-1], edges[1:]):
            piece = sub.iloc[start:stop]
            if piece.empty:
                continue
            px.append(piece["y_pred"].mean())
            py.append(piece["y_true"].mean())
        ax.plot(px, py, marker="o", color=color, label=label)
    lims = ax.get_xlim()
    ax.plot(lims, lims, color=NEUTRAL, lw=0.7)
    ax.set_xlabel("Mean predicted logit")
    ax.set_ylabel("Mean observed logit")
    ax.set_title(f"Calibration on {endpoint} (largest listed eligible endpoint)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, directory, "10_calibration")


def _risk(directory, fused, claims) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fracs = [0.4, 0.6, 0.8, 1.0]
    eligible = claims.get("eligible_endpoints") or []
    ys = []
    for frac in fracs:
        scores = []
        for endpoint in eligible:
            chem = fused.loc[
                (fused.phase == "transfer")
                & (fused.endpoint == endpoint)
                & (fused.model == "ridge")
                & (fused.representation == "chem")
                & (fused.structure_seen == 0)
            ]
            early = fused.loc[
                (fused.phase == "transfer")
                & (fused.endpoint == endpoint)
                & (fused.model == "ridge")
                & (fused.representation == "early")
                & (fused.structure_seen == 0)
            ][["inchikey", "y_pred"]]
            merged = chem.merge(early, on="inchikey", suffixes=("_chem", "_early"))
            if len(merged) < 15:
                continue
            disagree = (merged["y_pred_chem"] - merged["y_pred_early"]).abs()
            keep = merged.nsmallest(max(8, int(round(len(merged) * frac))), "tmp") if False else merged.assign(disagree=disagree).nsmallest(
                max(8, int(round(len(merged) * frac))), "disagree"
            )
            # Low disagreement is high confidence in the chem/early agreement.
            rho = _rho(keep["y_true"], keep["y_pred_chem"])
            if np.isfinite(rho):
                scores.append(rho)
        ys.append(float(np.mean(scores)) if scores else np.nan)
    ax.plot(fracs, ys, marker="o", color=GATE)
    ax.set_xlabel("Coverage (lowest chem/early disagreement retained)")
    ax.set_ylabel("Macro Spearman of chemistry")
    ax.set_ylim(-0.2, 1.0)
    ax.set_title("Selective prediction by representation disagreement")
    _save(fig, directory, "11_risk_coverage")


def _rho(y, pred) -> float:
    from toxtransfer.evaluation.metrics import spearman_rho

    return spearman_rho(np.asarray(y), np.asarray(pred))


def _ablation(directory, metrics, claims) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    eligible = set(claims.get("eligible_endpoints") or [])
    sub = metrics.loc[
        (metrics["phase"] == "transfer")
        & (metrics["slice"] == "structure_unseen")
        & (metrics["endpoint"].isin(eligible))
        & (metrics["model"].isin(["ridge", "postprocess"]))
    ]
    if len(sub):
        macro = sub.groupby("representation")["spearman"].mean().sort_values()
        colors = [EARLY if "early" in name or "gate" in name else CHEM for name in macro.index]
        ax.barh(macro.index, macro.values, color=colors)
        ax.tick_params(axis="y", labelsize=7)
    ax.axvline(0, color=NEUTRAL, lw=0.6)
    ax.set_xlabel("Macro Spearman, structure-unseen")
    ax.set_title("Ablation and control summary")
    _save(fig, directory, "12_ablation_summary")


def _failures(directory, eligible: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if len(eligible):
        worst = eligible.nsmallest(min(8, len(eligible)), "delta_early")
        ax.barh(worst["endpoint"], worst["delta_early"], color=PERM)
        ax.axvline(0, color=NEUTRAL, lw=0.8)
    ax.set_xlabel("Early fusion minus chemistry")
    ax.set_title("Largest negative early-fusion deltas")
    _save(fig, directory, "13_failure_cases")
