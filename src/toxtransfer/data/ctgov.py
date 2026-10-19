"""Load the ClinicalTrials.gov structure-level handoff tables.

The files are derived aggregates, not a download this module performs. Place
them under the configured raw directory. Chemistry is recomputed from SMILES
so imputation is not taken from a pre-joined matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from toxtransfer.features.chemistry import featurize_mols, murcko_scaffold, standardize_mol


REQUIRED = {
    "ctgov_adult_structure_outcome.csv": [
        "inchikey",
        "outcome",
        "num_affected",
        "num_at_risk",
        "logit_incidence",
        "estimate_type",
        "age_bucket",
    ],
    "ctgov_pediatric_structure_outcome.csv": [
        "inchikey",
        "outcome",
        "num_affected",
        "num_at_risk",
        "logit_incidence",
        "estimate_type",
        "age_bucket",
    ],
    "drug_name_mapping.csv": ["inchikey", "canonical_smiles", "status"],
    "chembl_target_profile.csv": ["inchikey"],
}


def smoothed_logit(affected: np.ndarray, at_risk: np.ndarray) -> np.ndarray:
    """(a + 0.5) / (n + 1), then the logit. This is the pre-specified label."""
    incidence = (affected + 0.5) / (at_risk + 1.0)
    incidence = np.clip(incidence, 1e-6, 1.0 - 1e-6)
    return np.log(incidence / (1.0 - incidence))


def _require(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Place the handoff CSV in {path.parent}. "
            "Raw tables are gitignored; see data/README.md."
        )
    frame = pd.read_csv(path)
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    return frame


def load_ctgov_tables(raw_dir: Path) -> dict:
    """Join labels, one SMILES per InChIKey, and the ChEMBL target profile."""
    adult = _require(raw_dir / "ctgov_adult_structure_outcome.csv", REQUIRED["ctgov_adult_structure_outcome.csv"])
    ped = _require(
        raw_dir / "ctgov_pediatric_structure_outcome.csv",
        REQUIRED["ctgov_pediatric_structure_outcome.csv"],
    )
    mapping = _require(raw_dir / "drug_name_mapping.csv", REQUIRED["drug_name_mapping.csv"])
    bio = _require(raw_dir / "chembl_target_profile.csv", REQUIRED["chembl_target_profile.csv"])

    for frame, domain in ((adult, "adult"), (ped, "pediatric")):
        bad = sorted(set(frame["estimate_type"].dropna()) - {"clinical_trial_smoothed_incidence"})
        if bad:
            raise RuntimeError(
                f"{domain} labels contain estimate types {bad}. "
                "FAERS or other reporting fractions must not be mixed into this endpoint."
            )
        if not (frame["age_bucket"] == domain).all():
            raise RuntimeError(f"{domain} file has rows whose age_bucket is not '{domain}'.")

    adult = adult.copy()
    ped = ped.copy()
    adult["domain"] = "adult"
    ped["domain"] = "pediatric"
    labels = pd.concat([adult, ped], ignore_index=True)
    dup = labels.duplicated(["inchikey", "outcome", "domain"])
    if dup.any():
        raise RuntimeError(f"{int(dup.sum())} duplicate structure-endpoint-domain rows.")
    labels["y_stored"] = labels["logit_incidence"].astype(float)
    labels["y"] = smoothed_logit(
        labels["num_affected"].astype(float).to_numpy(),
        labels["num_at_risk"].astype(float).to_numpy(),
    )
    logit_gap = float(np.nanmax(np.abs(labels["y"] - labels["y_stored"])))

    smiles_rows = mapping.loc[
        mapping["canonical_smiles"].fillna("").astype(str).str.len().gt(0)
        & mapping["inchikey"].fillna("").astype(str).str.len().gt(0),
        ["inchikey", "canonical_smiles"],
    ].copy()
    nunique = smiles_rows.groupby("inchikey")["canonical_smiles"].nunique()
    conflicts = int((nunique > 1).sum())
    smiles = (
        smiles_rows.groupby("inchikey")["canonical_smiles"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("smiles")
        .reset_index()
    )

    tgt_cols = [c for c in bio.columns if c.startswith("tgt_")]
    miss_cols = [c for c in bio.columns if c.startswith("miss_")]
    if not tgt_cols or len(tgt_cols) != len(miss_cols):
        raise RuntimeError("ChEMBL profile must contain paired tgt_ and miss_ columns.")
    if bio["inchikey"].duplicated().any():
        raise RuntimeError("Duplicate InChIKeys in the ChEMBL target profile.")

    needed = set(labels["inchikey"])
    have = set(smiles["inchikey"]) & set(bio["inchikey"])
    missing = needed - have
    if missing:
        raise RuntimeError(
            f"{len(missing)} labeled structures have no SMILES or no bio profile. "
            "Refusing to drop them silently."
        )
    compounds = smiles[smiles["inchikey"].isin(needed)].merge(
        bio[["inchikey"] + tgt_cols + miss_cols], on="inchikey", how="inner"
    )
    compounds = compounds.sort_values("inchikey").reset_index(drop=True)
    adult_ids = set(adult["inchikey"])
    ped_ids = set(ped["inchikey"])
    return {
        "compounds": compounds,
        "labels": labels[["inchikey", "outcome", "domain", "y", "y_stored"]],
        "tgt_cols": tgt_cols,
        "miss_cols": miss_cols,
        "adult_ids": adult_ids,
        "pediatric_ids": ped_ids,
        "smiles_conflicts": conflicts,
        "logit_max_abs_diff": logit_gap,
        "estimate_types": ["clinical_trial_smoothed_incidence"],
        "source": "ctgov_handoff",
    }


def featurize_compounds(table: pd.DataFrame, tgt_cols: list[str], miss_cols: list[str]) -> dict:
    """Recompute Morgan fingerprints, descriptors, and scaffolds from SMILES."""
    mols = []
    keep = []
    for i, smiles in enumerate(table["smiles"].tolist()):
        mol = standardize_mol(smiles)
        if mol is None:
            continue
        mols.append(mol)
        keep.append(i)
    failure = len(table) - len(keep)
    if len(table) and failure / len(table) > 0.05:
        raise RuntimeError(f"{failure} of {len(table)} SMILES failed standardization (>5%).")
    kept = table.iloc[keep].reset_index(drop=True)
    fingerprints, descriptors, names, chem_continuous = featurize_mols(mols)
    scaffolds = np.array([murcko_scaffold(mol) for mol in mols])
    longest = max((len(str(s)) for s in scaffolds), default=0)
    if longest > 2048:
        raise RuntimeError(f"Scaffold SMILES length {longest} exceeds the cache width.")
    chem = np.hstack([fingerprints, descriptors])
    tgt = kept[tgt_cols].to_numpy(dtype=np.float64)
    miss = kept[miss_cols].to_numpy(dtype=np.float64)
    values_nan = tgt.copy()
    values_nan[miss == 1] = np.nan
    filled = tgt.copy()
    filled[miss == 1] = 0.0
    bio = np.hstack([filled, miss])
    bio_continuous = np.array([True] * tgt.shape[1] + [False] * miss.shape[1])
    return {
        "inchikey": kept["inchikey"].to_numpy(),
        "scaffold": scaffolds,
        "chem": chem,
        "chem_continuous": chem_continuous,
        "bio": bio,
        "bio_values_nan": values_nan,
        "miss": miss,
        "bio_continuous": bio_continuous,
        "n_targets": int(tgt.shape[1]),
        "descriptor_names": names,
        "featurization_failures": int(failure),
        "kept_inchikeys": set(kept["inchikey"].tolist()),
    }
