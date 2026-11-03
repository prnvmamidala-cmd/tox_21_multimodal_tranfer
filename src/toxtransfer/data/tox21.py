"""Tox21 loader. Assay labels are in-vitro activities, not clinical outcomes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from toxtransfer.features.chemistry import murcko_scaffold, standardize_mol

TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
NR_TASKS = [task for task in TASKS if task.startswith("NR-")]
SR_TASKS = [task for task in TASKS if task.startswith("SR-")]


def prepare_tox21(path: Path) -> pd.DataFrame:
    """Standardize structures and collapse duplicate SMILES the way the audit describes.

    Global median imputation is intentionally not done here.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing Tox21 file at {path}. Run scripts/download_or_prepare_data.py.")
    raw = pd.read_csv(path)
    missing = [task for task in TASKS if task not in raw.columns]
    if missing or "smiles" not in raw.columns:
        raise ValueError(f"Unexpected Tox21 columns; missing {missing}")
    std = raw["smiles"].map(lambda s: _std_smiles(s))
    frame = raw.copy()
    frame["std_smiles"] = std
    n_unparseable = int(frame["std_smiles"].isna().sum())
    frame = frame.dropna(subset=["std_smiles"])
    dedup = frame.groupby("std_smiles", as_index=False)[TASKS].mean()
    for task in TASKS:
        values = dedup[task].to_numpy()
        dedup[task] = np.where(np.isnan(values), np.nan, np.where(values > 0.5, 1.0, np.where(values < 0.5, 0.0, np.nan)))
    dedup = dedup[dedup[TASKS].notna().any(axis=1)].reset_index(drop=True)
    mols = []
    keep = []
    scaffolds = []
    for i, smiles in enumerate(dedup["std_smiles"].tolist()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mols.append(mol)
        keep.append(i)
        scaffolds.append(murcko_scaffold(mol))
    dedup = dedup.iloc[keep].reset_index(drop=True)
    from toxtransfer.features.chemistry import featurize_mols

    fingerprints, _, _, _ = featurize_mols(mols)
    dedup["scaffold"] = scaffolds
    dedup.attrs["n_unparseable"] = n_unparseable
    dedup.attrs["fingerprints"] = fingerprints
    return dedup


def _std_smiles(smiles: str):
    mol = standardize_mol(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
