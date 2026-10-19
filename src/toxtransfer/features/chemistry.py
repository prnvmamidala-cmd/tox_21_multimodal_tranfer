"""Morgan fingerprints, RDKit descriptors, and Bemis–Murcko scaffolds.

Descriptor transforms that do not use labels (log1p of Ipc) are applied before
splitting. Imputation and scaling are not applied here.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_LARGEST = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_LOG_DESCRIPTORS = {"Ipc", "AvgIpc"}


def standardize_mol(smiles: str):
    """Largest fragment, cleanup, and uncharge. Returns None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def murcko_scaffold(mol) -> str:
    """Bemis–Murcko scaffold without chirality.

    Acyclic molecules have an empty Murcko scaffold. They are given a
    per-compound scaffold id so they are not collapsed into one group.
    That choice is documented in the analysis plan: there is no shared ring
    system that could leak across those compounds.
    """
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""
    if scaffold:
        return scaffold
    return f"acyclic:{Chem.MolToSmiles(mol)}"


def featurize_mols(mols: list) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return fingerprints, raw descriptors, descriptor names, and a continuous mask.

    The continuous mask is False for fingerprint bits and True for descriptors.
    """
    names = [name for name, _ in Descriptors.descList]
    n = len(mols)
    fingerprints = np.zeros((n, 2048), dtype=np.float64)
    descriptors = np.full((n, len(names)), np.nan, dtype=np.float64)
    for i, mol in enumerate(mols):
        fingerprints[i] = _MORGAN.GetFingerprintAsNumPy(mol)
        raw = Descriptors.CalcMolDescriptors(mol)
        for j, name in enumerate(names):
            value = raw.get(name, np.nan)
            if value is None:
                value = np.nan
            if name in _LOG_DESCRIPTORS and np.isfinite(value):
                value = np.log1p(max(float(value), 0.0))
            descriptors[i, j] = value
    continuous = np.concatenate(
        [np.zeros(2048, dtype=bool), np.ones(len(names), dtype=bool)]
    )
    return fingerprints, descriptors, names, continuous
