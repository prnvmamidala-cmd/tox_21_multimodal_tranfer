"""Chemical features: Morgan ECFP4 2048 + RDKit descriptors; adult-only preprocessing."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from ..utils.io_utils import atomic_write_df, atomic_write_json
from ..utils.logging_utils import log
from ..utils.paths import FEATURES, CACHE

DESC_NAMES = [d[0] for d in Descriptors._descList]


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def rdkit_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    out = {}
    for name, fn in Descriptors._descList:
        try:
            val = float(fn(mol))
            if name in ("Ipc", "AvgIpc") and val > 0:
                val = float(np.log(val + 1e-12))
                out[f"log_{name}"] = val
            else:
                out[name] = val
        except Exception:
            out[name] = np.nan
    return out


def build_chem_matrix(structures: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """structures need inchikey, canonical_smiles."""
    rows = []
    for _, r in structures.iterrows():
        smi = r["canonical_smiles"]
        ik = r["inchikey"]
        fp = morgan_fp(smi)
        desc = rdkit_descriptors(smi)
        if fp is None or desc is None:
            continue
        row = {"inchikey": ik}
        for i, v in enumerate(fp):
            row[f"ecfp4_{i}"] = v
        row.update(desc)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("inchikey")
    # drop columns with >5% missing
    miss = df.isna().mean()
    keep = miss[miss <= 0.05].index.tolist()
    df = df[keep]
    atomic_write_df(FEATURES / "chem_raw.csv", df.reset_index())
    log(f"chem raw features shape={df.shape}")
    return df, {"n_bits": 2048, "radius": 2, "desc_kept": [c for c in keep if not c.startswith("ecfp4_")]}


def fit_preprocess_adult(chem_df: pd.DataFrame, adult_inchikeys: List[str]) -> dict:
    """Median impute + drop constants using ADULT training only; clip."""
    adult = chem_df.loc[chem_df.index.isin(adult_inchikeys)]
    medians = adult.median(numeric_only=True)
    filled = adult.fillna(medians)
    nunique = filled.nunique()
    keep_cols = nunique[nunique > 1].index.tolist()
    filled = filled[keep_cols]
    q_lo = filled.quantile(0.001)
    q_hi = filled.quantile(0.999)
    params = {
        "medians": medians[keep_cols].to_dict(),
        "keep_cols": keep_cols,
        "clip_lo": q_lo.to_dict(),
        "clip_hi": q_hi.to_dict(),
    }
    atomic_write_json(FEATURES / "chem_preprocess_params.json", params)
    log(f"chem preprocess keep_cols={len(keep_cols)} adult_n={len(adult)}")
    return params


def transform_chem(chem_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    cols = params["keep_cols"]
    medians = pd.Series(params["medians"])
    X = chem_df.reindex(columns=cols)
    X = X.fillna(medians)
    lo = pd.Series(params["clip_lo"])
    hi = pd.Series(params["clip_hi"])
    X = X.clip(lower=lo, upper=hi, axis=1)
    return X
