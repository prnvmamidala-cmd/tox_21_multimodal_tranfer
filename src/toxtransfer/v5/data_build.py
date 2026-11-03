"""Rebuild public chemistry and biology features for V5.

Labels and the structure-unseen flag stay in the committed prediction file.
SMILES come from PubChem. Target values come from the ChEMBL API. Tox21 assay
outcomes come from the DeepChem file. None of these matrices are written into
git; the cache lives under data/interim/v5.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from toxtransfer.features.chemistry import featurize_mols, murcko_scaffold, standardize_mol

TARGETS = [
    "CHEMBL612545",
    "CHEMBL3879801",
    "CHEMBL240",
    "CHEMBL1293235",
    "CHEMBL340",
    "CHEMBL229",
    "CHEMBL364",
    "CHEMBL238",
    "CHEMBL228",
    "CHEMBL1833",
    "CHEMBL222",
    "CHEMBL224",
    "CHEMBL234",
    "CHEMBL233",
    "CHEMBL256",
]
TOX21_TASKS = [
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


def _get_json(url: str, timeout: int = 60, attempts: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url[:120]}: {last}")


def load_prediction_index(predictions: Path) -> pd.DataFrame:
    """One row per structure, plus the long label table for ridge chemistry."""
    frame = pd.read_csv(predictions)
    chem = frame.loc[(frame["model"] == "ridge") & (frame["representation"] == "chem")].copy()
    return chem


def unique_inchikeys(chem: pd.DataFrame) -> list[str]:
    return sorted(set(chem["inchikey"].astype(str)))


def fetch_smiles(inchikeys: list[str], cache: Path, batch: int = 80) -> dict[str, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    found: dict[str, str] = {}
    if cache.exists():
        found = json.loads(cache.read_text())
    missing = [key for key in inchikeys if key not in found]
    for start in range(0, len(missing), batch):
        chunk = missing[start : start + batch]
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
            + ",".join(chunk)
            + "/property/InChIKey,ConnectivitySMILES/JSON"
        )
        try:
            payload = _get_json(url)
        except Exception as exc:  # noqa: BLE001
            print(f"pubchem batch failed ({exc}); trying keys one at a time", flush=True)
            for key in chunk:
                one = (
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
                    f"{key}/property/InChIKey,ConnectivitySMILES/JSON"
                )
                try:
                    payload = _get_json(one, attempts=1)
                except Exception:
                    found[key] = ""
                    continue
                hit = False
                for row in payload.get("PropertyTable", {}).get("Properties", []):
                    if row.get("InChIKey") == key and row.get("ConnectivitySMILES"):
                        found[key] = row["ConnectivitySMILES"]
                        hit = True
                        break
                if not hit:
                    found[key] = ""
            cache.write_text(json.dumps(found))
            continue
        for row in payload.get("PropertyTable", {}).get("Properties", []):
            found.setdefault(row["InChIKey"], row["ConnectivitySMILES"])
        cache.write_text(json.dumps(found))
        print(f"pubchem {min(start + batch, len(missing))}/{len(missing)}", flush=True)
    return {key: found[key] for key in inchikeys if key in found}


def fetch_chembl(inchikeys: list[str], cache: Path, batch: int = 12) -> tuple[np.ndarray, dict]:
    """Median pChEMBL for the frozen 15-target vocabulary. Cache is not committed."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache.with_suffix(".json")
    keys_path = cache.parent / "chembl_inchikeys.json"
    if cache.exists() and meta_path.exists() and keys_path.exists():
        stored = json.loads(keys_path.read_text())
        matrix = np.load(cache)["pchembl"]
        if stored == inchikeys and matrix.shape == (len(inchikeys), len(TARGETS)):
            meta = json.loads(meta_path.read_text())
            meta["inchikeys"] = stored
            return matrix, meta
    id_map: dict[str, str] = {}
    id_cache = cache.parent / "chembl_ids.json"
    if id_cache.exists():
        id_map = json.loads(id_cache.read_text())
    need = [key for key in inchikeys if key not in id_map]
    for start in range(0, len(need), batch):
        chunk = need[start : start + batch]
        url = (
            "https://www.ebi.ac.uk/chembl/api/data/molecule.json?"
            "molecule_structures__standard_inchi_key__in="
            + ",".join(chunk)
            + "&limit=50"
        )
        try:
            payload = _get_json(url)
        except Exception as exc:  # noqa: BLE001
            print(f"chembl molecule batch failed: {exc}", flush=True)
            for key in chunk:
                id_map.setdefault(key, "")
            continue
        returned = set()
        for molecule in payload.get("molecules", []):
            structures = molecule.get("molecule_structures") or {}
            key = structures.get("standard_inchi_key")
            chembl_id = molecule.get("molecule_chembl_id") or ""
            if key and key not in id_map and chembl_id:
                id_map[key] = chembl_id
                returned.add(key)
        for key in chunk:
            id_map.setdefault(key, "")
        id_cache.write_text(json.dumps(id_map))
        print(f"chembl ids {min(start + batch, len(need))}/{len(need)}", flush=True)
    values: dict[tuple[str, str], list[float]] = {}
    ids = [id_map[key] for key in inchikeys if id_map.get(key)]
    id_to_keys: dict[str, list[str]] = {}
    for key, chembl_id in id_map.items():
        if chembl_id:
            id_to_keys.setdefault(chembl_id, []).append(key)
    targets = ",".join(TARGETS)
    for start in range(0, len(ids), 20):
        chunk = ids[start : start + 20]
        url = (
            "https://www.ebi.ac.uk/chembl/api/data/activity.json?"
            "molecule_chembl_id__in="
            + ",".join(chunk)
            + "&target_chembl_id__in="
            + targets
            + "&pchembl_value__isnull=false&limit=1000"
        )
        try:
            payload = _get_json(url, timeout=90)
        except Exception as exc:  # noqa: BLE001
            print(f"chembl activity batch failed: {exc}", flush=True)
            continue
        for row in payload.get("activities", []):
            chembl_id = row.get("molecule_chembl_id")
            target = row.get("target_chembl_id")
            raw = row.get("pchembl_value")
            if chembl_id not in id_to_keys or target not in TARGETS or raw in (None, ""):
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            for key in id_to_keys[chembl_id]:
                values.setdefault((key, target), []).append(number)
        print(f"chembl activities {min(start + 20, len(ids))}/{len(ids)}", flush=True)
    matrix = np.full((len(inchikeys), len(TARGETS)), np.nan, dtype=np.float64)
    index = {key: i for i, key in enumerate(inchikeys)}
    target_index = {target: j for j, target in enumerate(TARGETS)}
    for (key, target), samples in values.items():
        if key in index:
            matrix[index[key], target_index[target]] = float(np.median(samples))
    meta = {
        "inchikeys": inchikeys,
        "targets": TARGETS,
        "n_with_chembl_id": int(sum(bool(id_map.get(key)) for key in inchikeys)),
        "n_with_any_value": int(np.isfinite(matrix).any(axis=1).sum()),
        "mean_missingness": float(np.isnan(matrix).mean()),
        "source": "ChEMBL API median pChEMBL for the frozen 15-target vocabulary",
    }
    np.savez_compressed(cache, pchembl=matrix)
    meta_path.write_text(json.dumps({k: v for k, v in meta.items() if k != "inchikeys"}))
    # inchikeys are long; store them beside the matrix for the cache check.
    (cache.parent / "chembl_inchikeys.json").write_text(json.dumps(inchikeys))
    meta["inchikeys"] = inchikeys
    return matrix, meta


def load_chembl_cache(inchikeys: list[str], cache: Path) -> tuple[np.ndarray, dict] | None:
    meta_path = cache.with_suffix(".json")
    keys_path = cache.parent / "chembl_inchikeys.json"
    if not (cache.exists() and meta_path.exists() and keys_path.exists()):
        return None
    stored = json.loads(keys_path.read_text())
    if stored != inchikeys:
        return None
    matrix = np.load(cache)["pchembl"]
    meta = json.loads(meta_path.read_text())
    meta["inchikeys"] = stored
    return matrix, meta


def tox21_matrix(tox21_path: Path, inchikeys: list[str]) -> tuple[np.ndarray, dict]:
    """Assay outcomes aligned by the same standardization used for trial SMILES.

    InChIKeys are computed locally from Tox21 SMILES. They are not taken from a
    prejoined file.
    """
    table = pd.read_csv(tox21_path)
    smiles_col = "smiles" if "smiles" in table.columns else "SMILES"
    grouped: dict[str, list[np.ndarray]] = {}
    task_cols = [column for column in TOX21_TASKS if column in table.columns]
    for _, row in table.iterrows():
        mol = standardize_mol(str(row[smiles_col]))
        if mol is None:
            continue
        key = Chem.MolToInchiKey(mol)
        vector = np.array([row[task] if pd.notna(row[task]) else np.nan for task in task_cols], dtype=np.float64)
        grouped.setdefault(key, []).append(vector)
    means = {key: np.nanmean(np.vstack(rows), axis=0) for key, rows in grouped.items()}
    matrix = np.full((len(inchikeys), len(task_cols)), np.nan, dtype=np.float64)
    for i, key in enumerate(inchikeys):
        if key in means:
            matrix[i] = means[key]
    observed = np.isfinite(matrix).any(axis=1)
    meta = {
        "n_tasks": len(task_cols),
        "n_structures_with_any_assay": int(observed.sum()),
        "tasks": task_cols,
        "source": str(tox21_path),
    }
    return matrix, meta


def featurize(inchikeys: list[str], smiles: dict[str, str]) -> dict:
    keep = [key for key in inchikeys if key in smiles]
    mols = []
    kept = []
    scaffolds = []
    for key in keep:
        mol = standardize_mol(smiles[key])
        if mol is None:
            continue
        mols.append(mol)
        kept.append(key)
        scaffolds.append(murcko_scaffold(mol))
    fingerprints, descriptors, names, continuous = featurize_mols(mols)
    chem = np.concatenate([fingerprints, descriptors], axis=1)
    return {
        "inchikeys": kept,
        "smiles": [smiles[key] for key in kept],
        "scaffolds": scaffolds,
        "chem": chem,
        "fingerprints": fingerprints,
        "descriptor_names": names,
        "continuous": continuous,
    }
