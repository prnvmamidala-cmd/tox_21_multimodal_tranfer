#!/usr/bin/env python3
"""Run the pre-registered V5 phase. Refuses to fit without the frozen protocol."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxtransfer.v5.data_build import (  # noqa: E402
    fetch_chembl,
    fetch_smiles,
    featurize,
    load_prediction_index,
    tox21_matrix,
    unique_inchikeys,
)
from toxtransfer.v5.experiment import load_config, run, write_outputs  # noqa: E402


def _download_tox21(root: Path) -> Path:
    dest = root / "data" / "raw" / "tox21" / "tox21.csv.gz"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    urllib.request.urlretrieve(url, dest)
    return dest


def main() -> None:
    prereg = ROOT / "reports" / "v5_preregistration.md"
    if not prereg.exists() or "FROZEN V5 PROTOCOL" not in prereg.read_text():
        sys.exit("Refusing V5 fit: reports/v5_preregistration.md is missing the frozen protocol.")
    cfg = load_config(ROOT / "configs" / "v5.yaml")
    chem_rows = load_prediction_index(ROOT / "results" / "predictions" / "predictions.csv.gz")
    keys = unique_inchikeys(chem_rows)
    cache = ROOT / cfg["outputs"]["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    smiles = fetch_smiles(keys, cache / "smiles.json")
    print(f"smiles {len(smiles)}/{len(keys)}", flush=True)
    built = featurize(keys, smiles)
    kept = built["inchikeys"]
    position = {key: i for i, key in enumerate(keys)}
    order = [position[key] for key in kept]
    bio, bio_meta = fetch_chembl(keys, cache / "chembl_pchembl.npz")
    bio = bio[order]
    tox_path = _download_tox21(ROOT)
    tox, tox_meta = tox21_matrix(tox_path, keys)
    tox = tox[order]
    (cache / "build_meta.json").write_text(
        json.dumps(
            {
                "n_requested": len(keys),
                "n_featurized": len(kept),
                "bio": {k: v for k, v in bio_meta.items() if k != "inchikeys"},
                "tox21": tox_meta,
            },
            indent=2,
        )
        + "\n"
    )
    pack = {
        "inchikeys": kept,
        "scaffolds": built["scaffolds"],
        "chem": built["chem"],
        "fingerprints": built["fingerprints"],
        "continuous": built["continuous"],
        "bio": bio,
        "tox21": tox,
    }
    print(
        f"features {built['chem'].shape} bio missing {np.isnan(bio).mean():.3f} "
        f"tox21 any {np.isfinite(tox).any(axis=1).mean():.3f}",
        flush=True,
    )
    claims = run(cfg, ROOT, pack, chem_rows)
    write_outputs(claims, cfg, ROOT)
    print(f"wrote {cfg['outputs']['report']}", flush=True)


if __name__ == "__main__":
    main()
