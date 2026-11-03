"""Tiny synthetic transfer problem for smoke tests. Not real chemical data."""

from __future__ import annotations

import numpy as np


def make_synthetic(seed: int = 0) -> dict:
    """Adult and pediatric compounds, two continuous endpoints, sparse biology.

    The label depends on the first chemical column. Biology is noise with
    structured missingness. Ten pediatric identifiers also occur in the adult
    set so structure-seen and structure-unseen slices both exist.
    """
    rng = np.random.default_rng(seed)
    n_adult, n_ped_only, n_overlap = 48, 24, 8
    adult_ids = [f"A{i:03d}" for i in range(n_adult)]
    overlap_ids = [f"O{i:03d}" for i in range(n_overlap)]
    ped_ids = [f"P{i:03d}" for i in range(n_ped_only)] + overlap_ids
    # Compound table is the union, adult rows first.
    inchikey = np.array(adult_ids + overlap_ids + ped_ids[:n_ped_only])
    n = len(inchikey)
    scaffolds = np.array([f"scaffold-{i % 12}" for i in range(n)])
    chem = rng.normal(size=(n, 12))
    chem[:, 0] = np.linspace(-2, 2, n) + rng.normal(scale=0.05, size=n)
    n_bio = 4
    bio_values = rng.normal(size=(n, n_bio))
    miss = np.zeros((n, n_bio))
    miss[rng.random((n, n_bio)) < 0.55] = 1.0
    miss[: n // 5] = 1.0  # a block with no biology
    bio_filled = bio_values.copy()
    bio_filled[miss == 1] = 0.0
    bio = np.hstack([bio_filled, miss])
    values_nan = bio_values.copy()
    values_nan[miss == 1] = np.nan
    in_adult = np.array([ik.startswith("A") or ik.startswith("O") for ik in inchikey])
    in_ped = np.array([ik.startswith("P") or ik.startswith("O") for ik in inchikey])
    index = {ik: i for i, ik in enumerate(inchikey)}

    def labels_for(ids: list[str], domain: str, endpoint: str, shift: float) -> list[dict]:
        rows = []
        for ik in ids:
            i = index[ik]
            y = 1.4 * chem[i, 0] + shift + rng.normal(scale=0.3)
            rows.append({"inchikey": ik, "endpoint": endpoint, "domain": domain, "y": y})
        return rows

    label_rows = []
    for endpoint, shift in (("EP1", 0.0), ("EP2", -0.4)):
        label_rows.extend(labels_for(adult_ids + overlap_ids, "adult", endpoint, shift))
        label_rows.extend(labels_for(ped_ids, "pediatric", endpoint, shift + 0.2))
    chem_continuous = np.zeros(chem.shape[1], dtype=bool)
    chem_continuous[6:] = True
    bio_continuous = np.array([True] * n_bio + [False] * n_bio)
    return {
        "inchikey": inchikey,
        "scaffold": scaffolds,
        "in_adult": in_adult,
        "in_pediatric": in_ped,
        "chem": chem,
        "chem_continuous": chem_continuous,
        "bio": bio,
        "bio_values_nan": values_nan,
        "miss": miss,
        "bio_continuous": bio_continuous,
        "n_targets": n_bio,
        "descriptor_names": [f"d{i}" for i in range(6)],
        "labels": label_rows,
        "smiles_conflicts": 0,
        "featurization_failures": 0,
        "logit_max_abs_diff": 0.0,
        "estimate_types": ["synthetic"],
        "source": "synthetic",
    }
