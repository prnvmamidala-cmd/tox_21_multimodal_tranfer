#!/usr/bin/env python3
"""Write endpoint-specific adult scaffold folds without fitting models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxtransfer.config import load_config  # noqa: E402
from toxtransfer.pipeline.ctgov_run import prepare_bundle  # noqa: E402
from toxtransfer.splits.scaffold import assert_split_integrity, group_kfold_indices  # noqa: E402


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/reduced.yaml", ROOT)
    bundle = prepare_bundle(cfg, ROOT)
    index = {ik: i for i, ik in enumerate(bundle["inchikey"].tolist())}
    rows = []
    n_splits = int(cfg["adult_cv"]["n_splits"])
    endpoints = sorted({row["endpoint"] for row in bundle["labels"]})
    for endpoint in endpoints:
        adult = [row for row in bundle["labels"] if row["endpoint"] == endpoint and row["domain"] == "adult"]
        idx = [index[row["inchikey"]] for row in adult]
        groups = bundle["scaffold"][idx]
        ids = bundle["inchikey"][idx]
        folds = group_kfold_indices(groups, n_splits)
        for fold_id, (train, test) in enumerate(folds):
            assert_split_integrity(ids, groups, train, test)
            for local in test:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "inchikey": ids[local],
                        "scaffold": groups[local],
                        "fold": fold_id,
                    }
                )
    dest = ROOT / "results" / "splits"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "adult_scaffold_folds.csv", index=False)
    print(f"wrote {len(rows)} fold rows")


if __name__ == "__main__":
    main()
