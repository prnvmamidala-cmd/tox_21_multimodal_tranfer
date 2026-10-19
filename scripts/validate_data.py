#!/usr/bin/env python3
"""Check the ClinicalTrials.gov handoff tables without fitting a model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxtransfer.config import load_config  # noqa: E402
from toxtransfer.data.ctgov import load_ctgov_tables  # noqa: E402


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/reduced.yaml", ROOT)
    tables = load_ctgov_tables(ROOT / cfg["data"]["ctgov_dir"])
    labels = tables["labels"]
    summary = {
        "n_label_rows": int(len(labels)),
        "n_adult_structures": len(tables["adult_ids"]),
        "n_pediatric_structures": len(tables["pediatric_ids"]),
        "n_overlap": len(tables["adult_ids"] & tables["pediatric_ids"]),
        "n_bio_targets": len(tables["tgt_cols"]),
        "smiles_conflicts": tables["smiles_conflicts"],
        "logit_max_abs_diff_vs_recompute": tables["logit_max_abs_diff"],
        "estimate_types": tables["estimate_types"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
