#!/usr/bin/env python3
"""Standalone summary of pediatric_transfer_aligned_v2 results."""
import pandas as pd, json
from pathlib import Path
R = Path("/workspace/pediatric_transfer_aligned_v2/results")
print(pd.read_csv(R/"transfer_metrics.csv").query("model=='xgb'").pivot_table(index="outcome", columns="representation", values="spearman"))
print(pd.read_csv(R/"bootstrap_comparisons.csv").query("contrast=='chem_true_bio_minus_chem'")[["outcome","delta","ci_lo","ci_hi","p_value","p_bh"]])
print(json.load(open(R/"hypothesis_verdict.json")))
