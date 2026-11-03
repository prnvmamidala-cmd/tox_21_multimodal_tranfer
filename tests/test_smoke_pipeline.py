"""End-to-end smoke test on synthetic data. It does not claim a scientific result."""

from __future__ import annotations

import json
from pathlib import Path

from toxtransfer.analysis.summarize import summarize
from toxtransfer.config import load_config
from toxtransfer.pipeline.ctgov_run import prepare_bundle, run_bundle


def test_smoke_pipeline_writes_traceable_outputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config("configs/smoke.yaml", root)
    bundle = prepare_bundle(cfg, root)
    pred_path = run_bundle(bundle, cfg, tmp_path)
    claims = summarize(pred_path, cfg, tmp_path, report_path=tmp_path / "results.md")
    frame_path = tmp_path / "predictions" / "predictions.csv.gz"
    assert frame_path.exists()
    leakage = json.loads((tmp_path / "leakage_checks.json").read_text())
    assert leakage["permutation"]["preserves_multiset"]
    assert leakage["unseen"]["n_overlap"] == 0
    assert "multimodal_benefit_supported" in claims
    text = (tmp_path / "results.md").read_text()
    assert "not clinical pediatric-toxicity" in text
    # Coverage 0 forces the gate to match chemistry exactly.
    import pandas as pd

    fused = pd.read_csv(tmp_path / "predictions" / "predictions_with_fusion.csv.gz")
    chem = fused.loc[
        (fused.phase == "transfer") & (fused.model == "ridge") & (fused.representation == "chem")
    ]
    gate = fused.loc[
        (fused.phase == "transfer")
        & (fused.model == "postprocess")
        & (fused.representation == "gate_coverage")
    ]
    merged = chem.merge(gate, on=["endpoint", "inchikey"], suffixes=("_chem", "_gate"))
    zero = merged.loc[merged["bio_coverage_chem"] <= 0]
    assert len(zero) > 0
    assert (zero["y_pred_chem"] - zero["y_pred_gate"]).abs().max() < 1e-8
