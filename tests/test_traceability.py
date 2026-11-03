"""Reported decimals must match the claim file and the endpoint table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _claims() -> dict:
    path = ROOT / "results" / "metrics" / "primary_claims.json"
    if not path.exists():
        pytest.skip("confirmatory claims are not in this checkout")
    return json.loads(path.read_text())


def test_primary_macros_match_endpoint_table():
    claims = _claims()
    table = pd.read_csv(ROOT / "results" / "statistics" / "endpoint_contrasts.csv")
    eligible = table.loc[table["eligible"].astype(str).isin(["True", "true", "1"])]
    assert len(eligible) == claims["n_endpoints_eligible"]
    assert eligible["spearman_chem"].mean() == pytest.approx(claims["macro_spearman_chem"])
    assert eligible["spearman_early"].mean() == pytest.approx(claims["macro_spearman_early"])
    delta = (eligible["spearman_early"] - eligible["spearman_chem"]).mean()
    assert delta == pytest.approx(claims["delta_early_minus_chem"]["estimate"])


def test_report_and_readme_quote_claim_file():
    claims = _claims()
    report = (ROOT / "reports" / "results.md").read_text()
    readme = (ROOT / "README.md").read_text()
    chem = f"{claims['macro_spearman_chem']:.4f}"
    delta = f"{claims['delta_early_minus_chem']['estimate']:.4f}"
    assert chem in report
    assert chem in readme
    assert delta in report
    assert delta in readme
    supported = "yes" if claims["multimodal_benefit_supported"] else "no"
    assert f"Multimodal benefit supported: {supported}." in report
    assert "not clinical pediatric-toxicity" in report
    tox = json.loads((ROOT / "results" / "metrics" / "tox21_summary.json").read_text())
    assert f"{tox['scaffold_oof']['macro_auroc']:.4f}" in report
    figures = list((ROOT / "results" / "figures").glob("*.pdf"))
    assert len(figures) >= 13
