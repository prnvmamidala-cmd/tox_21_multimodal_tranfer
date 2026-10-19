#!/usr/bin/env python3
"""Recompute ablation tables from saved predictions. Does not refit models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxtransfer.analysis.summarize import summarize  # noqa: E402
from toxtransfer.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/reduced.yaml", ROOT)
    pred = ROOT / "results" / "predictions" / "predictions.csv.gz"
    summarize(pred, cfg, ROOT / "results", report_path=ROOT / "reports" / "results.md")
    print("ablation tables refreshed from saved predictions")


if __name__ == "__main__":
    main()
