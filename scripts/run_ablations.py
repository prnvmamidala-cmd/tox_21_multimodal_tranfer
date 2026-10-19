#!/usr/bin/env python3
"""Recompute ablation tables from saved predictions. Does not refit models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxtransfer.analysis.summarize import summarize  # noqa: E402
from toxtransfer.config import load_config  # noqa: E402


def _config_arg(argv: list[str]) -> str:
    if "--config" in argv:
        index = argv.index("--config")
        if index + 1 >= len(argv):
            raise SystemExit("--config requires a path")
        return argv[index + 1]
    positional = [arg for arg in argv[1:] if not arg.startswith("-")]
    return positional[0] if positional else "configs/reduced.yaml"


def main() -> None:
    cfg = load_config(_config_arg(sys.argv), ROOT)
    pred = ROOT / "results" / "predictions" / "predictions.csv.gz"
    summarize(pred, cfg, ROOT / "results", report_path=ROOT / "reports" / "results.md")
    print("ablation tables refreshed from saved predictions")


if __name__ == "__main__":
    main()
