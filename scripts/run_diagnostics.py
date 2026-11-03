#!/usr/bin/env python3
"""Regenerate metrics, figures, and the results report from saved predictions."""

from __future__ import annotations

import runpy
from pathlib import Path

# Same reader as the ablation entry point: figures are a function of predictions.
runpy.run_path(str(Path(__file__).resolve().parent / "run_ablations.py"), run_name="__main__")
