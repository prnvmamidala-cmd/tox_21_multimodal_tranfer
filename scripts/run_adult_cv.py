#!/usr/bin/env python3
"""Fit adult scaffold out-of-fold models and the paired transfer models.

Adult OOF and zero-shot transfer share feature matrices and leakage checks,
so this entry point runs both. `run_transfer.py` calls the same command.
"""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "reproduce_all.py"), run_name="__main__")
