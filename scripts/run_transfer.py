#!/usr/bin/env python3
"""Zero-shot adult-trial to pediatric-trial transfer, with adult OOF.

See run_adult_cv.py. The phases are one command so the transfer test cannot
be fit with a different preprocessor than the adult folds.
"""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "reproduce_all.py"), run_name="__main__")
