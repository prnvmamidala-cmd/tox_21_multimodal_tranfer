"""Canonical paths for pediatric_transfer_aligned_v3 (FINAL run)."""
from pathlib import Path

RUN_DIR = Path("/workspace/pediatric_transfer_aligned_v3")
RAW = RUN_DIR / "raw"
CACHE = RUN_DIR / "cache"
PROCESSED = RUN_DIR / "processed"
FEATURES = RUN_DIR / "features"
MODELS = RUN_DIR / "models"
PREDICTIONS = RUN_DIR / "predictions"
RESULTS = RUN_DIR / "results"
LOGS = RUN_DIR / "logs"
REPORTS = RUN_DIR / "reports"
CODE = RUN_DIR / "code"

for p in (RAW, CACHE, PROCESSED, FEATURES, MODELS, PREDICTIONS, RESULTS, LOGS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
