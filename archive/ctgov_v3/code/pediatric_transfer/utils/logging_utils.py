"""Run logging + failure log."""
from __future__ import annotations
import csv
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .paths import LOGS

PT = ZoneInfo("America/Los_Angeles")
RUN_LOG = LOGS / "run.log"
FAIL_LOG = LOGS / "failure_log.csv"


def _now() -> str:
    return datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S %Z")


def log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Avoid double-write when stdout is already redirected to RUN_LOG
    try:
        import os
        if os.path.exists("/proc/self/fd/1") and Path("/proc/self/fd/1").resolve() == RUN_LOG.resolve():
            return
    except Exception:
        pass
    with open(RUN_LOG, "a") as f:
        f.write(line + "\n")


def log_failure(stage: str, item: str, error: str, detail: str = "") -> None:
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not FAIL_LOG.exists()
    with open(FAIL_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "stage", "item", "error", "detail"])
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": _now(),
            "stage": stage,
            "item": item,
            "error": error[:2000],
            "detail": (detail or "")[:4000],
        })
    log(f"FAILURE [{stage}] {item}: {error}")


def log_exc(stage: str, item: str, exc: BaseException) -> None:
    log_failure(stage, item, type(exc).__name__ + ": " + str(exc), traceback.format_exc())
