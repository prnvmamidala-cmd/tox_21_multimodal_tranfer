#!/usr/bin/env python3
"""Reproduce the smoke test or the reduced confirmatory run.

Confirmatory runs refuse to start unless reports/analysis_plan.md contains the
frozen-protocol marker. That file is committed before the confirmatory fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxtransfer.analysis.summarize import summarize  # noqa: E402
from toxtransfer.config import load_config  # noqa: E402
from toxtransfer.pipeline.ctgov_run import prepare_bundle, run_bundle  # noqa: E402
from toxtransfer.pipeline.tox21_run import run_tox21  # noqa: E402
from toxtransfer.provenance import (  # noqa: E402
    git_commit,
    hardware_summary,
    json_sha256,
    package_versions,
    utc_now,
    write_manifest,
)


def _output_dir(cfg: dict, root: Path) -> Path:
    if cfg.get("name") == "smoke" or cfg["data"].get("dataset") == "synthetic":
        return root / "results" / "smoke"
    return root / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reduced.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, ROOT)
    confirmatory = cfg["data"].get("dataset") != "synthetic"
    plan = ROOT / "reports" / "analysis_plan.md"
    if confirmatory:
        if not plan.exists() or "FROZEN CONFIRMATION PROTOCOL" not in plan.read_text():
            sys.exit(
                "Refusing confirmatory run: reports/analysis_plan.md is missing the frozen protocol."
            )
    output = _output_dir(cfg, ROOT)
    bundle = prepare_bundle(cfg, ROOT)
    pred_path = run_bundle(bundle, cfg, output)
    report = None if not confirmatory else ROOT / "reports" / "results.md"
    summarize(pred_path, cfg, output, report_path=report)
    if confirmatory and cfg["transfer"].get("run_tox21", False):
        tox_path = ROOT / cfg["data"]["tox21_path"]
        if tox_path.exists():
            run_tox21(cfg, ROOT, output)
        else:
            print(f"Tox21 file not found at {tox_path}; chemical scaffold arm skipped.")
    manifest = {
        "timestamp_utc": utc_now(),
        "git_commit": git_commit(ROOT),
        "config_sha256": json_sha256(cfg),
        "config_name": cfg.get("name"),
        "packages": package_versions(),
        "hardware": hardware_summary(),
        "predictions": str(pred_path.relative_to(ROOT)),
        "confirmatory": confirmatory,
    }
    write_manifest(output / "run_manifest.json", manifest)
    print(f"wrote {pred_path}")


if __name__ == "__main__":
    main()
