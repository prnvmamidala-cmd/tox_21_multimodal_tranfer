#!/usr/bin/env python3
"""Record whether Chemical Checker signatures can be generated here.

This probe does not install TensorFlow and does not download signature models
unless `signaturizer` already imports. A failure is a blocker, not a result.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report = {"status": "blocked", "reason": "", "attempted_import": True}
    try:
        import signaturizer  # type: ignore  # noqa: F401
    except Exception as exc:
        report["reason"] = (
            "signaturizer is not importable in this environment "
            f"({type(exc).__name__}: {exc}). Chemical Checker signature matrices "
            "were not regenerated. Notebook 2's prior Tox21 multimodal numbers "
            "remain archived exploratory results and are not re-executed."
        )
    else:
        report["status"] = "importable"
        report["reason"] = (
            "The package imported. This probe does not run a full 7,587-compound "
            "signature job. Treat any later signature file as unverified until "
            "its source, version, and assay overlap are recorded."
        )
    dest = ROOT / "reports" / "signaturizer_probe.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
