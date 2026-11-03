#!/usr/bin/env python3
"""Recompute chemistry and cache it under data/interim (gitignored)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxtransfer.config import load_config  # noqa: E402
from toxtransfer.pipeline.ctgov_run import prepare_bundle  # noqa: E402


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/reduced.yaml", ROOT)
    bundle = prepare_bundle(cfg, ROOT)
    print(
        f"compounds={len(bundle['inchikey'])} chem={bundle['chem'].shape} "
        f"bio={bundle['bio'].shape} failures={bundle['featurization_failures']}"
    )


if __name__ == "__main__":
    main()
