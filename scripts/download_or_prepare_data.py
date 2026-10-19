#!/usr/bin/env python3
"""Download Tox21 and copy a local ClinicalTrials.gov handoff into data/raw.

Raw files stay gitignored. No API key is read or written.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxtransfer.config import load_config  # noqa: E402

HANDOFF_FILES = [
    "ctgov_adult_structure_outcome.csv",
    "ctgov_pediatric_structure_outcome.csv",
    "drug_name_mapping.csv",
    "chembl_target_profile.csv",
    "chembl_target_vocab.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reduced.yaml")
    parser.add_argument("--handoff", default="", help="Directory containing the CT.gov handoff CSVs.")
    parser.add_argument("--skip-tox21", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, ROOT)
    if args.handoff:
        source = Path(args.handoff)
        dest = ROOT / cfg["data"]["ctgov_dir"]
        dest.mkdir(parents=True, exist_ok=True)
        for name in HANDOFF_FILES:
            src = source / name
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dest / name)
            print(f"copied {name}")
    if not args.skip_tox21:
        dest = ROOT / cfg["data"]["tox21_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"tox21 already present at {dest.relative_to(ROOT)}")
        else:
            url = cfg["data"]["tox21_url"]
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"wrote {dest.relative_to(ROOT)} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
