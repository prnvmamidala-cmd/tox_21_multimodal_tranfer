# Reproduction

## Confirmatory command that was run

```bash
python scripts/reproduce_all.py --config configs/reduced.yaml
```

The run that produced `results/predictions/predictions.csv.gz` started at
2026-10-19T15:19:19Z and wrote `results/run_manifest.json` at
2026-10-19T15:29:45Z (626 seconds). `OMP_NUM_THREADS` was 4. The feature cache
already existed, so that wall clock does not include descriptor calculation
(about 100 seconds when the cache was built).

Fit commit recorded in the manifest: `a9324cf9bcffd5d406cb21b5e70c16f60382d017`.

Prediction sha256:

- `results/predictions/predictions.csv.gz`
  `37dbf42edac39337d9e76aeb446661dd7e3a0e16429d453cfd4d93abf68faa0d`
- `results/predictions/tox21_predictions.csv.gz`
  `6f4b1107c0ccd1810d253869e92f4b5e13ca07f8cfd45ad46dfed4c35dd35736`

Tables, figures, and `reports/results.md` are functions of the trial prediction
file. Refresh them without refitting:

```bash
python scripts/run_diagnostics.py --config configs/reduced.yaml
```

## Smoke command

```bash
pytest
python scripts/reproduce_all.py --config configs/smoke.yaml
```

The smoke config uses an in-memory cohort. It does not read the handoff files
and it does not update `reports/results.md`.

## Full command, not run

```bash
python scripts/reproduce_all.py --config configs/full.yaml
```

That file turns on inner scaffold selection of the ridge alpha, a larger
bootstrap, and larger trees with adult out-of-fold predictions. It is not the
source of the numbers in `reports/results.md`.

## What has to be on disk

Raw tables are gitignored. `data/README.md` describes the handoff layout.
`scripts/download_or_prepare_data.py` copies a local handoff into
`data/raw/ctgov_v3/` and can fetch the Tox21 CSV. Without those files the
confirmatory command cannot rebuild features. The committed predictions are
enough to rebuild the tables and figures.

## Hardware and packages

The fit manifest records Linux x86_64, Python 3.12.3, scikit-learn 1.9.1,
RDKit 2026.9.1, XGBoost 3.4.1, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1,
matplotlib 3.11.2, PyYAML 6.0.1, and joblib 1.6.0. Versions are pinned in
`requirements.txt`.
