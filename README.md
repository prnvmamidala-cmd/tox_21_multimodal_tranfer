# Chemical and biological representations under domain shift

Tox21 assay hits, ClinicalTrials.gov adverse-event counts, and FAERS reports
are not the same endpoint. This repository asks a narrower question: on
structures that appear in pediatric-only trials and not in adult-only trials,
does a sparse ChEMBL target profile improve prediction of smoothed
adverse-event incidence beyond chemistry, once row-permuted biology and a
coverage gate are on the table?

The confirmatory answer is whatever `reports/results.md` says after
`scripts/reproduce_all.py`. That file is generated from saved predictions.
Do not quote a performance number that is not in it.

## Questions

1. Under Bemis–Murcko scaffold cross-validation in the adult-trial domain,
   how does chemistry compare with the target profile and with early fusion?
2. On structure-unseen pediatric-trial rows, does early fusion beat chemistry
   by more than the permutation controls?
3. Does mixing the two predictions with a weight equal to biological coverage
   reduce negative transfer relative to always using early fusion?
4. Separately, and only as a chemical baseline: how much does a random split
   overstate Tox21 assay AUROC relative to a scaffold split, and do nuclear
   receptor and stress-response assays differ under that scaffold split?

Question 4 does not use a biological representation unless a Chemical Checker
run is added later under the license note in `data/dataset_manifest.csv`.

## Preliminary results (reduced confirmatory run)

These sentences are copied from `reports/results.md`, which is generated from
`results/predictions/predictions.csv.gz`. They are not a clinical result.

On 12 eligible endpoints, structure-unseen pediatric-trial rows, ridge
`alpha=10`: macro Spearman chemistry 0.0314, early fusion 0.0339, coverage
gate 0.0308. Early fusion minus chemistry: delta 0.0025 (95% CI -0.0106 to
0.0149; the interval includes 0). Early fusion minus the mean of the
bio-permutation seeds: delta 0.0031 (95% CI -0.0143 to 0.0213; the interval
includes 0). Coverage gate minus chemistry: delta -0.0007 (95% CI -0.0029 to
0.0014; the interval includes 0). Multimodal benefit supported: no. The
coverage gate does not reduce negative transfer (7 endpoints versus 5 for
early fusion).

The same chemistry model on structure-seen rows has macro Spearman 0.1923.
That is same-drug continuity. The primary slice removes those rows.

The Tox21 chemical arm is logistic regression on Morgan fingerprints, not the
archived XGBoost figure. Scaffold-split macro AUROC 0.6895 versus random-split
0.7331 (gap 0.0436) on 7,587 compounds. Chemical Checker signatures were not
regenerated.

## What is not claimed

- Clinical readiness, pediatric dosing, or developmental toxicity.
- That FAERS "serious" is a Tox21 endpoint. The historical 137- and
  46-compound probes are documented as invalid comparisons.
- That a prior experiment's Spearman or AUROC can be pasted into this one.
  Verdicts live in `archive/prior_verdicts/` and are cited in
  `reports/initial_audit.md`.

## Layout

```
archive/                  sanitized notebooks, prior verdicts, prior code
configs/                  data, models, splits, reduced/smoke/full overlays
data/                     manifest and README; raw files are gitignored
src/toxtransfer/          loaders, features, splits, models, metrics, figures
scripts/                  download, validate, fit, summarize
tests/                    leakage, metrics, smoke pipeline
results/                  predictions, metrics, figures (after a run)
reports/                  audit, frozen plan, methods, results, limitations
paper/                    workshop outline and tables
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/reproduce_all.py --config configs/smoke.yaml
```

The smoke config uses a synthetic cohort in memory. It does not need the
handoff files and it does not write the confirmatory claim.

## Trial data

Raw tables are not in git. See `data/README.md` and `data/dataset_manifest.csv`.
After placing the handoff CSVs:

```bash
python scripts/download_or_prepare_data.py --handoff /path/to/ctgov_v3
python scripts/reproduce_all.py --config configs/reduced.yaml
```

The reduced command is the confirmatory run: ridge `alpha=10`, five
permutation seeds, 1000 endpoint-bootstrap draws, random forest and XGBoost
at the sizes in `configs/models.yaml`, plus the Tox21 chemical scaffold arm
if `tox21.csv.gz` has been downloaded.

Expected cost on a 4-core machine: on the order of 15 minutes for the trial
models after features are cached, plus Tox21 logistic regression. Feature
construction itself is about two minutes for the 3,057 trial structures.

The full nested-alpha command is specified and was not the confirmatory run:

```bash
python scripts/reproduce_all.py --config configs/full.yaml
```

Fit provenance, file hashes, and the command that was actually run are in
`reports/reproduction.md`.

## Reproducing figures

Figures and tables are functions of `results/predictions/predictions.csv.gz`.

```bash
python scripts/run_diagnostics.py --config configs/reduced.yaml
```

## Licenses

Code in this repository is MIT. Data licenses differ and are in
`data/dataset_manifest.csv`. ChEMBL is CC BY-SA 3.0; its derived matrix is
not committed. openFDA content used historically is CC0. Do not add API keys.

## Citation

See `CITATION.cff`. Please cite Wu et al., Chemical Science, 2018
(DOI 10.1039/C7SC02664A) if you use the Tox21 split of this code, and the
Chemical Checker paper if you extend the signature arm.

## Responsible use

Predictions are research diagnostics on trial reporting constructs and
in-vitro assays. They are not a substitute for clinical, regulatory, or
preclinical safety assessment.
