# Archive

These files are the sanitized inputs this study audited. They are not the
confirmatory implementation.

- `notebooks/` — Notebook 1 and Notebook 2 with the hard-coded openFDA key
  removed. Notebook 2 still shows a Colab secret read (`userdata.get`), which
  is empty unless a secret is configured. Do not put a key back.
- `prior_verdicts/` — JSON verdicts from earlier experiments. Their metrics
  use different targets (trial incidence, FAERS reporting ratios, conformal
  intervals). Do not pool them with Tox21 AUROC.
- `ctgov_v3/` — methods, limitations, the prior transfer metric table, and
  the prior Python package. The large outcome, mapping, chemistry, and
  ChEMBL matrices are not in git. The code reads `OPENFDA_API_KEY` from the
  environment or from a local file path; no key value is stored here.
- `prior_experiments_summary_build_full.py` — the narrative summary of those
  experiments. It is a document generator, not a modeling entry point.

`archive/ctgov_v3/transfer_metrics.csv` is a historical result table. It is
not reproduced by `scripts/reproduce_all.py` and is not a source of numbers
for `reports/results.md`.
