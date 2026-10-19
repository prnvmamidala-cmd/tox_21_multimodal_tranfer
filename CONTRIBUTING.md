# Contributing

This repository is a confirmatory methods study. Changes that touch evaluation
must preserve the frozen protocol in `reports/analysis_plan.md`.

- Do not tune hyperparameters, eligibility thresholds, or fusion weights on a
  final transfer test set.
- Do not describe ClinicalTrials.gov adverse-event incidence, FAERS reporting
  fractions, or Tox21 assay hits as clinically validated pediatric toxicity.
- Do not commit raw downloads, Chemical Checker signature matrices, ChEMBL
  bioactivity matrices, API keys, or patient-level records.
- New claims need saved predictions, a paired uncertainty estimate, and a
  pre-specified control (chemistry only, and a permutation control when the
  claim is about biological features).
- Run `pytest` and `python scripts/reproduce_all.py --config configs/smoke.yaml`
  before opening a pull request.
- Read API keys only from environment variables.
