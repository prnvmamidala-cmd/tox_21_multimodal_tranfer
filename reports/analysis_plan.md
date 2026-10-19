# Analysis plan

FROZEN CONFIRMATION PROTOCOL

This file is the pre-specified plan for the confirmatory run. It is committed
before `scripts/reproduce_all.py --config configs/reduced.yaml` fits models.
Exploratory analyses are labeled below and cannot replace the primary tests.

## Question

When a biological target profile is added to chemistry, does prediction of
smoothed adverse-event incidence improve when the test structures were not in
the adult-trial training set, and does a coverage gate reduce negative
transfer when that profile is missing?

## Title of the estimand

Adult-trial to pediatric-trial adverse-event incidence transfer.

This is not a study of clinically validated pediatric toxicity. Tox21 labels
are not the transfer labels. FAERS "serious" is not an endpoint in this plan.

## Primary data

- Source domain: ClinicalTrials.gov interventional trials with a parseable
  minimum age of at least 18 years, single identifiable drug arm, results
  posted. One row per structure and outcome.
- Target domain: the same construction with a parseable maximum age of at
  most 17 years. Mixed-age trials are already excluded in the handoff.
- Label: smoothed incidence `(affected + 0.5) / (at_risk + 1)`, then logit.
  The loader recomputes this from the counts and checks it against the stored
  column. Rows whose `estimate_type` is not
  `clinical_trial_smoothed_incidence` are rejected.
- Chemical features: Morgan fingerprint, radius 2, 2048 bits, plus the RDKit
  `Descriptors.descList` set. `Ipc` and `AvgIpc` receive a parameter-free
  `log1p` of the non-negative value before any split. Imputation, clipping,
  variance filtering, and scaling are fit on the training rows of the current
  split only.
- Biological features: the frozen 15-target ChEMBL profile shipped with the
  handoff (median pChEMBL, zero-filled, plus a missingness indicator per
  target). The vocabulary is not refit. This is not a Chemical Checker
  signature.
- Acyclic molecules have an empty Bemis–Murcko scaffold. They receive a
  per-compound scaffold id (`acyclic:` plus the SMILES). They do not share a
  ring system, so they are not placed in one giant group.

## Primary population

Pediatric-trial structures whose InChIKey does not appear in any adult-trial
row. Shared InChIKeys are a same-drug continuity check, reported separately,
and are not the primary test.

A second reported slice, not primary, keeps only structure-unseen compounds
whose scaffold is also absent from the adult set.

## Eligibility

An endpoint is in the confirmatory macro mean if:

- at least 200 adult-trial structures have that label, and
- at least 40 structure-unseen pediatric-trial structures have that label.

Thresholds 20 and 60 for the unseen count are sensitivity analyses. They are
not used to choose the claim. Endpoints that fail the primary cutoff are
listed and are not dropped from the prediction file.

## Primary model

Ridge regression with pre-specified `alpha = 10` after train-only scaling.
`configs/reduced.yaml` does not search this alpha. `configs/full.yaml` selects
alpha by inner scaffold GroupKFold on training rows only. That command is not
part of the confirmatory run.

Random forest and XGBoost use the pre-specified sizes in `configs/models.yaml`.
They are robustness checks. They do not choose the claim. Their adult
out-of-fold loop is off in the reduced run.

## Representations fit in the reduced run

- chemistry
- biology only (values and missingness indicators)
- early fusion of chemistry and biology
- missingness indicators only
- early fusion with target values median-imputed inside the fold and no
  missingness indicators
- chemistry plus row-permuted biology, seeds 101, 202, 303, 404, 505
- chemistry plus dimension-matched standard normal noise, seed 101
- chemistry with adult labels permuted, seed 101
- CORAL on a source-fit PCA of chemistry, using unlabeled pediatric chemistry
- importance-weighted chemistry, weights from a source-versus-target feature
  classifier, unlabeled target features only
- a target-only ridge reference fit on half of the structure-unseen pediatric
  compounds, grouped by scaffold, evaluated on the other half

CORAL, importance weighting, and the target-only reference are not zero-shot
inductive methods. CORAL and importance weighting are transductive: they see
unlabeled target features. The target-only reference sees target labels. None
of the three is the primary test.

Permutation of biology is applied to the training rows of the current fit.
Test rows keep their own biological vectors. The control asks whether a model
trained on broken compound–biology correspondence still helps. It does not put
test biology into training.

## Fusion, decided without target labels

Adult scaffold out-of-fold predictions choose weights. Pediatric labels do not.

- Fixed 0.5 chemistry + 0.5 early fusion.
- Fixed 0.5 chemistry + 0.5 biology-only.
- One global weight on the grid `{0, 0.25, 0.5, 0.75, 1}` maximizing mean
  adult out-of-fold Spearman of chemistry/early mixtures.
- An endpoint-specific weight on the same grid, fit on that endpoint's adult
  out-of-fold predictions only.
- Coverage gate: `g = fraction of the 15 targets observed`,
  `prediction = (1 - g) * chemistry + g * early`.
- Binary coverage gate: `g = 1` if any target is observed, else 0.
- Chemistry fallback: `g = 0`. This must match chemistry exactly.
- Random reliability: gates in `[0, 1]` from a SHA-256 of the identifier and
  seed 101.
- Permuted reliability: the coverage vector is permuted inside the evaluation
  set with seed 202.
- Oracle: `g = 1` when early fusion is closer to the label than chemistry.
  This uses labels. It is not deployable.
- Isotonic regression fit on adult out-of-fold predictions and applied to
  pediatric predictions, per endpoint, for chemistry and early fusion.

The pre-specified gate for the third confirmatory test is `g = coverage`.

## Validation inside the adult domain

Five-fold `GroupKFold` on Bemis–Murcko scaffolds, deterministic given the
sorted InChIKey order. Out-of-fold predictions are the in-domain result.
Folds are checked for identifier overlap and scaffold overlap. A failure
aborts the run.

## Confirmatory tests

All three use ridge, `alpha = 10`, the eligible endpoints, and the
structure-unseen slice. The endpoint-level effect is a difference of Spearman
correlations. The reported effect is the unweighted mean of those differences.
Uncertainty is a percentile bootstrap over endpoints, 1000 resamples, seed 0.
A paired bootstrap over compounds is also stored for early-versus-chemistry
and gate-versus-chemistry inside each endpoint.

1. Early fusion minus chemistry.
2. Early fusion minus the mean, across the five permutation seeds, of
   chemistry-plus-permuted-biology.
3. Coverage gate minus chemistry.

Benjamini–Hochberg q-values are reported across these three bootstrap tests.
A multimodal benefit is supported only if the 95% intervals for tests 1 and 2
both lie entirely above 0. A gate gain is supported only if the interval for
test 3 lies entirely above 0. The gate is also credited with reducing
negative transfer if it has strictly fewer endpoints with a negative
early-style delta than early fusion and its macro Spearman is at least
chemistry minus 0.005. That second gate rule is a count, not an accuracy
claim.

Wilcoxon signed-rank p-values on the endpoint deltas are descriptive. Endpoints
share compounds, so the independence assumption is false.

No superiority language is used when an interval includes 0.

## Metrics

For the trial task: Spearman, RMSE, MAE, calibration slope, n, mean label,
mean biological coverage. Accuracy is not used. AUROC is not used on the logit
incidence label.

For the Tox21 chemical arm, if the file is present: AUROC, AUPRC, MCC,
balanced accuracy, Brier, ECE, sensitivity and specificity at 0.5. An assay
is scored only with at least 20 positives and 20 negatives. Macro averages
are unweighted across scored assays.

## Exploratory, not confirmatory

- Within-Tox21 scaffold out-of-fold logistic regression on Morgan fingerprints,
  compared with a shuffled KFold of the same model. The random split is an
  optimistic reference.
- NR assay family versus SR assay family under the scaffold split. This is
  not label-aligned transfer.
- Chemical Checker signatures. Not run unless a probe succeeds and a license
  is recorded. Spaces B5 and other bioassay spaces would be leakage-suspect
  for Tox21 even then.
- Tox21 models scored against FAERS "serious". Not run. The prior probes
  (137 and 46 compounds, including a set with 4 negatives) are endpoint-
  mismatched and are not performance claims.
- Moderator correlations of the early-minus-chemistry delta with coverage,
  scaffold novelty, adult Spearman, n, and mean label. These are associations
  across about a dozen endpoints.
- Seen-structure metrics, which mostly restate same-drug continuity.
- Tree-model metrics.
- DANN and MMD are not fit. The sample is a few thousand labeled structures,
  the biological channel is sparse, and those methods would not be tunable
  without touching the transfer test.

## What would change the claim

Changing alpha, the eligibility cutoff, the primary slice, or the fusion
weight after seeing pediatric Spearman is a protocol violation. Sensitivity
tables may be computed. They are not a second primary test.
