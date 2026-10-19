# Methods

The confirmatory definitions are in `reports/analysis_plan.md`. This note
records the implementation those definitions map to.

## Cohort construction

The handoff already restricted trials to single-drug arms and to age windows
that do not mix children and adults. This repository does not re-query
ClinicalTrials.gov. It checks that every label row is
`clinical_trial_smoothed_incidence` and that the age bucket matches the file.
The modeling label is recomputed from `num_affected` and `num_at_risk`.

Structures are keyed by InChIKey. Twelve InChIKeys had more than one
canonical SMILES in the name map; the most frequent SMILES is kept. RDKit
standardization keeps the largest fragment, cleans the molecule, and removes
charges. The InChIKey is not recomputed, so the join to the trial labels stays
on the handoff identifier. Zero molecules failed standardization in the
2026-10-19 feature build.

## Why chemistry is recomputed

`chem_raw.csv` is not a model input. A matrix that was imputed, clipped, or
filtered on adult and pediatric rows together would leak test information into
preprocessing. Fingerprints and descriptors are stored raw. Each ridge or tree
fit learns its own median, clipping quantiles, variance mask, and, for ridge,
scale.

## Splits

Adult out-of-fold prediction uses `sklearn.model_selection.GroupKFold` on
Bemis–Murcko scaffolds without chirality. Compounds are ordered by InChIKey
before the split. The run raises if any fold shares an identifier or a
scaffold.

The transfer model is fit on all adult rows for that endpoint and applied to
all pediatric rows. Metrics are then sliced. The primary slice drops every
pediatric InChIKey that occurs anywhere in the adult tables, including other
endpoints.

## Models

Ridge uses `solver="lsqr"` and `alpha=10` in the reduced configuration.
Random forests use 40 trees, depth 12, minimum leaf 5, `max_features="sqrt"`.
XGBoost uses 40 trees, depth 3, learning rate 0.1, histogram tree method.
Logistic regression for Tox21 uses `C=1` and LBFGS, with scaling fit inside
each fold, on Morgan bits only.

## Fusion

`src/toxtransfer/models/fusion.py` implements the mixture
`(1 - g) * p_chem + g * p_other` with `g` clipped to `[0, 1]`. The summary
script builds these rows from saved chemistry and early-fusion predictions.
It does not refit them, except isotonic regression, which is fit on adult
out-of-fold predictions only.

## Adaptation

`project_then_coral` fits PCA on adult chemistry after the adult preprocessor
and aligns second-moment structure to unlabeled pediatric chemistry in that
space (CORAL). Importance weights are the clipped odds from a logistic
regression that distinguishes adult from pediatric feature rows. Both are
labeled transductive in the prediction file.

## Statistics

`src/toxtransfer/statistics/inference.py` implements the paired bootstrap,
the endpoint-mean bootstrap, Wilcoxon signed-rank, and Benjamini–Hochberg.
The primary interval is the endpoint bootstrap. Compound-level intervals are
secondary because the claim is about a set of endpoints.

## Tox21

If `data/raw/tox21/tox21.csv.gz` is present, the reduced run also fits
chemical logistic regression under scaffold GroupKFold and under a shuffled
KFold. Duplicate standardized SMILES are collapsed by the mean label, with
exact 0.5 ties set to missing. Biological signatures are not filled in with
noise or with ChEMBL activities. ChEMBL activities can overlap the Tox21
assays, so using them as features would be a leakage risk even if the API
were called.

## Software

Pinned versions are in `requirements.txt`. The run manifest records the git
commit, a hash of the merged config, package versions, and the host platform.
