# Initial audit

Date: 2026-10-19. Inputs: the handoff archive extracted outside the git tree,
then copied into gitignored `data/raw/` for the trial tables and into
`archive/` for notebooks, verdicts, and prior code. No model was fit for this
audit. Descriptive counts below come from the tables and from the feature
build that preceded confirmatory fitting.

## Research question the materials actually support

The notebooks ask whether Chemical Checker signatures improve Tox21 assay
prediction under scaffold cross-validation, and they attempt to score those
models on a FAERS-derived "serious" flag. The trial project asks whether a
15-target ChEMBL profile improves prediction of smoothed adverse-event
incidence from adult-only trials to pediatric-only trials.

Those are different questions. Tox21 is an in-vitro assay panel. The trial
label is a reporting construct from ClinicalTrials.gov. The FAERS flag is a
spontaneous-report fraction. This repository keeps them apart.

The question retained for the confirmatory run is: on pediatric-trial
structures that were not in the adult-trial set, does the ChEMBL profile
improve Spearman correlation of predicted versus observed logit incidence
beyond chemistry and beyond row-permuted biology, and does a pre-specified
coverage gate reduce negative transfer?

## Datasets and provenance

### Tox21

Notebook 1 downloaded
`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`.
The stored output is 7,831 rows and 12 assay columns. Positive rates in that
output range from about 0.029 (NR-PPAR-gamma) to about 0.162 (SR-ARE).
Standardization dropped 8 unparseable SMILES, collapsed 234 duplicate
structures, and a later cell dropped 2 round-trip failures, leaving 7,587
compounds and 75,521 non-missing labels. Scaffolds in the single split were
2,267. Five scaffold folds were sized 1,698, 1,473, 1,472, 1,472, 1,472.

MoleculeNet code is MIT. The challenge data are public. A dedicated license
line for the CSV was not found; see `data/dataset_manifest.csv`.

### ClinicalTrials.gov aggregate

The v3 handoff, dated in `archive/ctgov_v3/methods.md` as started 2026-09-19,
contains:

- 2,856 adult-trial structures and 835 pediatric-trial structures
- 634 InChIKeys in both
- 201 pediatric-only structures
- 2,228 adult scaffolds and 688 pediatric scaffolds, of which 537 also occur
  in the adult set
- 48 of the 201 pediatric-only structures still share a scaffold with an adult
  structure

Labels exist for 15 outcomes. Structure-unseen counts, measured before
fitting, are 200 (ANY_OTHER_AE, ANY_SERIOUS), 49 (CARDIAC), 24
(CONGENITAL_DEVELOPMENTAL), 82 (DERMATOLOGIC), 72 (ENDOCRINE_METABOLIC), 120
(GASTROINTESTINAL), 59 (HEMATOLOGIC), 32 (HEPATIC), 113 (IMMUNE_INFECTION),
144 (MORTALITY), 106 (NEUROLOGIC), 64 (PSYCHIATRIC), 34 (RENAL), 107
(RESPIRATORY). The frozen eligibility rule (at least 40 unseen structures)
therefore drops CONGENITAL_DEVELOPMENTAL, HEPATIC, and RENAL from the primary
mean. That rule was set at 40 because Spearman correlations below that size
are unstable, not because of a model result.

Inclusion rules in the prior methods: pediatric-only if max age is at most 17
with both ages parseable; adult-only if min age is at least 18; mixed trials
out; one non-control drug per arm; placebo, device, procedure, diet, and
ambiguous combinations out; within an arm the count is the max, not the sum;
across trials the counts are summed; the label is `(a+0.5)/(n+1)` on the logit
scale. Recomputation from the counts matches the stored logit to numerical
noise (max absolute gap about 1e-15).

Name mapping: 17,014 rows, 9,664 ok, 6,134 excluded as controls or non-drugs,
1,216 unresolved. Twelve InChIKeys have more than one SMILES; the mode is
kept. Every labeled structure in the outcome files has a SMILES and a bio
row. Feature standardization failed for 0 of 3,057 structures.

### ChEMBL profile

`chembl_target_vocab.json` lists 15 targets after an adult minimum-coverage
filter of 5% and a cap of 256. Mean missingness is 0.924. The fraction of
structures with any observed target is 0.381. The file says it is not
Chemical Checker. The ChEMBL release number is not in the file. License is
CC BY-SA 3.0. The matrix is not committed.

### Chemical Checker

Notebook 2 calls `signaturizer` for B1, B2, B4, B5, C3, C4, C5, D1, D2, each
128-dimensional, with retries and Drive checkpoints. The code license is MIT.
A data license was not established from the homepage extract. B5 is an HTS
bioassay space and is leakage-suspect for Tox21. Signatures were not
regenerated here.

### FAERS

Notebook 1 reads `OPENFDA_API_KEY` from the environment. The sanitized
notebook does not contain a key string. A prior summary says an earlier copy
of the notebook had a hard-coded key; that value is not in this archive.
Notebook outputs show pulls of 5,000 and 25,000 pediatric reports, 137
structures overlapping a Tox21 featurization, and a later 46-compound
chemical matrix. The 46-compound bootstrap intervals in the notebook output
are wide enough to be uninformative (chemistry versus chemistry-plus-biology
AUROC difference about 0.006, interval about -0.38 to 0.49). The handoff
description of 4 negatives matches an unusable class balance. EudraVigilance
is marked unavailable in the v3 limitations.

## Domain definition

"Adult" in the trial files means the trial's eligibility was adult, not that
the adverse events were shown to occur in adults as a biological population
free of confounding by indication. "Pediatric" means the trial's eligibility
maximum was 17 years. It is not a developmental-toxicology cohort, not a
pediatric FAERS case series, and not a clinically adjudicated toxicity label.

The shift that is defensible is: same outcome definition, different trial-age
eligibility, many test drugs absent from training, some test scaffolds still
shared. Transfer type: population-of-trials, with a large same-drug overlap
that must be removed before any generalization claim.

Within Tox21, the defensible shifts are scaffold holdout and a comparison of
nuclear-receptor versus stress-response assays. NR-to-SR is not a transfer of
one label onto another.

High-to-low biological coverage is a property of the ChEMBL matrix (most
compounds have coverage 0). It is a pre-specified stratum, not a separate
clinical population.

## Features and models in the prior notebooks

Notebook 1: Morgan 2048 and up to 214 RDKit descriptors, a single 80/10/10
scaffold split, then 5-fold scaffold GroupKFold. Models included a stratified
dummy, random forest, XGBoost, and an MLP. Descriptor medians in the early
cells were computed before the split, which is preprocessing leakage for those
cells. The later pooled scaffold-CV path is the one worth citing.

Stored single-split mean ROC AUC on the held-out scaffolds included about
0.707 for random forest on Morgan and about 0.759 for random forest on
Morgan plus descriptors. The 5-fold pooled task-macro ROC AUC in the quick
pipeline output is about 0.817 for random forest and about 0.811 for XGBoost
on the tasks that cell printed; the cell output is truncated after two tasks
in the captured text, so those two pooled numbers are not a verified 12-task
result from this audit. Notebook 2's complete 12-task table is the one used
below.

Notebook 2, XGBoost, 5-fold scaffold CV, pooled task-macro AUROC from the
saved cell output:

| Representation | Pooled task-macro AUROC |
| --- | --- |
| Chemistry | 0.8224 |
| Chemistry + true signatures | 0.8254 |
| Signatures only | 0.8007 |
| Chemistry + permuted signatures, seeds 101–505 | 0.7744 to 0.7807 |

The true-signature gain over chemistry is about 0.003. Permuted signatures
are worse than chemistry, which is consistent with extra noisy columns rather
than with a biological gain. This is prior exploratory work. It is not
re-executed here, and it is not an adult-to-pediatric result. The notebook
itself says so.

## Prior trial result

`archive/ctgov_v3/hypothesis_verdict.json` rejects a multimodal benefit.
`archive/ctgov_v3/transfer_metrics.csv` is the historical XGBoost, random
forest, and ridge table. An independent review in
`archive/prior_verdicts/pediatric_transfer_v3_independent_review.json` says
the overall Spearman near 0.12 is dominated by the 634 shared structures, that
the only BH-significant chemistry-versus-fusion contrast it highlights is
negative, and that the biological channel is sparse. This study refits the
question on structure-unseen compounds rather than treating that table as the
answer.

## Earlier experiments (not pooled)

The handoff verdict files, matching the E1–E12 list in the summary script:

| File | What it decided | Use here |
| --- | --- | --- |
| pediatric_transfer_aligned_v2 | Rejects ChEMBL-profile benefit | Historical null on an earlier aggregate |
| pediatric_transfer_v3_independent_review | Preserves the reject; flags same-drug leakage | Motivation for the unseen-structure primary slice |
| pediatric_transfer_ontogeny_v4_preliminary | Rejects age-conditioned enzyme features | Not re-run; ontogeny features are absent |
| pact_pediatric_conformal_v1 | Supports narrower intervals at nominal 90% coverage on a 501-structure test | Uncertainty result, not a multimodal claim |
| pact_validation_v2 | Reproduction passed; efficiency versus the best competitor called inconclusive; external validity not tested | Not a license to claim clinical intervals |
| react_fuse_data_aligned_v1 | Not supported; stop/go failed the residual test | Gate ideas may be reused only under a new protocol |
| pact_active_learning_v1 | Partially supported; primary interval includes 0 | Not confirmatory |
| dart_developmental_pharmacovigilance_v1 | Partially supported on secondary criteria; did not beat the best baseline (delta about -0.084) | FAERS reporting task, different metric |
| ccad_real_faers_v1 | Not supported; pilot failed; final test not opened | Different task |
| stage_constrained_selective_prediction_v1 | Confirmatory protocol not run | Different task |
| adaptive_pediatric_borrowing_v1 | Not supported; headroom gate failed | Different task |

E13 in the summary script had no artifact. It is not cited as a result.

## Leakage and bias already visible

1. Same InChIKey in adult and pediatric rows. Primary metrics must drop them.
2. Shared scaffolds among structure-unseen drugs (48/201). Report a
   scaffold-unseen slice.
3. Notebook 1 imputed descriptors before splitting.
4. The ChEMBL vocabulary was chosen using adult coverage. That is acceptable
   if frozen before pediatric evaluation. It is not acceptable to refit it
   using pediatric activity. This study does not refit it.
5. ChEMBL bioactivity can overlap Tox21 assays. It is not used as a Tox21
   feature.
6. Chemical Checker B5 can overlap Tox21. It is not used.
7. Pediatric labels must not enter alpha, weights, or eligibility cutoffs.
   The plan fixes those choices.
8. Mapping success is incomplete (1,216 unresolved names). The modeled set is
   biased toward structures that resolve in PubChem or ChEMBL.
9. Indication, trial design, and dose confound incidence. A Spearman
   correlation is not an effect of the drug.
10. The prior permutation left true biology on the pediatric rows. This study
    does the same, on purpose, and says so. The training correspondence is
    what is destroyed.
11. FAERS and trial counts were not added together in the v3 label. The loader
    rejects any other estimate type so that mix cannot reappear.

## Missing pieces

- ChEMBL release identifier.
- A data license for Chemical Checker signatures.
- A cached signature matrix that could be checked for assay overlap.
- EudraVigilance.
- An openFDA key, deliberately absent.
- Nested scaffold hyperparameter search and a 2,000-draw bootstrap. They are
  specified in `configs/full.yaml` and are not the confirmatory run.
- DANN and MMD, omitted as not tunable on this sample without using the test.

## Claim that the materials do not support

They do not support a statement that biological representations improve
pediatric toxicity prediction. The in-vitro result is a 0.003 AUROC difference
against a strong chemical model. The trial result was a pre-specified reject,
and it was not estimated on the unseen-structure slice that a generalization
claim needs.
