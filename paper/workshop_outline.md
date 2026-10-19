# Workshop outline

This is a draft outline for a workshop submission. It is not an accepted paper,
and the claims below stop where `reports/results.md` stops.

## Title candidates

1. When does a target profile help? Chemical versus ChEMBL features for adult-trial to pediatric-trial adverse-event incidence.
2. Same-drug continuity is not generalization: a pre-specified multimodal check on ClinicalTrials.gov adverse-event rates.
3. Reliability-aware fusion does not rescue a sparse biological channel under structure shift.

The phrase "pediatric toxicity" is the wrong name for this estimand. Tox21 labels
are in-vitro assay activities. The trial labels are smoothed adverse-event
incidence in adult-only versus pediatric-only trials.

## Abstract structure

- One sentence on the question: whether a biological representation improves
  toxicity-related prediction when the compounds change.
- One sentence on why the obvious adult-to-pediatric reading is not available
  in these files.
- The renamed confirmatory task and the primary slice (structure-unseen).
- The three pre-specified contrasts and the negative result, with intervals.
- The continuity contrast (structure-seen Spearman much larger than
  structure-unseen) as the interpretation, not as a second primary test.
- The Tox21 scaffold-versus-random chemical arm as a separate, positive-control
  setting where chemistry does rank labels.
- An explicit non-claim: not a clinical tool, and not a verdict on biological
  representations in general.

## Introduction

Toxicity labels get attached to whatever table is convenient. A high AUROC on
Tox21 does not answer whether the same model, or a richer biological feature,
tracks adverse events in a new set of structures. Prior work in this project
family mixed those questions. This paper separates them.

The confirmatory question is narrow. Given structures that appear in
pediatric-only trials and not in adult-only trials, does appending a 15-target
ChEMBL profile to Morgan fingerprints and RDKit descriptors improve Spearman
correlation with smoothed logit adverse-event incidence, relative to chemistry
and relative to chemistry plus row-permuted biology? A coverage gate is the
reliability rule under test: trust the multimodal score only to the extent the
profile is observed.

## Related work

- MoleculeNet / Tox21 as an in-vitro multi-task benchmark (Wu et al., Chemical
  Science, 2018, DOI 10.1039/C7SC02664A). Cite the scaffold-split literature
  only for the split we actually run (Bemis–Murcko `GroupKFold`).
- Chemical Checker (Duran-Frigola et al., Nature Biotechnology, 2020, as cited
  on chemicalchecker.com). This submission does not rerun those signatures.
  The data license was not established here. Notebook 2's archived Tox21
  comparison (chemistry 0.8224 versus chemistry plus signatures 0.8254, XGBoost)
  stays in the audit as an exploratory historical number.
- ChEMBL (CC BY-SA 3.0). The profile used here is 15 targets inherited from a
  prior adult-coverage filter. The release identifier was not in the handoff.
- CORAL and importance weighting as transductive feature alignment, reported
  as secondary because they see unlabeled target chemistry.
- Earlier experiments in `archive/prior_verdicts/` (ontogeny, conformal
  intervals, active learning, and several FAERS designs). Their verdicts are
  context. Their metrics are not pooled with these Spearman values or with
  Tox21 AUROC.

## Formal problem

Source domain: adult-only interventional trials, minimum age at least 18.
Target domain: pediatric-only trials, maximum age at most 17. Mixed-age trials
were already excluded by the upstream aggregate.

Label: `(affected + 0.5) / (at_risk + 1)`, then logit. The loader checks that
this matches the stored column. The quantity is a reporting construct. It is
not a causal risk and not a developmental-toxicology call.

Primary population: pediatric-trial structures whose InChIKey is absent from
every adult-trial row (201 structures at the compound level). Endpoints need
at least 200 adult rows and at least 40 structure-unseen rows. That rule was
fixed before the confirmatory fit and drops congenital/developmental, hepatic,
and renal. Twelve endpoints remain.

Representations: Morgan radius 2, 2048 bits; 217 RDKit descriptors with a
parameter-free `log1p` on `Ipc` and `AvgIpc`; 15 target values plus 15
missingness indicators. Preprocessing (median, clipping, variance filter,
scaling) is fit on training rows only.

Primary model: ridge, solver `lsqr`, alpha 10, not tuned on the target.
Trees are a robustness check at pre-specified sizes. Fusion weights and the
isotonic maps are fit on adult scaffold out-of-fold predictions only.

## Data and representation provenance

See `data/dataset_manifest.csv` and `reports/initial_audit.md`.

Biological coverage is low: mean missingness 0.9242, and about 38% of
structures have any observed target. A null result applies to this channel.
It does not apply to Chemical Checker signatures, which were not built
(`reports/signaturizer_probe.json`).

## Leakage-controlled protocol

`reports/leakage_audit.md` and `results/leakage_checks.json` record the checks
that ran: no shared InChIKey or scaffold across adult folds, permutation
preserves the training multiset, preprocessing statistics ignore the held-out
fold, and the structure-unseen mask is disjoint from the adult set.

Known historical leaks that this run does not repeat: descriptor imputation
before the split in Notebook 1, and a full-matrix biology permutation before
folding in Notebook 2. Those notebooks are in `archive/notebooks/` and are not
re-executed as confirmatory evidence.

Hyperparameters for the claim were not selected on pediatric labels. The full
inner-scaffold alpha search is specified in `configs/full.yaml` and was not run.

## Reliability-aware fusion

The pre-specified gate is `g = fraction of the 15 targets observed`, with
prediction `(1-g) * chemistry + g * early fusion`. Comparisons that were fit
or computed: fixed 50/50 mixtures, a global weight and endpoint weights chosen
on adult out-of-fold Spearman, a binary coverage gate, chemistry fallback,
SHA-256 random gates (seed 101), coverage permuted inside the evaluation set
(seed 202), and a non-deployable oracle that looks at the label.

Adult out-of-fold selection set the global weight to 1, which discards
chemistry and returns early fusion. There was essentially no in-domain
headroom with which to learn a cautious gate.

## Results to show

Lead with the three confirmatory intervals from `reports/results.md`:

- Early fusion minus chemistry: 0.0025 (95% CI -0.0106 to 0.0149).
- Early fusion minus the mean of permutation seeds 101, 202, 303, 404, 505:
  0.0031 (95% CI -0.0143 to 0.0213).
- Coverage gate minus chemistry: -0.0007 (95% CI -0.0029 to 0.0014).

Benjamini–Hochberg q-values on the three bootstrap tests are all 0.7600.
Multimodal benefit is not supported. The gate has 7 negative-transfer
endpoints against 5 for early fusion, so it also fails the negative-transfer
count rule.

Then show, labeled exploratory:

- Chemistry's own macro interval on the primary slice includes 0.
- Structure-seen chemistry is 0.1923; structure-unseen chemistry is 0.0314.
- Biology-only and CORAL have level intervals above 0 and contrast intervals
  that include 0. Do not promote them to a win.
- On rows with any observed target, mean n is 24 and early fusion does not
  beat chemistry.
- Tox21 Morgan logistic regression: scaffold macro AUROC 0.6895, random
  0.7331, gap 0.0436; NR 0.7034 versus SR 0.6700 under the scaffold split.
  This is an assay-family comparison, not a transfer.

Figures `results/figures/01` through `13` are the schematic, composition,
coverage, per-endpoint comparison, paired differences, permutation control,
transfer gap, scaffold novelty, coverage, calibration, risk-coverage,
ablations, and failure cases.

## Negative-transfer analysis

Five of twelve eligible endpoints have a negative early-minus-chemistry
delta. Dermatologic is the largest negative point estimate. The coverage
gate increases that count to seven. The oracle, which uses labels, is higher
than the deployable rules and is only a ceiling (`results/tables/transfer_macro.md`).

Moderator correlations across the twelve endpoints (coverage, scaffold
novelty, adult Spearman, n, mean label) are exploratory and are not causal.

## Biological interpretation

The pattern that the data support is prosaic. Where the pediatric structure
was already in the adult trials, chemistry ranks smoothed incidence moderately
(macro Spearman about 0.19). Where the structure is new, that association
collapses, and adding a mostly missing 15-target profile does not restore it
under the pre-specified tests. In-domain adult scaffold correlation is itself
only about 0.02, so there is little rank signal to transfer. The trial label
is entangled with indication and reporting. A biological feature would have
to be both observed and relevant to that entanglement before fusion could help.

## Limitations

Use `reports/limitations.md` rather than softening it. The items a reviewer
will care about most are the sparse unidentified ChEMBL vocabulary, the
pre-specified alpha that leaves chemistry poorly scaled, the small
structure-unseen sets, and the fact that a multimodal null here is not a
null about biology in general.

## Broader impacts

Publishing a negative, correctly named result is the useful outcome. Misnaming
the endpoint as pediatric toxicity would overstate what a trial-reporting
model can do. The repository stays public only while ChEMBL matrices,
signature files, and API keys stay out.

## Reproducibility statement

`python scripts/reproduce_all.py --config configs/reduced.yaml` is the
confirmatory command. `configs/smoke.yaml` is the synthetic check.
`configs/full.yaml` is specified and was not executed. Predictions, the fit
manifest, and the generated results file are in the repository. Details are
in `reports/reproduction.md` and `reports/experiment_log.md`.
