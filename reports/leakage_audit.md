# Leakage audit

Automated checks from a confirmatory run are written to
`results/leakage_checks.json`. This note is the design those checks implement.
A failed check raises and stops the fit.

## Checks

1. **Compound identity.** Adult scaffold folds are `GroupKFold` on InChIKey
   groups only in the sense that each row is one InChIKey. The integrity
   function also rejects any identifier on both sides. The primary transfer
   slice is the set difference of pediatric and adult InChIKeys. The loader
   rejects duplicate structure-endpoint-domain rows before the split.

2. **Scaffolds.** Folds are grouped by Bemis–Murcko scaffold. The same
   function rejects a scaffold on both sides. Acyclic molecules do not share
   one empty scaffold id. Among the 201 pediatric-only structures, 48 share a
   scaffold with an adult structure. They remain in the primary slice, and a
   scaffold-unseen slice is reported beside it. That slice is a sensitivity
   analysis, not a silent filter.

3. **Preprocessing.** Medians, clipping quantiles, variance masks, and ridge
   scales are fit on the training matrix of the current call.
   `chem_raw.csv` is not read by the model code. A unit test mutates a test
   matrix and checks that the training median is unchanged. The run repeats
   that check on the first adult fold.

4. **Hyperparameters.** Reduced-run ridge uses alpha 10, fixed in
   `configs/models.yaml` before the fit. Tree sizes are fixed. Fusion weights
   are chosen on adult out-of-fold predictions. The full config's inner
   GroupKFold, if run, receives only the outer training rows. Pediatric
   labels are not an argument of `select_alpha_groupkfold`.

5. **Biology is not the label.** The ChEMBL profile is a target-activity
   summary. The label is trial adverse-event incidence. They can be
   confounded by indication. That is bias, and it is not the same failure as
   pasting the label into the feature matrix. Tox21 is not predicted from
   ChEMBL in this repository because those assays can coincide.

6. **Identifiers.** One row per InChIKey after the name map. Conflicting
   SMILES are reduced to the mode (12 keys). The InChIKey is not regenerated
   after RDKit cleanup, so salt stripping cannot silently split one trial
   structure into two modeling keys.

7. **Missing biology.** Missing targets are an indicator, not a code for the
   domain. Coverage is a function of those indicators. The gate at coverage 0
   is exactly the chemistry prediction, which the smoke test checks.
   Median imputation of missing pChEMBL, used only in the no-indicator
   ablation, is fit on training rows.

8. **Target-domain information.** Zero-shot ridge, forests, and boosting see
   pediatric features only at prediction time. CORAL and importance weighting
   see unlabeled pediatric chemistry and are stored under representation names
   that say `transductive`. The target-only reference is a separate phase,
   `target_reference`, and uses a scaffold holdout inside the unseen pediatric
   set. Its labels are not used to pick the zero-shot model.

9. **Endpoint filtering.** Eligibility is applied when the macro mean is
   computed, not by refitting on the endpoints that look good. Predictions for
   ineligible endpoints remain in the file. The cutoffs are 200 adult and 40
   unseen structures, with 20 and 60 as sensitivity cutoffs.

10. **Permutations.** Row permutation uses a seed from {101, 202, 303, 404,
    505}, is applied to the training block only, and keeps the multiset of
    biological rows. The run checks the multiset on the first training fold.
    Label permutation and reliability permutation are separate controls and
    do not use the evaluation labels to build the permutation.

## Known leaks in the archived notebooks, not inherited

Notebook 1's early descriptor cells fill medians before the scaffold split.
Notebook 2's Chemical Checker permutation, as written in the cell that
shuffles the full compound list before the fold loop, can attach a test
compound's signature to a training row. The confirmatory permutation is
inside the training index of each fit. The historical Tox21 numbers are not
recomputed, so that notebook bug is not "fixed" by citation; it is a reason
not to treat the 0.8254 AUROC as a confirmatory estimate.

## What this audit cannot prove

It cannot prove that trial incidence is independent of the ChEMBL literature
that defined the targets. Drugs with dense target profiles are annotated
drugs. The coverage gate is there so that density is visible, not so that
confounding disappears.
