# FROZEN V5 PROTOCOL

This file is the pre-registration for phase V5. Outcome models are not fit until this marker is in the file. V5 does not edit `paper/`, `reports/results.md`, or `results/predictions/`.

The confirmatory V4 claim stays the ridge early-fusion result already stored. V5 asks a different question: whether other combinations of a biological channel, or a small neural model, beat a matched chemistry model on the same structure-unseen slice.

## Slice and labels

Labels and the structure-unseen flag come from the committed file `results/predictions/predictions.csv.gz`. The raw ClinicalTrials.gov tables are not in this workspace. Structures are recovered by InChIKey from PubChem. ChEMBL pChEMBL values for the 15 targets in `archive/ctgov_v3/chembl_target_vocab.json` are queried from the public ChEMBL API when that API responds. Those rebuilt features are a new matrix, not a copy of the gitignored original profile, and the report must say so.

The evaluation slice is the original structure-unseen InChIKeys for which a molecule could be rebuilt. Every comparison uses that same subset. The twelve endpoints are the eligible list in `configs/v5.yaml`. Congenital/developmental, hepatic, and renal stay out.

Pediatric labels are not used to choose a hyperparameter, a weight, a gate, or an epoch.

## Primary arms

Each arm is compared with its matched chemistry model and with the mean of the same arm under biology permutations of the training rows. Permutation seeds are 101, 202, 303, 404, and 505. Test rows keep their own biology. The metric is Spearman on the logit incidence label. The reported effect is the unweighted mean of the twelve endpoint differences. Uncertainty is a percentile bootstrap over endpoints, 1000 draws, seed 0.

A primary arm is supported only if both 95% intervals lie entirely above 0: arm minus matched chemistry, and arm minus the mean of its five permutation fits. Benjamini–Hochberg q-values are computed across every primary contrast that is not blocked (both contrasts for each arm). An interval that includes 0 is not a win. A blocked arm is not put into the q-value family.

Arms:

1. `late_stack`. Ridge chemistry and ridge biology-only are fit with penalty 10. One weight on the grid `{0, 0.25, 0.5, 0.75, 1}` is chosen by adult scaffold out-of-fold Spearman and applied to models refit on all adult rows.
2. `miss_gate`. `g = clip(a + b * coverage)` with `a` and `b` on the grids in `configs/v5.yaml`, chosen on adult out-of-fold predictions. Coverage is the fraction of the 15 rebuilt targets that are observed. The biology score inside the gate is the biology-only ridge.
3. `imputed_early`. Each target is imputed by a ridge from chemistry, fit on adult rows where that target is observed. A ridge then maps chemistry plus the imputed targets to the trial label.
4. `mechanism`. A ridge maps the predicted target activities, and not the fingerprint, to the trial label. Concatenating those same predicted activities with chemistry is the `imputed_early` arm, so it is not a second primary test.
5. `tox21_late`. The twelve Tox21 assay outcomes, plus missingness indicators, are a biology block. A ridge on that block is stacked with chemistry using a weight chosen on an adult scaffold holdout. If fewer than 30 adult structures have any observed Tox21 label, this arm is blocked before any test Spearman is computed.
6. `mlp_early`. A one-hidden-layer multitask network (64 units, 12 epochs, batch 256, learning rate 0.001, weight decay 0.0001) on Morgan fingerprints plus the rebuilt target block. Seeds 0, 1, and 2 are averaged. The matched chemistry model is the same network without the target block. Hyperparameters are frozen in this file, not searched.

Ridge chemistry for arms 1–6 uses Morgan radius 2 (2048 bits) and RDKit descriptors, penalty 10, solver `lsqr`. Adult out-of-fold predictions use five-fold Bemis–Murcko `GroupKFold`. Acyclic molecules are not one shared group.

## Exploratory, not in the q-value family

- Whether the rebuilt biology block predicts the frozen ridge chemistry residual on adult out-of-fold rows.
- Spearman of saved and new models inside coverage bins.
- How well chemistry predicts each ChEMBL target (redundancy).
- Per-endpoint deltas.
- Zero-filled versus train-median biology inside an early ridge, without using the test set to choose which one is primary.
- A small message-passing network written in PyTorch. Chemprop and PyTorch Geometric are used only if they import. Otherwise they are blocked, and the small network is not renamed to Chemprop.
- Tox21 scaffold-split AUROC for the neural models, as a competence check, not as a transfer result.
- Multitask Tox21 pretraining before the trial fine-tune.
- Pretrained molecular embeddings, only if a weight file with a license that allows use is actually obtained.
- Chemical Checker signatures, only if `signaturizer` imports. A failure is a blocker.

## What would change the claim

Choosing a weight, a gate, a seed, or an arm after seeing the structure-unseen Spearman is a protocol violation. Exploratory tables may be computed. They are not a second primary test.
