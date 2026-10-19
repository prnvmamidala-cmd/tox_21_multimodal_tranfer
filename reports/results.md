# Results

Numbers in this file are written by `src/toxtransfer/analysis/summarize.py` from `results/predictions/predictions.csv.gz`.
Do not edit the numeric sentences by hand. Re-run the summary script instead.

## Primary estimand

Adult-only-trial to pediatric-only-trial transfer of smoothed logit adverse-event incidence.
Primary rows are pediatric-trial structures whose InChIKey is absent from the adult-trial set.
Eligible endpoints: 12 of 15 (minimum adult n and minimum structure-unseen n are in the analysis plan).
Structure overlap (all endpoints, compound level): 634 shared, 201 pediatric-only.
ChEMBL target-profile width: 15 targets; mean missingness 0.9242.

## Confirmatory contrasts

Ridge alpha mode: prespecified; alpha 10.0.
Macro Spearman, chemistry: 0.0314.
Macro Spearman, early fusion: 0.0339.
Macro Spearman, coverage gate: 0.0308.

Early fusion minus chemistry: delta 0.0025 (95% CI -0.0106 to 0.0149; the interval includes 0). Early fusion minus the mean of the bio-permutation seeds: delta 0.0031 (95% CI -0.0143 to 0.0213; the interval includes 0). Coverage gate minus chemistry: delta -0.0007 (95% CI -0.0029 to 0.0014; the interval includes 0). The pre-specified rule for a multimodal benefit is not met. A benefit required both the early-versus-chemistry interval and the early-versus-permutation interval to lie above 0. The coverage gate does not meet the gain rule or the negative-transfer rule (negative-transfer endpoints: gate 7, early fusion 5). These statements are about smoothed adverse-event incidence in pediatric-only trials, on structures absent from the adult-trial training set. They are not clinical pediatric-toxicity performance.

Benjamini–Hochberg q-values for the three macro bootstrap tests (early−chem, early−permutation, gate−chem): 0.7600, 0.7600, 0.7600.
Adult out-of-fold weight on early fusion: 1.0000. Weight g in (1-g)*chemistry + g*early, chosen on adult scaffold out-of-fold Spearman. g=1 uses early fusion only.
Wilcoxon signed-rank p on endpoint early−chem deltas: 0.5186. Wilcoxon signed-rank assumes a symmetric null distribution of paired deltas and independent pairs. Endpoints share compounds, so independence fails; treat p as descriptive.

Eligibility sensitivity (same predictions, different pre-specified unseen-n cutoffs):
- min unseen n=20: 15 endpoints, macro early−chem 0.0049, macro gate−chem -0.0023
- min unseen n=60: 10 endpoints, macro early−chem 0.0019, macro gate−chem -0.0014

Exploratory moderator associations (Spearman across endpoints, not compounds):
- mean_coverage_unseen: -0.5324 (n=12). Exploratory association across endpoints. Not a causal effect.
- frac_scaffold_novel: 0.5464 (n=12). Exploratory association across endpoints. Not a causal effect.
- adult_chem_spearman: -0.1119 (n=12). Exploratory association across endpoints. Not a causal effect.
- n_unseen: -0.2312 (n=12). Exploratory association across endpoints. Not a causal effect.
- mean_y_unseen: 0.1748 (n=12). Exploratory association across endpoints. Not a causal effect.

Multimodal benefit supported: no.
Coverage-gate macro gain supported: no.
Coverage gate reduces the negative-transfer count under the pre-specified rule: no.

## Endpoint deltas (eligible, structure-unseen, early fusion minus chemistry)

- DERMATOLOGIC: n=82, chem 0.0298, early -0.0243, delta -0.0541
- ANY_OTHER_AE: n=200, chem 0.0443, early 0.0316, delta -0.0127
- MORTALITY: n=144, chem 0.0117, early 0.0004, delta -0.0113
- ENDOCRINE_METABOLIC: n=72, chem -0.0169, early -0.0211, delta -0.0042
- NEUROLOGIC: n=106, chem 0.0526, early 0.0523, delta -0.0003
- ANY_SERIOUS: n=200, chem 0.0477, early 0.0520, delta 0.0043
- CARDIAC: n=49, chem 0.2402, early 0.2449, delta 0.0047
- HEMATOLOGIC: n=59, chem -0.1594, early -0.1526, delta 0.0068
- RESPIRATORY: n=107, chem 0.0815, early 0.0915, delta 0.0100
- IMMUNE_INFECTION: n=113, chem -0.0309, early -0.0171, delta 0.0138
- GASTROINTESTINAL: n=120, chem -0.0012, early 0.0223, delta 0.0235
- PSYCHIATRIC: n=64, chem 0.0777, early 0.1275, delta 0.0498

## Exploratory intervals

These endpoint-bootstrap intervals are not in the three-test confirmatory family and are not Benjamini–Hochberg adjusted with those tests.
CORAL and importance weighting use unlabeled pediatric chemistry. Adult Spearman is the in-domain scaffold out-of-fold result on the same eligible endpoints.

- macro Spearman, chemistry: 0.0314 (95% CI -0.0165 to 0.0828; n=12).
- macro Spearman, early fusion: 0.0339 (95% CI -0.0165 to 0.0856; n=12).
- macro Spearman, biology only: 0.0723 (95% CI 0.0042 to 0.1441; n=12).
- macro Spearman, missingness only: 0.0246 (95% CI -0.0381 to 0.0821; n=12).
- macro Spearman, CORAL chemistry: 0.0985 (95% CI 0.0172 to 0.1753; n=12).
- macro Spearman, importance-weighted chemistry: 0.0239 (95% CI -0.0154 to 0.0639; n=12).
- macro adult-scaffold Spearman, chemistry: 0.0220 (95% CI 0.0092 to 0.0386; n=12).
- biology minus chemistry: 0.0409 (95% CI -0.0383 to 0.1220; n=12).
- missingness minus chemistry: -0.0068 (95% CI -0.0747 to 0.0739; n=12).
- CORAL chemistry minus chemistry: 0.0671 (95% CI -0.0588 to 0.1711; n=12).
- importance-weighted chemistry minus chemistry: -0.0075 (95% CI -0.0469 to 0.0249; n=12).
- early fusion without indicators minus chemistry: 0.0025 (95% CI -0.0041 to 0.0100; n=12).

## Other slices (exploratory)

Structure-seen rows are the same InChIKey in an adult trial and a pediatric trial. They are continuity, not the primary generalization test. Scaffold-unseen is a sensitivity slice. Coverage strata split the structure-unseen rows.

- all_pediatric / bio: macro Spearman 0.0766, mean n 514.3333, endpoints 12.
- all_pediatric / chem: macro Spearman 0.1434, mean n 514.3333, endpoints 12.
- all_pediatric / early: macro Spearman 0.1480, mean n 514.3333, endpoints 12.
- scaffold_unseen / bio: macro Spearman 0.0582, mean n 84.8333, endpoints 12.
- scaffold_unseen / chem: macro Spearman 0.0180, mean n 84.8333, endpoints 12.
- scaffold_unseen / early: macro Spearman 0.0183, mean n 84.8333, endpoints 12.
- structure_seen / bio: macro Spearman 0.0837, mean n 404.6667, endpoints 12.
- structure_seen / chem: macro Spearman 0.1923, mean n 404.6667, endpoints 12.
- structure_seen / early: macro Spearman 0.1959, mean n 404.6667, endpoints 12.
- structure_unseen / bio: macro Spearman 0.0723, mean n 109.6667, endpoints 12.
- structure_unseen / chem: macro Spearman 0.0314, mean n 109.6667, endpoints 12.
- structure_unseen / early: macro Spearman 0.0339, mean n 109.6667, endpoints 12.
- unseen_coverage_positive / bio: macro Spearman 0.1843, mean n 24.0000, endpoints 12.
- unseen_coverage_positive / chem: macro Spearman 0.2040, mean n 24.0000, endpoints 12.
- unseen_coverage_positive / early: macro Spearman 0.1831, mean n 24.0000, endpoints 12.
- unseen_coverage_zero / bio: macro Spearman NA, mean n 85.6667, endpoints 12.
- unseen_coverage_zero / chem: macro Spearman -0.0093, mean n 85.6667, endpoints 12.
- unseen_coverage_zero / early: macro Spearman -0.0081, mean n 85.6667, endpoints 12.

## Tox21 chemical arm

NR versus SR is a comparison of assay families under the same split, not a label-aligned transfer. Random-split AUROC is an optimistic reference that allows scaffold leakage.
Compounds: 7587. Labels: 75521. Biological signatures: not_run. Chemical Checker signatures were not regenerated. The signaturizer stack was not executed in this run; see reports/signaturizer_probe.json.
- scaffold_oof: macro AUROC 0.6895, macro AUPRC 0.2440, NR macro AUROC 0.7034, SR macro AUROC 0.6700, tasks 12.
- random_oof: macro AUROC 0.7331, macro AUPRC 0.3200, NR macro AUROC 0.7466, SR macro AUROC 0.7141, tasks 12.
Random minus scaffold macro AUROC: 0.0436. This is an optimistic-split gap, not a transfer result.

