# Methods

## Objective
Determine whether adding biological representations (`chembl_target_profile`) to chemical representations improves adult-to-pediatric transfer of drug-safety predictions.

## Data sources
- **ClinicalTrials.gov API v2** (`https://clinicaltrials.gov/api/v2/studies`): interventional studies with posted results and AE module.
- **FAERS via openFDA** (`https://api.fda.gov/drug/event.json`): spontaneous reports for pediatric-vs-adult ROR (external PV evidence only; **not incidence**).
- **EudraVigilance**: UNAVAILABLE — see raw/EUDRAVIGILANCE_NOT_AVAILABLE.txt.
- **ChEMBL web services** (`https://www.ebi.ac.uk/chembl/api/data`): experimental target activities for `chembl_target_profile`.
- **PubChem PUG REST**: name→structure.

## Inclusion / exclusion
- Pediatric-only: max age ≤17 with parseable min and max; adult-only: min age ≥18 with parseable min and max.
- Mixed child+adult trials excluded.
- Arms with exactly one identifiable non-control drug; exclude placebo/untreated/behavioral/device/dietary/procedure/ambiguous combinations.
- Outcomes mapped via MedDRA SOC when available else frozen keyword map.
- Within arm/category: **MAX** affected (not sum). Across trials: sum affected and at-risk by structure+outcome.
- Smoothed incidence `(a+0.5)/(n+1)`; logit as continuous target.

## Representations
- `chem`: Morgan/ECFP4 r=2 2048-bit + RDKit descriptors (adult-only median impute, drop constants, clip).
- `bio_only` / multimodal bio: `chembl_target_profile` (pChEMBL median per target; adult-coverage vocab; missing indicators). **Not** Chemical Checker / BioFeat / signaturizer.
- `chem_true_bio`: concatenation.
- `chem_permuted_bio_seed_{101,202,303,404,505}`: row-permute bio among adult compounds.

## Transfer protocol
- Train on adult only; predict pediatric **without** pediatric model fitting (primary zero-shot).
- Primary model: XGBoost regressor; robustness: RF, Ridge/ElasticNet.
- HPs tuned with adult-only scaffold/grouped CV; frozen before pediatric eval.
- Metrics: Spearman, RMSE, MAE on pediatric logit incidence.
- Seen vs unseen structures/scaffolds reported separately.
- Paired compound bootstrap ≥2000; BH correction across primary endpoint comparisons.

## Scientific guardrails
- Do not add raw CT.gov, FAERS, and EV counts together.
- Do not describe spontaneous reports as confirmed causal toxicities.
- Do not treat AE reporting frequency as incidence.
- Multimodal claimed helpful only if true bio beats chem-only **and** permuted-bio controls.

## Run metadata
- Started: 2026-09-19 11:11:07 PDT
- Seeds: [101, 202, 303, 404, 505]
- Package versions: see run_manifest.json
