# Repository visibility

The GitHub repository `prnvmamidala-cmd/tox_21_multimodal_tranfer` is public.
That is acceptable for what this study commits.

## Recommendation

Keep the repository **public**, and keep the following out of git:

- ClinicalTrials.gov structure-level extracts and the drug-name map
- the ChEMBL-derived target-profile matrix (CC BY-SA 3.0; sharing it would
  pull a share-alike obligation onto that file, and the ChEMBL release is
  not identified)
- Chemical Checker signature matrices (data license not established here)
- FAERS reports and any openFDA key
- `chem_raw.csv` and other pre-joined feature matrices

The committed tree is code, configuration, the audit, synthetic-test behavior,
aggregate tables, figures, and saved predictions. Predictions contain an
InChIKey, an endpoint name, the smoothed logit label, and a model score.
They are committed so every reported number can be recomputed. They are not
patient records. The raw affected and at-risk counts, the drug-name map, and
the ChEMBL matrix stay out of git: the counts because they are a large
derived extract, the matrix because of CC BY-SA 3.0 and an unidentified
release.

## Stop condition

If a future change needs to commit a ChEMBL matrix, a Chemical Checker
matrix, an API key, or a FAERS case listing, stop and move the repository to
private before that commit. This study did not hit that condition.
