# Data

Raw and derived matrices are gitignored. The repository commits the manifest,
code, and small summary outputs only.

## ClinicalTrials.gov handoff

Place these files in `data/raw/ctgov_v3/`:

- `ctgov_adult_structure_outcome.csv`
- `ctgov_pediatric_structure_outcome.csv`
- `drug_name_mapping.csv`
- `chembl_target_profile.csv`
- `chembl_target_vocab.json`

They are aggregates from a prior pipeline (adult-only vs pediatric-only
trials). This study recomputes Morgan fingerprints and RDKit descriptors from
`canonical_smiles`. It does not use `chem_raw.csv`, because that matrix may
have been imputed on the pooled adult-plus-pediatric set.

```bash
python scripts/download_or_prepare_data.py --handoff /path/to/ctgov_v3 --skip-tox21
```

## Tox21

```bash
python scripts/download_or_prepare_data.py --skip-tox21=false
```

The default URL is the DeepChem MoleculeNet file
`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`.
The download is gitignored.

## Chemical Checker

Signature matrices are not in this repository. Do not add them until the data
license and the overlap between signature source assays and Tox21 labels are
recorded. `scripts/probe_signaturizer.py` only checks whether the package
imports.

## What is not here

FAERS patient reports, EudraVigilance extracts, API keys, and the prior
experiment result directories are not part of the confirmatory inputs. Prior
verdicts are archived as JSON under `archive/prior_verdicts/` and are not
pooled with new metrics.
