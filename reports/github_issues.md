# Proposed GitHub issues

These were not opened through the GitHub API. The agent environment's GitHub
CLI is read-only, so the issues live here until someone creates them in the
repository UI.

## 1. Chemical Checker signatures were not regenerated

**Title:** Blocked: signaturizer import failed, so the Chemical Checker arm is absent

**Body:**

`scripts/probe_signaturizer.py` records a `ModuleNotFoundError` in
`reports/signaturizer_probe.json`. No signature matrix was written. Notebook 2's
archived Tox21 comparison (XGBoost, chemistry 0.8224 versus chemistry plus nine
128-d spaces 0.8254) is exploratory history, not a result of this repository.

Before any rerun:

- Confirm a data license for Chemical Checker. The code license is MIT. That
  is not a data license. The homepage citation is Duran-Frigola et al., Nature
  Biotechnology 2020; the DOI was not re-fetched here.
- Exclude space B5 and other bioassay spaces from any Tox21 model. They can
  overlap the assay labels.
- Do not install a TensorFlow stack into the confirmatory environment just to
  force a number.

## 2. Nested ridge alpha was not run

**Title:** Optional: run configs/full.yaml without touching the transfer labels

**Body:**

The confirmatory ridge uses alpha 10. Adult chemistry calibration slopes are
about 0.014, so the predictions are poorly scaled. `configs/full.yaml` selects
alpha by inner scaffold GroupKFold on training rows and uses 2000 bootstrap
draws. That command was not executed.

If it is run, freeze the chosen alpha from adult folds only, then score the
existing structure-unseen slice once. Do not pick the alpha by pediatric
Spearman.

## 3. EudraVigilance is not in the handoff

**Title:** Blocker: no EudraVigilance extract

**Body:**

The archived code has an Eudra detector and no extract that this study could
audit. Do not invent a European spontaneous-report arm. If a file appears
later, define the endpoint before fitting and keep it separate from the
ClinicalTrials.gov incidence label.

## 4. Do not revive Tox21-to-FAERS "serious" as a performance claim

**Title:** Documentation: FAERS pediatric probes stay invalid

**Body:**

Notebook 1's 137-compound probe and Notebook 2's 46-compound probe (including a
set with 4 negatives) compare in-vitro assay scores to a FAERS reporting
fraction. The endpoints do not match. This repository does not report an AUROC
for that comparison. A future issue that asks for one should be closed unless
the label definition changes and a new protocol is frozen first.

## 5. Structure-unseen samples are small

**Title:** Limitation: 201 structure-unseen compounds, 12 endpoints, shared compounds

**Body:**

Primary n per endpoint ranges from 49 (cardiac) to 200 (the two "any" labels).
Endpoints share compounds, so the Wilcoxon p-value is descriptive only. The
endpoint bootstrap matches the claim and is still a bootstrap of 12 numbers.
Coverage-positive unseen rows average n=24. Do not add a post-hoc subgroup as
a primary result.

## 6. ChEMBL vocabulary is 15 targets and the release is unknown

**Title:** Data gap: unidentified ChEMBL release and a 15-target adult-coverage vocabulary

**Body:**

`chembl_target_vocab.json` does not name a release. Mean missingness is 0.9242.
The null multimodal result indicts this channel only. Rebuilding a wider
profile is a new representation, not a silent rerun. The matrix is CC BY-SA 3.0
and must stay gitignored unless the repository's public status is reconsidered.
See `reports/repository_visibility.md`.

## 7. Manuscript tasks

**Title:** Manuscript: workshop outline is drafted; figures need a human pass

**Body:**

`paper/workshop_outline.md` and `paper/abstract_draft.md` track
`reports/results.md`. Remaining human work: read figures 01–13 for axis
wording, confirm the design schematic matches the frozen protocol, and decide
whether the exploratory biology-only and CORAL intervals belong in a supplement
only. Do not rewrite the three confirmatory sentences by hand.

## 8. DANN and MMD stay out until there is a tuning plan

**Title:** Optional method: DANN/MMD only with a locked validation plan

**Body:**

Both were excluded because the labeled structure count is a few thousand, the
biological channel is sparse, and a fair tuning loop would be tempted to use
the transfer test. Add either method only with a source-domain validation rule
written down first.

## 9. Pre-register the biology-only contrast before citing it

**Title:** Analysis: biology-only level interval is exploratory

**Body:**

Biology-only macro Spearman on the primary slice is 0.0723 (95% CI 0.0042 to
0.1441). Biology minus chemistry is 0.0409 (95% CI -0.0383 to 0.1220). The
second interval includes 0, and neither contrast was in the confirmatory
family. A follow-up protocol can pre-register biology-only versus chemistry.
This run cannot be re-labeled as that test.
