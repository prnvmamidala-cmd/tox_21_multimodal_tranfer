# Abstract draft

Biological features are often added to chemical fingerprints in the hope that
they travel to a new population. The files available for that hope do not
support the usual caption. Tox21 is an in-vitro assay panel. The clinical-trial
tables used here record smoothed adverse-event incidence, not a clinically
validated pediatric toxicity endpoint. We therefore study adult-only-trial to
pediatric-only-trial transfer, and we score only structures whose InChIKey is
absent from the adult trials.

On that slice, twelve endpoints meet a cutoff fixed before fitting (at least
200 adult rows and 40 unseen structures). A ridge model with pre-specified
alpha 10, using Morgan fingerprints and RDKit descriptors, has macro Spearman
0.0314. Early fusion with a 15-target ChEMBL profile, mean missingness 0.9242,
reaches 0.0339. The difference is 0.0025 (95% endpoint-bootstrap CI -0.0106 to
0.0149). The difference against five training-row permutations of the profile
is 0.0031 (95% CI -0.0143 to 0.0213). A coverage gate that falls back to
chemistry when targets are missing differs from chemistry by -0.0007 (95% CI
-0.0029 to 0.0014) and increases the number of negative-transfer endpoints
from 5 to 7. The pre-specified multimodal benefit is not supported.

The same chemistry model scores 0.1923 on structures that do appear in both
trial sets. Apparent "pediatric" performance in the pooled table is largely
same-drug continuity. A separate Tox21 arm, logistic regression on Morgan
fingerprints only, does rank assay labels: scaffold-split macro AUROC 0.6895
versus 0.7331 under a random split. Chemical Checker signatures were not
regenerated. Nothing in this study is a clinical decision rule.
