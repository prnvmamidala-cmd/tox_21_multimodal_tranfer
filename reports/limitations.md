# Limitations

1. The pediatric target is a trial-eligibility construct. It is not a
   pediatric toxicity cohort and not a developmental-toxicology assay.
2. Smoothed incidence is confounded by indication, dose, schedule, and how
   completely a trial reports adverse events. Spearman correlation is not a
   causal drug effect.
3. Six hundred thirty-four of 835 pediatric-trial structures also appear in
   adult trials. Analyses that ignore that overlap mostly rediscover the same
   drug. The primary slice removes them and is much smaller (201 structures,
   and fewer per endpoint).
4. Forty-eight of those 201 structure-unseen compounds still share a
   Bemis–Murcko scaffold with an adult compound.
5. The biological representation is 15 ChEMBL targets with mean missingness
   0.924. A null result indicts this channel, not "biology" as a class.
   Chemical Checker signatures were not regenerated.
6. The ChEMBL release is unidentified. The vocabulary was inherited from the
   prior adult-coverage filter.
7. Name resolution dropped unresolved strings. The chemical space is whatever
   PubChem-style mapping succeeded for the prior pipeline.
8. CONGENITAL_DEVELOPMENTAL, HEPATIC, and RENAL fall below the pre-specified
   unseen-n cutoff of 40. They are not in the primary mean.
9. Endpoint-level tests are few. A Wilcoxon p-value pretends endpoints are
   independent. They share compounds. The bootstrap over endpoints is the
   uncertainty that matches the claim, and it is still a bootstrap of a
   dozen numbers.
10. Reduced-run ridge does not tune alpha. That avoids test tuning and leaves
    open the possibility that another fixed alpha would move the point
    estimate. The full nested search was not run.
11. Tree models in the reduced run are not cross-validated inside the adult
    domain. They are a transfer robustness check at pre-specified sizes.
12. CORAL and importance weighting see unlabeled pediatric chemistry. They are
    not inductive.
13. The oracle gate uses labels. It is an upper bound, not a method.
14. Tox21-to-FAERS was not given a performance number. The historical probes
    are endpoint-mismatched.
15. Nothing here is a clinical decision tool.
