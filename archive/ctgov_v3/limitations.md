# Limitations

1. Clinical trial AE modules vary in coding quality; SOC mapping may miss terms.
2. Single-drug arm restriction excludes many oncology combinations (informative but out of scope).
3. Smoothed incidence is not a causal drug effect estimate; confounding by indication remains.
4. FAERS ROR reflects reporting behavior, not incidence or confirmed causality; used only as external PV concordance.
5. EudraVigilance export was not available on this box; EV validation pending.
6. `chembl_target_profile` coverage is sparse for some structures; missingness indicators used.
7. Chemical Checker / BioFeat remotes were intentionally not called (RemoteDisconnected risk); no validated CC caches present.
8. Sample sizes for some organ-class endpoints may be below eligibility thresholds — see endpoint_feasibility.csv.
9. openFDA rate limits constrain FAERS sample size without an API key.
10. Zero-shot pediatric evaluation does not adapt to pediatric pharmacology (PK/PD differences).
