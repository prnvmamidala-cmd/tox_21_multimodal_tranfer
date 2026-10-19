# Experiment log

| UTC time | Event |
| --- | --- |
| 2026-10-19 | Repository initialized from an empty public GitHub repo. Handoff archive audited. Confirmatory fits had not been started when the analysis plan was written. |
| 2026-10-19 | Descriptive counts measured while building features, before model fitting: 3057 structures featurized, 0 standardization failures, 12 InChIKeys with conflicting SMILES (mode retained), logit recomputation matches the stored column to 1e-15, 2856 adult-trial and 835 pediatric-trial structures, 634 shared InChIKeys, 201 pediatric-only structures, 48 of those 201 share a Bemis–Murcko scaffold with an adult structure, ChEMBL profile has 15 targets and mean missingness 0.924. |

Confirmatory command, not yet executed at the time of the plan commit:

```bash
python scripts/reproduce_all.py --config configs/reduced.yaml
```

Smoke command:

```bash
python scripts/reproduce_all.py --config configs/smoke.yaml
```

Full command, specified and not executed:

```bash
python scripts/reproduce_all.py --config configs/full.yaml
```
