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

| UTC time | Event |
| --- | --- |
| 2026-10-19T15:19:19Z | Started `python scripts/reproduce_all.py --config configs/reduced.yaml` with `OMP_NUM_THREADS=4`. Features were already cached. |
| 2026-10-19T15:29:45Z | Process exited 0. `results/run_manifest.json` records git commit `a9324cf9bcffd5d406cb21b5e70c16f60382d017`, config sha256 `f0226a96caad9ecd195eec70abd44ded8398f6e8fcc33e9714df0238407e60f1`, and that timestamp. Wall clock from process start to the manifest timestamp was 626 seconds. That interval includes 15 trial endpoints and the Tox21 logistic arm. It does not include the earlier feature build (about 100 seconds, cache hit on this run). |
| 2026-10-19 | Prediction file `results/predictions/predictions.csv.gz` sha256 `37dbf42edac39337d9e76aeb446661dd7e3a0e16429d453cfd4d93abf68faa0d`. Tox21 predictions sha256 `6f4b1107c0ccd1810d253869e92f4b5e13ca07f8cfd45ad46dfed4c35dd35736`. |
| 2026-10-19 | Summary tables and `reports/results.md` were regenerated from those predictions after the fit, so the manifest commit is the fit commit rather than the commit that added the exploratory interval text. Primary estimates were not refit. |
| 2026-10-19 | `scripts/probe_signaturizer.py` wrote `reports/signaturizer_probe.json`: status blocked, `ModuleNotFoundError` for `signaturizer`. No signature matrix was created. |
