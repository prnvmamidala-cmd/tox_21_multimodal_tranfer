# V5 results

Generated from `results/v5/v5_claims.json`. Do not edit the numbers by hand.

Rebuilt structures: 3057.
Adult structures in the rebuilt matrix: 2856.
Structure-unseen structures with a rebuilt molecule: 201.
Rebuilt ChEMBL mean missingness: 0.9242.
Late-stack weight on biology: 0.7500.
Missingness gate intercept and slope: 0.7500, 1.0000.

## Verdicts

- `late_stack`: **null**. Minus chemistry 0.0082 (95% CI -0.0014 to 0.0200, q=0.2448). Minus permutation mean 0.0087 (95% CI -0.0010 to 0.0215, q=0.2448).
- `miss_gate`: **null**. Minus chemistry 0.0021 (95% CI -0.0124 to 0.0200, q=0.9796). Minus permutation mean 0.0017 (95% CI -0.0153 to 0.0201, q=0.9900).
- `imputed_early`: **null**. Minus chemistry -0.0016 (95% CI -0.0035 to 0.0001, q=0.2448). Minus permutation mean -0.0017 (95% CI -0.0035 to 0.0000, q=0.2448).
- `mechanism`: **null**. Minus chemistry 0.0444 (95% CI -0.0391 to 0.1170, q=0.4680). Minus permutation mean 0.0151 (95% CI -0.0385 to 0.0625, q=0.7520).
- `mlp_early`: **null**. Minus chemistry -0.0167 (95% CI -0.0401 to 0.0079, q=0.3943). Minus permutation mean -0.0013 (95% CI -0.0140 to 0.0092, q=0.9432).
- `tox21_late`: **null**. Minus chemistry 0.0543 (95% CI -0.0291 to 0.1333, q=0.3943). Minus permutation mean 0.0651 (95% CI -0.0032 to 0.1363, q=0.2448).

## Blocked

- `signaturizer`: blocked: ModuleNotFoundError: No module named 'signaturizer'
- `chemprop`: blocked: ModuleNotFoundError
- `pyg`: blocked: ModuleNotFoundError
- `pretrained_embeddings`: blocked: no pretrained embedding weights with a checked license were downloaded

## Diagnostics

Adult out-of-fold Spearman of a biology model predicting the frozen chemistry residual, macro mean -0.0143.
Scaffold out-of-fold Spearman of rebuilt chemistry predicting each ChEMBL target, mean across 15 targets 0.2815.
Exploratory early ridge, zero-filled biology macro Spearman -0.0142; train-median fill -0.0081.
Tox21 scaffold logistic Morgan macro AUROC 0.6886 on 12 tasks. This is a competence check, not a transfer result.

Exploratory zero-fill versus train-median biology is in `v5_claims.json` under `diagnostics.zero_vs_median`.

