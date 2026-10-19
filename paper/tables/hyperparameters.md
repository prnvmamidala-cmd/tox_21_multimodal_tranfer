| item | value | role |
| --- | --- | --- |
| ridge_alpha | 10.0 | prespecified confirmatory |
| ridge_alpha_mode | prespecified | not tuned on the transfer test |
| ridge_alpha_grid_full_only | [0.1, 1.0, 10.0, 100.0] | full.yaml only |
| rf_n_estimators | 40 | robustness |
| rf_max_depth | 12 | robustness |
| rf_min_samples_leaf | 5 | robustness |
| rf_adult_oof | False | reduced run |
| xgb_n_estimators | 40 | robustness |
| xgb_max_depth | 3 | robustness |
| xgb_learning_rate | 0.1 | robustness |
| bootstrap_n | 1000 | endpoint percentile interval |
| bootstrap_seed | 0 | endpoint percentile interval |
| permutation_seeds | [101, 202, 303, 404, 505] | training-row biology permutation |
| pca_components | 32 | CORAL, transductive |
| min_adult_n | 200 | eligibility, frozen |
| min_unseen_n | 40 | eligibility, frozen |
