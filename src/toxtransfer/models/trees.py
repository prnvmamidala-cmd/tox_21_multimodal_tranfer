"""Tree models with train-only imputation and no scaling."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from toxtransfer.features.preprocess import fit_preprocess, transform


def fit_predict_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    continuous_mask: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int = 0,
) -> np.ndarray:
    state = fit_preprocess(X_train, continuous_mask, clip=True, scale=False)
    Z_train = transform(X_train, state, scale=False)
    Z_test = transform(X_test, state, scale=False)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(Z_train, y_train)
    return model.predict(Z_test)


def fit_predict_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    continuous_mask: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int = 0,
) -> np.ndarray:
    from xgboost import XGBRegressor

    state = fit_preprocess(X_train, continuous_mask, clip=True, scale=False)
    Z_train = transform(X_train, state, scale=False)
    Z_test = transform(X_test, state, scale=False)
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(Z_train, y_train)
    return model.predict(Z_test)
