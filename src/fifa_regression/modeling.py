"""Model definitions: null baseline, raw baseline (A), full (B) and reduced (C)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from fifa_regression.config import P_VALUE_THRESHOLD
from fifa_regression.diagnostics import fit_ols

logger = logging.getLogger(__name__)

MODEL_NULL = "Null (training mean)"
MODEL_A = "A: raw variables"
MODEL_B = "B: transformed + standardised"
MODEL_C = "C: reduced (p <= 0.05)"


@dataclass
class FittedModel:
    name: str
    features: list
    test_pred: np.ndarray        # predictions on the ORIGINAL target scale
    train_pred: np.ndarray       # predictions on the ORIGINAL target scale
    estimator: LinearRegression | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.features)


def backward_elimination(X: pd.DataFrame, y, threshold: float = P_VALUE_THRESHOLD):
    """Drop the least significant variable until all p-values <= threshold.

    Returns:
        (kept_features, removed) where removed is a list of (feature, p_value)
        in the order they were dropped.
    """
    kept = list(X.columns)
    removed = []
    while len(kept) > 1:
        pvals = fit_ols(X[kept], y, kept).coefficients["p_value"].drop("const")
        worst = pvals.idxmax()
        if pvals[worst] <= threshold:
            break
        removed.append((worst, float(pvals[worst])))
        kept.remove(worst)
        logger.debug("removed %s (p = %.3f)", worst, pvals[worst])
    return kept, removed


def fit_all_models(X_train_raw, X_test_raw, X_train, X_test, y_train,
                   y_train_t, forward, inverse) -> dict[str, FittedModel]:
    """Fit the four models compared in the report.

    Args:
        X_train_raw, X_test_raw: untransformed explanatory variables.
        X_train, X_test: transformed + standardised explanatory variables.
        y_train: target on the original scale.
        y_train_t: target after the target transformation (may equal y_train).
        forward, inverse: target transformation functions.
    """
    models: dict[str, FittedModel] = {}

    mean = float(np.mean(y_train))
    models[MODEL_NULL] = FittedModel(
        MODEL_NULL, [], np.full(len(X_test), mean), np.full(len(X_train), mean))

    lr_a = LinearRegression().fit(X_train_raw, y_train)
    models[MODEL_A] = FittedModel(MODEL_A, list(X_train_raw.columns),
                                  lr_a.predict(X_test_raw), lr_a.predict(X_train_raw), lr_a)

    lr_b = LinearRegression().fit(X_train, y_train_t)
    models[MODEL_B] = FittedModel(MODEL_B, list(X_train.columns),
                                  inverse(lr_b.predict(X_test)), inverse(lr_b.predict(X_train)), lr_b)

    kept, removed = backward_elimination(X_train, y_train_t)
    lr_c = LinearRegression().fit(X_train[kept], y_train_t)
    models[MODEL_C] = FittedModel(MODEL_C, kept, inverse(lr_c.predict(X_test[kept])),
                                  inverse(lr_c.predict(X_train[kept])), lr_c,
                                  meta={"eliminated": removed})
    logger.info("Backward elimination removed %s -> kept %s",
                [r[0] for r in removed], kept)
    return models
