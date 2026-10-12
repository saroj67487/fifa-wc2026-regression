"""Transformation and standardisation of explanatory variables and the target.

All parameters are learned from the TRAINING split only and then applied to the
test split, so no test information leaks into the model.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, StandardScaler

from fifa_regression.config import MAX_DISCRETE_LEVELS, SKEW_THRESHOLD


class FeaturePreprocessor:
    """Log / Yeo-Johnson transform skewed variables, then z-score every variable.

    Rules (decided on the training data):
        1. Variables in ``log_vars`` get the natural log (strictly positive,
           diminishing-returns relationships such as money).
        2. Remaining continuous variables with |skewness| > ``skew_threshold``
           get a Yeo-Johnson power transform (handles zero and negative values).
        3. Indicator variables and variables with <= ``max_discrete_levels``
           distinct values are not power-transformed - a monotone transform
           cannot make a variable with so few levels normal.
        4. Every variable is standardised: z = (x - mean) / sd.
    """

    def __init__(self, features, indicator_vars=(), log_vars=(),
                 skew_threshold: float = SKEW_THRESHOLD,
                 max_discrete_levels: int = MAX_DISCRETE_LEVELS):
        self.features = list(features)
        self.indicator_vars = set(indicator_vars)
        self.log_vars = list(log_vars)
        self.skew_threshold = skew_threshold
        self.max_discrete_levels = max_discrete_levels
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        nonpositive = [c for c in self.log_vars if (X[c] <= 0).any()]
        if nonpositive:
            raise ValueError(f"log transform needs strictly positive values: {nonpositive}")
        X = self._apply_log(X[self.features])

        self.skew_before_ = X.apply(stats.skew)
        self.discrete_vars_ = {c for c in self.features if X[c].nunique() <= self.max_discrete_levels}
        self.yeo_johnson_vars_ = [
            c for c in self.features
            if c not in self.indicator_vars
            and c not in self.log_vars
            and c not in self.discrete_vars_
            and abs(self.skew_before_[c]) > self.skew_threshold
        ]
        self.power_transformer_ = PowerTransformer(method="yeo-johnson", standardize=False)
        if self.yeo_johnson_vars_:
            X[self.yeo_johnson_vars_] = self.power_transformer_.fit_transform(X[self.yeo_johnson_vars_])
        self.scaler_ = StandardScaler().fit(X)
        self._fitted = True
        return self

    # ------------------------------------------------------------ transform
    def transform_unscaled(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log / Yeo-Johnson only (useful for before/after plots)."""
        self._check_fitted()
        X = self._apply_log(X[self.features])
        if self.yeo_johnson_vars_:
            X[self.yeo_johnson_vars_] = self.power_transformer_.transform(X[self.yeo_johnson_vars_])
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Full transformation: log / Yeo-Johnson followed by z-score."""
        Xt = self.transform_unscaled(X)
        return pd.DataFrame(self.scaler_.transform(Xt), columns=self.features, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # --------------------------------------------------------------- report
    def treatment(self, column: str) -> str:
        """Plain-English description of what happened to one variable."""
        self._check_fitted()
        if column in self.log_vars:
            base = "log"
        elif column in self.yeo_johnson_vars_:
            base = "Yeo-Johnson"
        elif column in self.indicator_vars:
            base = "none (binary/indicator)"
        elif column in self.discrete_vars_:
            base = f"none (discrete, <= {self.max_discrete_levels} levels)"
        else:
            base = f"none (|skew| <= {self.skew_threshold})"
        return f"{base} + z-score"

    @property
    def yeo_johnson_lambdas(self) -> dict:
        self._check_fitted()
        if not self.yeo_johnson_vars_:
            return {}
        return dict(zip(self.yeo_johnson_vars_, map(float, self.power_transformer_.lambdas_)))

    # -------------------------------------------------------------- helpers
    def _apply_log(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.log_vars:
            X[c] = np.log(X[c])
        return X

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before use")


# ------------------------------------------------------------------ target
_TARGET_TRANSFORMS: dict[Optional[str], tuple[Callable, Callable]] = {
    None: (lambda v: v, lambda v: v),
    "sqrt": (np.sqrt, np.square),
    "log1p": (np.log1p, np.expm1),
}


def target_transform(name: Optional[str]) -> tuple[Callable, Callable]:
    """Return (forward, inverse) functions for a target transformation."""
    try:
        return _TARGET_TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown target transform {name!r}; "
                         f"choose from {list(_TARGET_TRANSFORMS)}") from exc
