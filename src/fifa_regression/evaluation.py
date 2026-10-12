"""Evaluation metrics and the repeated-split robustness check."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from fifa_regression.config import TEST_SIZE, ModelConfig
from fifa_regression.preprocessing import FeaturePreprocessor, target_transform


def regression_metrics(y_true, y_pred, n_features: int) -> dict:
    """MAE, MSE, RMSE, normalised RMSE (by range), R^2 and adjusted R^2."""
    y_true = np.asarray(y_true, dtype=float)
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n_features > 0 else r2
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": rmse,
        "NRMSE": rmse / float(y_true.max() - y_true.min()),
        "R2": float(r2),
        "Adj_R2": float(adj),
    }


def compare_models(models, y_train, y_test) -> pd.DataFrame:
    """One row per model: test metrics plus training R^2 / RMSE."""
    rows = []
    for name, m in models.items():
        row = {"model": name, "n_features": m.n_features,
               **regression_metrics(y_test, m.test_pred, m.n_features)}
        row["Train_R2"] = float(r2_score(y_train, m.train_pred))
        row["Train_RMSE"] = float(np.sqrt(mean_squared_error(y_train, m.train_pred)))
        rows.append(row)
    return pd.DataFrame(rows)


def repeated_split_r2(df: pd.DataFrame, cfg: ModelConfig, reduced_features,
                      n_splits: int, test_size: float = TEST_SIZE) -> pd.DataFrame:
    """Test R^2 of models A, B and C over ``n_splits`` random 60/40 splits.

    The whole pipeline (transformations + scaling) is re-fitted on each training
    split, so every split is evaluated exactly like the reported one.
    """
    X, y = df[list(cfg.features)], df[cfg.target].astype(float)
    forward, inverse = target_transform(cfg.target_transform)
    scores = {"A: raw": [], "B: transformed+std": [], "C: reduced": []}

    for seed in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        prep = FeaturePreprocessor(cfg.features, cfg.indicator_vars, cfg.log_vars).fit(X_tr)
        Xt_tr, Xt_te = prep.transform(X_tr), prep.transform(X_te)

        a = LinearRegression().fit(X_tr, y_tr)
        scores["A: raw"].append(r2_score(y_te, a.predict(X_te)))
        b = LinearRegression().fit(Xt_tr, forward(y_tr))
        scores["B: transformed+std"].append(r2_score(y_te, inverse(b.predict(Xt_te))))
        c = LinearRegression().fit(Xt_tr[reduced_features], forward(y_tr))
        scores["C: reduced"].append(r2_score(y_te, inverse(c.predict(Xt_te[reduced_features]))))

    return pd.DataFrame(scores)


def summarise_robustness(scores: pd.DataFrame) -> dict:
    return {col: {"mean": float(scores[col].mean()), "sd": float(scores[col].std()),
                  "p5": float(scores[col].quantile(0.05)),
                  "p95": float(scores[col].quantile(0.95))}
            for col in scores}
