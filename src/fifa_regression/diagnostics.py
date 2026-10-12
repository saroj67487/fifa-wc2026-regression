"""OLS inference and regression diagnostics (implemented with NumPy / SciPy).

These use the same formulas as statsmodels' ``OLS().summary()``, without the
extra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class OLSResult:
    coefficients: pd.DataFrame   # coef, std_err, t, p_value, ci_low, ci_high
    n: int
    k: int
    r2: float
    adj_r2: float
    f_stat: float
    f_pvalue: float
    residuals: np.ndarray
    fitted: np.ndarray

    def fit_summary(self) -> dict:
        return {"n": self.n, "k": self.k, "r2": self.r2, "adj_r2": self.adj_r2,
                "F": self.f_stat, "F_p": self.f_pvalue}


def fit_ols(X, y, feature_names) -> OLSResult:
    """Ordinary least squares with an intercept, plus t-tests and an F-test."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    design = np.column_stack([np.ones(n), X])

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    resid = y - fitted
    df_resid = n - k - 1

    sigma2 = resid @ resid / df_resid
    std_err = np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
    t_stat = beta / std_err
    p_val = 2 * stats.t.sf(np.abs(t_stat), df_resid)
    margin = stats.t.ppf(0.975, df_resid) * std_err

    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid
    f_stat = (r2 / k) / ((1 - r2) / df_resid)

    table = pd.DataFrame(
        {"coef": beta, "std_err": std_err, "t": t_stat, "p_value": p_val,
         "ci_low": beta - margin, "ci_high": beta + margin},
        index=["const", *feature_names],
    )
    return OLSResult(table, n, k, float(r2), float(adj_r2), float(f_stat),
                     float(stats.f.sf(f_stat, k, df_resid)), resid, fitted)


def variance_inflation_factors(X, feature_names) -> pd.Series:
    """VIF_j = 1 / (1 - R^2_j), regressing each variable on all the others."""
    X = np.asarray(X, dtype=float)
    vifs = {}
    for j, name in enumerate(feature_names):
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j])
        vifs[name] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(vifs, name="VIF")


def breusch_pagan(residuals, X) -> tuple[float, float]:
    """Breusch-Pagan LM test. H0: residual variance is constant."""
    e2 = np.asarray(residuals) ** 2
    X = np.asarray(X, dtype=float)
    r2 = LinearRegression().fit(X, e2).score(X, e2)
    lm = len(e2) * r2
    return float(lm), float(stats.chi2.sf(lm, X.shape[1]))


def durbin_watson(residuals) -> float:
    """Durbin-Watson statistic: ~2 means no autocorrelation."""
    r = np.asarray(residuals)
    return float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))


def residual_diagnostics(residuals, X, feature_names) -> dict:
    """All assumption checks used in the report, in one dictionary."""
    lm, bp_p = breusch_pagan(residuals, X)
    max_vif = (float(variance_inflation_factors(X, feature_names).max())
               if len(feature_names) > 1 else 1.0)
    return {
        "shapiro_resid_p": float(stats.shapiro(residuals).pvalue),
        "resid_skew": float(stats.skew(residuals)),
        "breusch_pagan_LM": lm,
        "breusch_pagan_p": bp_p,
        "durbin_watson": durbin_watson(residuals),
        "max_vif_model": max_vif,
    }
