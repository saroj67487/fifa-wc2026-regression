import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from fifa_regression.diagnostics import (
    breusch_pagan,
    durbin_watson,
    fit_ols,
    variance_inflation_factors,
)
from fifa_regression.evaluation import regression_metrics


class OLSTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(300, 3))
        self.y = 1.5 + self.X @ np.array([2.0, -1.0, 0.0]) + rng.normal(0, 0.5, 300)
        self.res = fit_ols(self.X, self.y, ["a", "b", "c"])

    def test_coefficients_match_sklearn(self):
        sk = LinearRegression().fit(self.X, self.y)
        np.testing.assert_allclose(self.res.coefficients["coef"].iloc[1:], sk.coef_)
        self.assertAlmostEqual(self.res.coefficients.loc["const", "coef"], sk.intercept_)

    def test_significance(self):
        p = self.res.coefficients["p_value"]
        self.assertLess(p["a"], 0.001)
        self.assertLess(p["b"], 0.001)
        self.assertGreater(p["c"], 0.01)   # true coefficient is zero

    def test_r2_bounds(self):
        self.assertTrue(0 < self.res.adj_r2 < self.res.r2 < 1)


class DiagnosticsTest(unittest.TestCase):
    def test_vif_detects_collinearity(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=500)
        X = np.column_stack([a, a + rng.normal(0, 0.05, 500), rng.normal(size=500)])
        vif = variance_inflation_factors(X, ["a", "a_copy", "indep"])
        self.assertGreater(vif["a"], 10)
        self.assertLess(vif["indep"], 1.5)

    def test_durbin_watson_white_noise_near_two(self):
        e = np.random.default_rng(1).normal(size=2000)
        self.assertAlmostEqual(durbin_watson(e), 2.0, delta=0.15)

    def test_breusch_pagan_flags_heteroscedasticity(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(1, 10, 500)
        _, p_hetero = breusch_pagan(rng.normal(0, x), x.reshape(-1, 1))
        self.assertLess(p_hetero, 0.01)


class MetricsTest(unittest.TestCase):
    def test_perfect_prediction(self):
        y = np.array([0.0, 1, 2, 3])
        m = regression_metrics(y, y, n_features=1)
        self.assertEqual(m["RMSE"], 0)
        self.assertEqual(m["R2"], 1)

    def test_nrmse_uses_range(self):
        y = np.array([0.0, 4.0])
        m = regression_metrics(y, np.array([1.0, 3.0]), n_features=0)
        self.assertAlmostEqual(m["NRMSE"], 1.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
