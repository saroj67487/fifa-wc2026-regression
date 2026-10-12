import unittest

import numpy as np
import pandas as pd
from scipy import stats

from fifa_regression.preprocessing import FeaturePreprocessor, target_transform


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "normal": rng.normal(0, 1, n),
        "skewed": rng.exponential(2.0, n),          # right-skewed, >= 0
        "money": rng.lognormal(5, 1, n),            # strictly positive
        "flag": rng.integers(0, 2, n),              # binary
        "few_levels": rng.integers(-1, 4, n),       # 5 levels
    })


class FeaturePreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.X = _frame()
        self.prep = FeaturePreprocessor(self.X.columns, indicator_vars=["flag"],
                                        log_vars=["money"]).fit(self.X)

    def test_output_is_standardised(self):
        Z = self.prep.transform(self.X)
        np.testing.assert_allclose(Z.mean(), 0, atol=1e-10)
        np.testing.assert_allclose(Z.std(ddof=0), 1, atol=1e-10)

    def test_only_continuous_skewed_columns_get_yeo_johnson(self):
        self.assertEqual(self.prep.yeo_johnson_vars_, ["skewed"])

    def test_yeo_johnson_reduces_skew(self):
        after = self.prep.transform_unscaled(self.X)["skewed"]
        self.assertLess(abs(stats.skew(after)), abs(stats.skew(self.X["skewed"])))

    def test_test_set_uses_training_parameters(self):
        train, test = self.X.iloc[:150], self.X.iloc[150:]
        prep = FeaturePreprocessor(self.X.columns, ["flag"], ["money"]).fit(train)
        np.testing.assert_allclose(prep.scaler_.mean_,
                                   prep.transform_unscaled(train).mean().values)
        self.assertEqual(len(prep.transform(test)), 50)

    def test_log_rejects_non_positive_values(self):
        X = self.X.copy()
        X.loc[0, "money"] = 0
        with self.assertRaises(ValueError):
            FeaturePreprocessor(X.columns, log_vars=["money"]).fit(X)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            FeaturePreprocessor(self.X.columns).transform(self.X)


class TargetTransformTest(unittest.TestCase):
    def test_round_trip(self):
        y = np.array([0.0, 1, 2, 7])
        for name in (None, "sqrt", "log1p"):
            fwd, inv = target_transform(name)
            np.testing.assert_allclose(inv(fwd(y)), y)

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            target_transform("cube")


if __name__ == "__main__":
    unittest.main()
