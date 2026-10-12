"""Integration tests on the real datasets in data/raw/."""

import tempfile
import unittest
from pathlib import Path

from fifa_regression.config import DATA_DIR, OBJECTIVE_2_1, OBJECTIVE_2_2, SHARED_CONCEPTS
from fifa_regression.data import check_consistency, check_shared_variables, load_dataset
from fifa_regression.pipeline import run_analysis


class DatasetRequirementsTest(unittest.TestCase):
    def test_row_counts(self):
        self.assertEqual(len(load_dataset(OBJECTIVE_2_1, DATA_DIR)), 104)
        self.assertEqual(len(load_dataset(OBJECTIVE_2_2, DATA_DIR)), 208)

    def test_exactly_eight_features_each(self):
        self.assertEqual(len(OBJECTIVE_2_1.features), 8)
        self.assertEqual(len(OBJECTIVE_2_2.features), 8)

    def test_at_most_four_shared_concepts(self):
        self.assertLessEqual(check_shared_variables(), 4)
        for col_21, col_22 in SHARED_CONCEPTS.values():
            self.assertIn(col_21, OBJECTIVE_2_1.features)
            self.assertIn(col_22, OBJECTIVE_2_2.features)

    def test_datasets_agree_on_scores(self):
        self.assertEqual(check_consistency(load_dataset(OBJECTIVE_2_1, DATA_DIR),
                                           load_dataset(OBJECTIVE_2_2, DATA_DIR)), 0)


class PipelineSmokeTest(unittest.TestCase):
    """Runs the full pipeline quickly (few splits, no plots) and checks key outputs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.r21 = run_analysis(OBJECTIVE_2_1, DATA_DIR, cls.out, n_splits=5, make_plots=False)
        cls.r22 = run_analysis(OBJECTIVE_2_2, DATA_DIR, cls.out, n_splits=5, make_plots=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sixty_forty_split(self):
        self.assertEqual((self.r21["n_train"], self.r21["n_test"]), (62, 42))
        self.assertEqual((self.r22["n_train"], self.r22["n_test"]), (124, 84))

    def test_reported_numbers_are_reproduced(self):
        c21 = {r["model"]: r for r in self.r21["comparison"]}
        c22 = {r["model"]: r for r in self.r22["comparison"]}
        self.assertAlmostEqual(c21["C: reduced (p <= 0.05)"]["R2"], 0.1274, places=3)
        self.assertAlmostEqual(c22["C: reduced (p <= 0.05)"]["RMSE"], 1.3827, places=3)
        self.assertEqual(self.r22["model_c_features"],
                         ["team_squad_value_eur_m", "opponent_draw_pot"])

    def test_models_beat_null_baseline(self):
        for res in (self.r21, self.r22):
            rows = {r["model"]: r for r in res["comparison"]}
            self.assertLess(rows["C: reduced (p <= 0.05)"]["RMSE"],
                            rows["Null (training mean)"]["RMSE"])

    def test_result_files_written(self):
        for name in ("2.1_model_comparison.csv", "2.2_vif.csv", "2.1_modelC_coefficients.csv"):
            self.assertTrue((self.out / "results" / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
