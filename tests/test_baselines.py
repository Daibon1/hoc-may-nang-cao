import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from nsl_kdd.baselines import anomaly_scores, choose_threshold


class BaselineTests(unittest.TestCase):
    def test_anomaly_score_orientation_and_positive_class(self):
        class Model:
            classes_ = np.array([1, 0])

            def predict_proba(self, values):
                return np.array([[0.8, 0.2], [0.1, 0.9]])

            def score_samples(self, values):
                return np.array([-0.7, -0.4])

            def decision_function(self, values):
                return np.array([-3, 2])
        model = Model()
        np.testing.assert_array_equal(anomaly_scores(model, None, 'Random Forest'), [0.8, 0.1])
        np.testing.assert_array_equal(anomaly_scores(model, None, 'Isolation Forest'), [0.7, 0.4])
        np.testing.assert_array_equal(anomaly_scores(model, None, 'One-Class SVM'), [3, -2])

    def test_negative_scores_and_feasible_threshold(self):
        selected, rows, feasible = choose_threshold(np.linspace(-10, -1, 1000), [-8, -5, 0, 1], [0, 0, 1, 1])
        self.assertTrue(feasible)
        self.assertLess(selected['threshold'], 0)
        self.assertEqual(selected['macro_f1'], 1)
        self.assertEqual(len(rows), 5)

    def test_ties_report_budget_violation(self):
        selected, _, feasible = choose_threshold(np.zeros(10), [0, 0, 1, 1], [0, 0, 1, 1])
        self.assertFalse(feasible)
        self.assertEqual(selected['false_positive_rate'], 1)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            choose_threshold([np.nan], [0, 1], [0, 1])
        with self.assertRaises(ValueError):
            choose_threshold([0], [0, 1], [1, 1])


if __name__ == '__main__':
    unittest.main()
