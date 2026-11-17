import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from nsl_kdd.autoencoder import Autoencoder
from nsl_kdd.risk import risk_scores, alert_levels


class RiskTests(unittest.TestCase):
    def test_percentile_ties_bounds_and_monotonicity(self):
        scores = risk_scores([0, 1, 2, 3, 100], [1, 2, 2, 3])
        np.testing.assert_array_equal(scores, [0, 25, 75, 100, 100])
        self.assertTrue((np.diff(scores) >= 0).all())

    def test_alert_boundaries_and_binary_consistency(self):
        np.testing.assert_array_equal(alert_levels([0, 2, 3, 4], 2, 4),
                                       ['Low', 'Medium', 'Medium', 'High'])

    def test_invalid_errors_and_reference(self):
        for reference in [[], [np.nan], [-1]]:
            with self.assertRaises(ValueError):
                risk_scores([0], reference)
        with self.assertRaises(ValueError):
            alert_levels([0], 2, 2)
        with self.assertRaises(ValueError):
            risk_scores([np.inf], [0, 1])

    def test_latent_shapes_and_checkpoint_roundtrip(self):
        for dim in [8, 16, 32]:
            model = Autoencoder(77, dim)
            values = torch.zeros((3, 77))
            self.assertEqual(model.encoder(values).shape, (3, dim))
            restored = Autoencoder(77, dim)
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(values), restored(values))


if __name__ == '__main__':
    unittest.main()
