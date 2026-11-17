"""Tests for score definition, validation threshold and checkpoint inference."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch
from nsl_kdd.autoencoder import (Autoencoder, predict_from_errors,
                                 reconstruction_error, select_threshold)


class AutoencoderTests(unittest.TestCase):
    def test_mse_definition_and_batching(self):
        model = Autoencoder(3)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        values = np.array([[1, 2, 3], [0, 0, 0], [3, 3, 3]], dtype=np.float32)
        np.testing.assert_allclose(reconstruction_error(model, values, batch_size=2),
                                   [14 / 3, 0, 9])

    def test_calibration_and_threshold_boundary(self):
        threshold = select_threshold(np.arange(100, dtype=float), 0.99)
        self.assertAlmostEqual(threshold, 98.01)
        np.testing.assert_array_equal(predict_from_errors([0, threshold, 100], threshold), [0, 1, 1])
        for invalid in [[], [np.nan], [-1]]:
            with self.assertRaises(ValueError):
                select_threshold(invalid)

    def test_linear_decoder_can_output_negative_values(self):
        model = Autoencoder(3)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        with torch.no_grad():
            model.decoder[-1].bias.fill_(-2)
        self.assertTrue(torch.all(model(torch.zeros((2, 3))) == -2))


if __name__ == '__main__':
    unittest.main()
