import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from nsl_kdd.dashboard import parse_csv, FEATURES, load_bundle, score_frame


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / 'data/raw/KDDTest+.txt', header=None, nrows=12)
        cls.frame = cls.raw.copy()
        cls.frame.columns = [*FEATURES, 'label', 'difficulty']

    def test_headerless_keeps_first_row_and_accepts_41_or_43(self):
        for count in [41, 43]:
            parsed, mode = parse_csv(self.raw.iloc[:, :count].to_csv(index=False, header=False).encode())
            self.assertEqual(len(parsed), 12)
            self.assertEqual(len(parsed.columns), count)
            self.assertEqual(mode, 'Không header')

    def test_named_columns_are_reordered_for_inference_and_bom_supported(self):
        frame = self.frame[list(reversed(self.frame.columns))]
        parsed, mode = parse_csv(frame.to_csv(index=False).encode('utf-8-sig'))
        self.assertEqual(mode, 'Có header')
        np.testing.assert_array_equal(parsed.duration, self.frame.duration)

    def test_rejects_schema_errors(self):
        invalid = [b'', b'1,2,3\n', self.frame.drop(columns='service').to_csv(index=False).encode(),
                   (','.join(FEATURES) + '\n').encode(), b'"unterminated',
                   self.frame.rename(columns={'duration': 'service'}).to_csv(index=False).encode()]
        for content in invalid:
            with self.subTest(content=content[:40]), self.assertRaises(ValueError):
                parse_csv(content)

    def test_rejects_bad_values_and_inconsistent_rows(self):
        for value in ['', 'not_numeric', 'NaN', 'Inf', '-1', '1.5']:
            frame = self.frame.astype(object)
            frame.loc[0, 'duration'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_csv(frame.to_csv(index=False).encode())
        frame = self.frame.copy()
        frame.loc[0, 'serror_rate'] = 2
        with self.assertRaises(ValueError):
            parse_csv(frame.to_csv(index=False).encode())
        with self.assertRaises(ValueError):
            parse_csv(self.raw.to_csv(index=False, header=False).encode() + b'1,2\n')

    def test_saved_bundle_matches_previous_test_scores(self):
        frame, _ = parse_csv(self.frame.to_csv(index=False).encode())
        result, _ = score_frame(frame, load_bundle(ROOT / 'artifacts/models/autoencoder_optimized'))
        expected = pd.read_csv(ROOT / 'reports/risk_optimization/test_risk_predictions.csv', nrows=len(frame))
        np.testing.assert_allclose(result.reconstruction_error, expected.reconstruction_error, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result.risk_score, expected.risk_score)
        np.testing.assert_array_equal(result.alert_level, expected.alert_level)
        self.assertTrue(result.risk_score.between(0, 100).all())


if __name__ == '__main__':
    unittest.main()
