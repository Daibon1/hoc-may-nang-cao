from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class StreamlitTests(unittest.TestCase):
    def test_demo_to_predictions_and_charts(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        app.checkbox[0].set_value(True).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.success), 1)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.metric[0].value, '100')
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.get('vega_lite_chart')), 2)
        result, _ = app.session_state['scoring_result']
        self.assertEqual(len(result), 100)
        self.assertTrue(result.risk_score.between(0, 100).all())
        self.assertTrue(((result.prediction == 'Attack') == (result.alert_level == 'High')).all())
        app.multiselect[0].set_value(['High']).run()
        self.assertEqual(len(app.exception), 0)


if __name__ == '__main__':
    unittest.main()
