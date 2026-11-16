import unittest
import pandas as pd
from depth_accuracy import error_profile
from export_forecast_replay import scenario_frames


class DepthAccuracyTests(unittest.TestCase):
    def test_replay_keeps_missing_slots_and_never_exports_future_labels(self):
        frame = pd.DataFrame({'deployment_id': ['station'], 'depth_lag_0_cm': [7.],
                              'future_secret': [99.]}, index=pd.to_datetime(['2023-04-23T00:05Z']))
        frames = scenario_frames(frame, [9.], 'station', '2023-04-23')
        self.assertEqual(len(frames), 288)
        self.assertIsNone(frames[0]['predicted_depth_cm'])
        self.assertEqual(frames[1]['predicted_depth_cm'], 9.)
        self.assertEqual(frames[1]['baseline_depth_cm'], 7.)
        self.assertIsNone(frames[2]['predicted_depth_cm'])
        self.assertNotIn('future_secret', frames[1])

    def test_error_and_bias_are_not_threshold_accuracy(self):
        profile = error_profile([9.8, 12.], [10.1, 11.7])
        self.assertAlmostEqual(profile['mae_cm'], .3)
        self.assertAlmostEqual(profile['bias_cm'], 0.)

    def test_no_fabricated_error_for_empty_group(self):
        self.assertIsNone(error_profile([], [])['mae_cm'])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            error_profile([10], [float('nan')])


if __name__ == '__main__':
    unittest.main()
