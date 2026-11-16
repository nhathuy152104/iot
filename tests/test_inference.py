import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
sys.path.insert(0, str(ROOT / 'tools'))
from predict_depth import build_features, predict, read_input
from train_from_download import prepare_station, month_key


def observations(values, times=None):
    times = pd.date_range('2026-01-01T10:00Z', periods=len(values), freq='5min') if times is None else pd.to_datetime(times, utc=True)
    return pd.DataFrame({'depth_raw_mm': values}, index=times)


class InferenceTests(unittest.TestCase):
    def test_training_prep_uses_future_proc_as_label_not_feature(self):
        frame = pd.DataFrame({'depth_raw_mm': [10.] * 8, 'depth_proc_mm': [20.] * 8},
                             index=pd.date_range('2022-01-01', periods=8, freq='5min', tz='UTC'))
        result = prepare_station(frame)
        self.assertEqual(set(result), {'train', 'validation'})
        self.assertEqual(len(result['train']), 4)
        self.assertTrue(result['train']['label'].eq(2.).all())
        self.assertTrue(result['train'].depth_lag_0_cm.eq(1.).all())
        self.assertEqual(len(result['validation']), 0)

    def test_source_month_key_allows_pretest_filtering_before_read(self):
        self.assertLess(month_key('floodnet-flood-sensor-jun-2023.csv'), (2023, 7))
        self.assertGreaterEqual(month_key('floodnet-flood-sensor-jul-2023.csv'), (2023, 7))

    def test_sample_matches_frozen_expected_prediction(self):
        frame, station = read_input(ROOT / 'examples/raw_synthetic.json')
        expected = json.loads((ROOT / 'examples/expected_synthetic.json').read_text(encoding='utf-8'))
        actual = predict(frame, frame.index.max(), station)
        self.assertEqual(actual['status'], 'ok')
        for key in ('current_depth_cm', 'predicted_delta_cm', 'predicted_depth_cm', 'persistence_depth_cm'):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, rtol=1e-6)

    def test_missing_history_is_unavailable_not_filled(self):
        frame = observations([10, 20], ['2026-01-01T10:00Z', '2026-01-01T10:10Z'])
        self.assertEqual(predict(frame, frame.index.max(), 'synthetic')['status'], 'unavailable')

    def test_last_physical_null_invalidates_bin(self):
        frame = observations([10, 20, 30, np.nan], ['2026-01-01T10:00Z', '2026-01-01T10:05Z', '2026-01-01T10:08Z', '2026-01-01T10:10Z'])
        self.assertTrue(np.isnan(build_features(frame, frame.index.max()).iloc[-1].depth_lag_0_cm))

    def test_negative_raw_is_floored_but_highs_not_clipped(self):
        frame = observations([-10, 2000, 3000])
        feature = build_features(frame, frame.index.max()).iloc[-1]
        self.assertEqual(feature.depth_lag_10_cm, 0.)
        self.assertEqual(feature.depth_lag_0_cm, 300.)

    def test_future_or_unfinished_bin_cannot_change_completed_features(self):
        past = observations([10, 20, 30])
        future = pd.concat([past, observations([9999], ['2026-01-01T10:11Z'])])
        a = build_features(past, '2026-01-01T10:11Z')
        b = build_features(future, '2026-01-01T10:11Z')
        pd.testing.assert_frame_equal(a, b)

    def test_duplicate_time_keeps_first_physical_row(self):
        frame = observations([10, 999, 20, 30], ['2026-01-01T10:00Z', '2026-01-01T10:00Z', '2026-01-01T10:05Z', '2026-01-01T10:10Z'])
        self.assertEqual(build_features(frame, frame.index.max()).iloc[-1].depth_lag_10_cm, 1.)

    def test_multistation_input_requires_explicit_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.csv'
            path.write_text('time,deployment_id,depth_raw_mm\n2026-01-01T10:00Z,a,10\n2026-01-01T10:05Z,b,20\n')
            with self.assertRaises(ValueError):
                read_input(path)
            frame, station = read_input(path, 'a')
            self.assertEqual((len(frame), station), (1, 'a'))

    def test_naive_cutoff_is_rejected(self):
        with self.assertRaises(ValueError):
            build_features(observations([10, 20, 30]), '2026-01-01T10:10')


if __name__ == '__main__':
    unittest.main()
