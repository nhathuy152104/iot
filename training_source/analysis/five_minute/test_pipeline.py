"""Only synthetic data; no FloodNet raw records are embedded."""
import unittest

import numpy as np
import pandas as pd

from pipeline import DEPTHS, TRAIN_END, EMBARGO, classify, make_features, make_grid, split_masks, temporal_groups


def frame(times, values):
    return pd.DataFrame({column: values for column in DEPTHS}, index=pd.to_datetime(times, utc=True))


class FiveMinuteTests(unittest.TestCase):
    def test_last_row_preserves_null_and_alignment(self):
        data = frame(['2022-01-01T10:01:00Z', '2022-01-01T10:04:00Z'], [20., np.nan])
        data.loc[data.index[-1], 'depth_raw_mm'] = 50
        grid = make_grid(data)
        row = grid.loc['2022-01-01T10:05:00Z']
        self.assertTrue(pd.isna(row['depth_proc_mm']))
        self.assertEqual(row['depth_raw_mm'], 50)
        self.assertEqual(row['source_age_seconds'], 60)

    def test_no_interpolation_and_right_closed_bins(self):
        data = frame(['2022-01-01T10:00:00Z', '2022-01-01T10:11:00Z'], [20., 30.])
        grid = make_grid(data)
        self.assertEqual(grid.loc['2022-01-01T10:00:00Z', 'depth_proc_mm'], 20)
        self.assertTrue(pd.isna(grid.loc['2022-01-01T10:05:00Z', 'depth_proc_mm']))
        self.assertTrue(pd.isna(grid.loc['2022-01-01T10:10:00Z', 'depth_proc_mm']))
        self.assertEqual(grid.loc['2022-01-01T10:15:00Z', 'depth_proc_mm'], 30)

    def test_bin_last_observation_not_average(self):
        data = frame(['2022-01-01T10:01:00Z', '2022-01-01T10:04:00Z'], [100., 200.])
        self.assertEqual(make_grid(data).iloc[0]['depth_proc_mm'], 200)

    def test_features_target_and_complete_history(self):
        times = pd.date_range('2022-01-01', periods=7, freq='5min', tz='UTC')
        grid = make_grid(frame(times, np.arange(7) * 10.))
        features, eligible = make_features(grid, 'synthetic')
        self.assertEqual(int(eligible.sum()), 3)
        first = features.loc[eligible].iloc[0]
        self.assertEqual(first['depth_lag_0_cm'], 2)
        self.assertEqual(first['depth_lag_10_cm'], 0)
        self.assertEqual(first['target_depth_bin_t_plus_10min_cm'], 4)
        self.assertEqual(first['audit_realized_target_lead_seconds'], 600)
        grid.loc[times[1], 'depth_proc_mm'] = np.nan
        _, eligible = make_features(grid, 'synthetic')
        self.assertEqual(int(eligible.sum()), 1)

    def test_label_only_requires_target_bin_not_all_future_bins(self):
        times = pd.date_range('2022-01-01', periods=7, freq='5min', tz='UTC')
        grid = make_grid(frame(times, np.arange(7) * 10.))
        grid.loc[times[3], 'depth_proc_mm'] = np.nan
        _, eligible = make_features(grid, 'synthetic')
        self.assertTrue(eligible.loc[times[2]])

    def test_no_implicit_future_in_features(self):
        times = pd.date_range('2022-01-01', periods=7, freq='5min', tz='UTC')
        grid = make_grid(frame(times, np.arange(7) * 10.))
        before, _ = make_features(grid, 'synthetic')
        grid.loc[times[3]:, DEPTHS] = 9000
        after, _ = make_features(grid, 'synthetic')
        cols = [c for c in before if c.startswith(('depth_lag_', 'source_age_', 'slope_', 'mean_', 'min_', 'max_', 'std_'))]
        pd.testing.assert_series_equal(before.loc[times[2], cols], after.loc[times[2], cols])

    def test_split_support_and_embargo(self):
        times = pd.date_range(TRAIN_END - pd.Timedelta(days=2), TRAIN_END + pd.Timedelta(days=2), freq='5min')
        masks = split_masks(times)
        self.assertTrue(((times[masks['train']] + pd.Timedelta(minutes=10)) < TRAIN_END - EMBARGO).all())
        self.assertTrue(((times[masks['validation']] - pd.Timedelta(minutes=15)) >= TRAIN_END + EMBARGO).all())
        self.assertFalse((masks['train'] & masks['validation']).any())

    def test_temporal_groups_are_not_count_of_windows(self):
        times = pd.to_datetime(['2022-01-01T00:00Z', '2022-01-01T00:05Z', '2022-01-01T07:00Z'], utc=True)
        self.assertEqual(temporal_groups(times), 2)
        self.assertEqual(temporal_groups(times[:0]), 0)

    def test_classification_train_only(self):
        train = {'windows': 10000, 'decision_days': 100,
                 'history_raw_proc_disagreement_fraction': .01, 'history_proc_jump_fraction': 0,
                 'wet_windows_ge_10cm': 100, 'nonwet_windows_lt_10cm': 9900,
                 'wet_days': 5, 'wet_temporal_groups_6h': 5}
        self.assertEqual(classify(train, .1)[0], 'candidate_core')
        train['history_proc_jump_fraction'] = .02
        self.assertEqual(classify(train, .1)[0], 'review_quality')
        self.assertEqual(classify({'windows': 0}, None)[0], 'no_train_windows')


if __name__ == '__main__':
    unittest.main()
