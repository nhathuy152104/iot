"""Synthetic checks for causal inputs, explicit exclusions and future-only labels."""
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from raw_branch.pipeline import eligibility_reasons, make_raw_features, make_raw_grid, membership_summary
from raw_branch.experiment import source_months
from raw_branch.report import write_report

LABEL = 'future_proc_cm'
FEATURES = [f'depth_lag_{lag}_cm' for lag in (0, 5, 10)] + [f'source_age_lag_{lag}_seconds' for lag in (0, 5, 10)] + [
    'slope_5min_cm_per_min', 'slope_10min_cm_per_min', 'mean_three_bins_cm',
    'min_three_bins_cm', 'max_three_bins_cm', 'std_three_bins_cm']


def observations(times, raw, proc=None):
    return pd.DataFrame({'depth_raw_mm': raw, 'depth_proc_mm': raw if proc is None else proc},
        index=pd.to_datetime(times, utc=True))


class RawPipelineTests(unittest.TestCase):
    def test_current_bin_median_suppresses_single_blip_but_not_a_whole_high_bin(self):
        x = observations(['2022-01-01 00:01', '2022-01-01 00:02', '2022-01-01 00:05', '2022-01-01 00:10'], [20, 1000, 20, 1500])
        grid = make_raw_grid(x)
        np.testing.assert_allclose(grid.raw_median_cm, [2, 150])
        self.assertEqual(grid.audit_high_raw_ge_100cm_rows.iloc[0], 1)

    def test_negative_values_are_floored_and_counted_not_dropped(self):
        grid = make_raw_grid(observations(['2022-01-01 00:05', '2022-01-01 00:10'], [-2000, 2000]))
        np.testing.assert_allclose(grid.raw_median_cm, [0, 200])
        np.testing.assert_array_equal(grid.audit_negative_raw_rows, [1, 0])

    def test_latest_null_is_not_replaced_by_earlier_valid_raw(self):
        x = observations(['2022-01-01 00:01', '2022-01-01 00:05', '2022-01-01 00:10'], [20, np.nan, 30], [20, 40, 50])
        grid = make_raw_grid(x)
        self.assertTrue(np.isnan(grid.raw_median_cm.iloc[0]))
        self.assertEqual(grid.depth_proc_mm.iloc[0], 40)

    def test_no_fill_across_empty_bins(self):
        grid = make_raw_grid(observations(['2022-01-01 00:05', '2022-01-01 00:15'], [20, 30]))
        self.assertTrue(np.isnan(grid.raw_median_cm.iloc[1]))
        self.assertFalse(grid.audit_bin_observed.iloc[1])

    def test_unfinished_bins_and_not_yet_observed_rows_are_unavailable(self):
        x = observations(['2022-01-01 00:05', '2022-01-01 00:07', '2022-01-01 00:10'], [10, 100, 200])
        grid = make_raw_grid(x, as_of=pd.Timestamp('2022-01-01 00:08', tz='UTC'))
        self.assertEqual(len(grid), 1)
        self.assertEqual(grid.raw_median_cm.iloc[0], 1)

    def test_prefix_replay_equals_full_history_inputs_even_when_future_is_extreme(self):
        times = pd.date_range('2022-01-01', periods=35, freq='min', tz='UTC')
        x = pd.DataFrame({'depth_raw_mm': np.arange(35.), 'depth_proc_mm': np.arange(35.)}, index=times)
        cutoff = times[20]
        full = make_raw_features(make_raw_grid(x), 'a', LABEL)
        prefix = make_raw_features(make_raw_grid(x, as_of=cutoff), 'a', LABEL)
        pd.testing.assert_frame_equal(full.loc[prefix.index, FEATURES], prefix[FEATURES])
        changed = x.copy()
        changed.loc[changed.index > cutoff, 'depth_raw_mm'] = 1000000
        other = make_raw_features(make_raw_grid(changed), 'a', LABEL)
        pd.testing.assert_frame_equal(full.loc[prefix.index, FEATURES], other.loc[prefix.index, FEATURES])

    def test_proc_cannot_change_raw_features_or_raw_missingness(self):
        times = pd.date_range('2022-01-01', periods=7, freq='5min', tz='UTC')
        x = pd.DataFrame({'depth_raw_mm': np.arange(7.) * 10, 'depth_proc_mm': np.arange(7.) * 10}, index=times)
        a = make_raw_features(make_raw_grid(x), 'a', LABEL)
        x['depth_proc_mm'] = np.nan
        b = make_raw_features(make_raw_grid(x), 'a', LABEL)
        pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])
        self.assertTrue(b[LABEL].isna().all())

    def test_label_is_last_proc_of_exact_future_bin_not_raw_or_future_maximum(self):
        x = observations(['2022-01-01 00:00', '2022-01-01 00:05', '2022-01-01 00:10',
            '2022-01-01 00:15', '2022-01-01 00:16', '2022-01-01 00:20'],
            [0, 10, 20, 30, 900, 40], [0, 10, 20, 30, 900, 50])
        f = make_raw_features(make_raw_grid(x), 'a', LABEL)
        self.assertEqual(f.loc[pd.Timestamp('2022-01-01 00:10', tz='UTC'), LABEL], 5)
        self.assertEqual(f.loc[pd.Timestamp('2022-01-01 00:10', tz='UTC'), 'depth_lag_0_cm'], 2)

    def test_exclusion_reasons_partition_all_rows_without_high_value_cut(self):
        times = pd.date_range('2022-01-01', periods=8, freq='5min', tz='UTC')
        x = pd.DataFrame({'depth_raw_mm': [0, 10, 20000, 30, np.nan, 50, 60, 70], 'depth_proc_mm': 100.}, index=times)
        f = make_raw_features(make_raw_grid(x), 'a', LABEL)
        reasons = eligibility_reasons(f, LABEL, FEATURES)
        self.assertEqual(len(reasons), reasons.value_counts().sum())
        self.assertEqual(reasons.iloc[2], 'kept')
        self.assertEqual(reasons.iloc[4], 'missing_or_invalid_current_raw_bin')
        self.assertEqual(reasons.iloc[5], 'missing_or_invalid_lag5_raw_bin')

    def test_membership_totals_and_lost_wet_windows_are_explicit(self):
        times = pd.date_range('2022-01-01', periods=3, freq='5min', tz='UTC')
        raw = pd.DataFrame({LABEL: [0., 20.], 'audit_negative_raw_history': [False, True],
                            'audit_sparse_raw_history': True}, index=times[:2])
        old = pd.DataFrame({LABEL: [20., 30.]}, index=times[1:])
        reasons = pd.Series(['kept', 'kept', 'missing_or_invalid_current_raw_bin'], index=times)
        row = membership_summary('a', 'train', raw, old, reasons, LABEL)
        self.assertEqual(row['common_windows'], 1)
        self.assertEqual(row['raw_only_windows'], 1)
        self.assertEqual(row['proc_only_windows'], 1)
        self.assertEqual(row['proc_only_wet_ge_10cm'], 1)

    def test_source_months_exclude_test_period_before_any_cache_access(self):
        files = ['floodnet-flood-sensor-jun-2023.csv', 'floodnet-flood-sensor-jul-2023.csv',
                 'floodnet-flood-sensor-jan-2023.csv', 'floodnet-flood-sensor-oct-2023.csv']
        selected = source_months({'source_files': files})
        self.assertEqual([p.name for p in selected],
            ['floodnet-flood-sensor-jan-2023.csv', 'floodnet-flood-sensor-jun-2023.csv'])

    def test_nonfinite_raw_is_missing_not_an_extreme_or_zero_depth(self):
        x = observations(['2022-01-01 00:05', '2022-01-01 00:10'], [np.inf, -np.inf], [10., 20.])
        grid = make_raw_grid(x)
        self.assertTrue(grid.raw_median_cm.isna().all())
        self.assertTrue(grid.audit_last_raw_missing.all())

    def test_future_label_can_be_missing_without_erasing_input_features(self):
        times = pd.date_range('2022-01-01', periods=6, freq='5min', tz='UTC')
        x = pd.DataFrame({'depth_raw_mm': 20., 'depth_proc_mm': [20., 20., 20., 20., np.nan, 20.]}, index=times)
        f = make_raw_features(make_raw_grid(x), 'a', LABEL)
        self.assertTrue(np.isfinite(f.loc[times[2], FEATURES].to_numpy(dtype=float)).all())
        reasons = eligibility_reasons(f, LABEL, FEATURES)
        self.assertEqual(reasons.loc[times[2]], 'missing_or_invalid_future_proc_label')

    def test_report_renders_dynamic_common_validation_counts_without_file_io(self):
        row = {'model': 'persistence', 'windows': 2, 'wet_targets_ge_10cm': 1,
               'mae_cm': .2, 'mae_wet_cm': .3, 'balanced_wet_nonwet_mae_cm': .2, 'fp': 0, 'fn': 1}
        summary = {'counts': {'train': 10, 'validation': 2}, 'coverage': [], 'exclusion_reasons': [],
                   'validation_metrics': [row], 'paired_validation_metrics': [row],
                   'best_delta': 'raw_delta_fixture', 'selected_forecaster': 'persistence'}
        output = MagicMock()
        write_report(output, summary)
        text = output.__truediv__.return_value.write_text.call_args.args[0]
        self.assertIn('raw 2, proc 0', text)
        self.assertIn('persistence', text)


if __name__ == '__main__':
    unittest.main()
