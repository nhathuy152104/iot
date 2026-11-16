"""Synthetic checks for retrospective partitions and source provenance guards."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from delta_experiment import DEFAULT_ORIGIN, verified_split
from diagnose_training import candidate_segments, error_rows, stages, support_masks

LABEL = 'target_depth_bin_t_plus_10min_cm'


def frame(current, future, station='a', start='2023-01-03'):
    return pd.DataFrame({'depth_lag_0_cm': current, LABEL: future, 'deployment_id': station,
        'review_raw_proc_diff_ge_10cm_history': False, 'review_proc_jump_history': False},
        index=pd.date_range(start, periods=len(current), freq='5min', tz='UTC'))


class DiagnosisTests(unittest.TestCase):
    def test_stages_partition_every_row_and_prioritize_crossings(self):
        x = frame([0, 12, 2, 8, 12, 1], [20, 0, 4, 6, 12.5, 1])
        np.testing.assert_array_equal(stages(x, LABEL),
            ['new_crossing', 'down_crossing', 'rising_gt_1cm', 'falling_gt_1cm', 'stable_high', 'stable_low'])
        rows = error_rows(x, LABEL, {'persistence': x.depth_lag_0_cm},
            {s: stages(x, LABEL) == s for s in np.unique(stages(x, LABEL))}, 'stage', 'validation')
        self.assertEqual(sum(r['windows'] for r in rows), len(x))
        self.assertAlmostEqual(sum(r['absolute_error_sum_cm'] for r in rows),
            np.abs(x[LABEL] - x.depth_lag_0_cm).sum())

    def test_support_uses_train_only_pooled_and_own_station_maxima(self):
        train = pd.concat([frame([1], [10], 'a'), frame([1], [20], 'b')])
        test = pd.concat([frame([1], [15], 'a'), frame([1], [25], 'b')])
        masks = support_masks(test, train, LABEL)
        np.testing.assert_array_equal(masks['future_gt_pooled_train_max'], [False, True])
        np.testing.assert_array_equal(masks['future_gt_own_station_train_max'], [True, True])

    def test_segments_do_not_merge_stations_or_long_gaps(self):
        x = pd.concat([frame([12, 12], [12, 12], 'a'), frame([12], [12], 'b'),
                       frame([12], [12], 'a', start='2023-01-03 01:00')]).sort_index(kind='stable')
        p = {'persistence': x.depth_lag_0_cm.to_numpy(), 'original_absolute_selected': x.depth_lag_0_cm.to_numpy()}
        rows, definitions = candidate_segments(x, LABEL, p, 'validation')
        self.assertEqual(len(definitions), 3)
        self.assertEqual(sum(d['active_windows'] for d in definitions), 4)
        self.assertEqual(len(rows), 6)

    def test_predicted_high_with_low_current_and_target_still_in_diagnostics(self):
        x = frame([0], [0])
        _, definitions = candidate_segments(x, LABEL,
            {'original_absolute_selected': np.array([12.])}, 'validation')
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]['active_windows'], 1)

    def test_changed_source_is_rejected_before_loading_data(self):
        provenance = [{'split': 'train', 'path': 'fixture.parquet', 'sha256': 'old'}]
        with patch('delta_experiment.read_json', return_value=provenance), \
             patch('delta_experiment.digest', return_value='changed'), \
             patch('delta_experiment.load_split') as load:
            with self.assertRaises(ValueError):
                verified_split('train', DEFAULT_ORIGIN, {}, [])
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
