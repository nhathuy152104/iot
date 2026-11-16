import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from delta_experiment import METRIC
from raw_expansion.selection import RULES, build_candidate_train, select_overall, selection_failures
from raw_expansion.experiment import candidate_train_manifest, load_frozen_split
from raw_expansion.report import write_report


FEATURES = [f'depth_lag_{lag}_cm' for lag in (0, 5, 10)] + [f'source_age_lag_{lag}_seconds' for lag in (0, 5, 10)] + [
    'slope_5min_cm_per_min', 'slope_10min_cm_per_min', 'mean_three_bins_cm',
    'min_three_bins_cm', 'max_three_bins_cm', 'std_three_bins_cm']


class ExpansionSelectionTests(unittest.TestCase):
    def test_all_coverage_rules_are_inclusive_and_failures_are_explicit(self):
        audit = dict(RULES)
        self.assertEqual(selection_failures(audit), [])
        for key in RULES:
            changed = {**audit, key: RULES[key] - 1}
            self.assertEqual(selection_failures(changed), [key])

    def test_one_high_day_is_not_diverse_just_because_it_has_many_rows(self):
        audit = {**RULES, 'wet_ge_10cm': 5000, 'wet_days': 1, 'wet_temporal_groups_6h': 1}
        self.assertEqual(selection_failures(audit), ['wet_days', 'wet_temporal_groups_6h'])

    def test_quality_flags_do_not_silently_drop_eligible_windows(self):
        times = pd.date_range('2022-01-01', periods=8, freq='5min', tz='UTC')
        source = pd.DataFrame({'depth_raw_mm': [-1000., 2000., 2000., -1000., 0., 0., 0., 0.],
                               'depth_proc_mm': [0., 2000., 2000., 0., 0., 0., 0., 0.]}, index=times)
        frame, audit, reasons = build_candidate_train('fixture', source, FEATURES, 'future_cm')
        self.assertEqual(len(frame), 4)
        self.assertGreater(audit['raw_ge_100cm_rows_retained'], 0)
        self.assertGreater(audit['raw_negative_history_fraction'], 0)
        self.assertEqual(sum(r['windows'] for r in reasons), len(source))

    def test_candidate_audit_ignores_validation_and_future_rows(self):
        train_times = pd.date_range('2022-01-01', periods=8, freq='5min', tz='UTC')
        val_times = pd.date_range('2023-01-05', periods=8, freq='5min', tz='UTC')
        a = pd.DataFrame({'depth_raw_mm': 10., 'depth_proc_mm': 10.}, index=train_times)
        b = pd.DataFrame({'depth_raw_mm': 10000., 'depth_proc_mm': 10000.}, index=val_times)
        first, audit_a, _ = build_candidate_train('fixture', a, FEATURES, 'future_cm')
        second, audit_b, _ = build_candidate_train('fixture', pd.concat([a, b]), FEATURES, 'future_cm')
        pd.testing.assert_frame_equal(first, second, check_freq=False)
        for key in RULES:
            self.assertEqual(audit_a[key], audit_b[key])

    def test_frozen_model_or_baseline_can_beat_expansion(self):
        def row(name, score):
            return {'model': name, METRIC: score, 'mae_cm': .1}
        self.assertEqual(select_overall([row('expanded_x', 2), row('three_x', 1)]), 'three_x')
        self.assertEqual(select_overall([row('expanded_x', 1), row('three_x', 1)]), 'three_x')
        self.assertEqual(select_overall([row('expanded_x', .9), row('three_x', 1)]), 'expanded_x')
        self.assertEqual(select_overall([row('expanded_x', 1), row('persistence', 1)]), 'persistence')
        with self.assertRaises(ValueError):
            select_overall([row('bad', None)])

    def test_candidate_manifest_excludes_all_2023_sources_without_mutation(self):
        manifest = {'source_files': ['floodnet-flood-sensor-dec-2022.csv',
            'floodnet-flood-sensor-jan-2023.csv', 'floodnet-flood-sensor-jul-2023.csv'], 'stations_scanned': 58}
        result = candidate_train_manifest(manifest)
        self.assertEqual(result['source_files'], ['floodnet-flood-sensor-dec-2022.csv'])
        self.assertEqual(len(manifest['source_files']), 3)

    def test_test_split_is_blocked_before_any_file_access(self):
        with patch('raw_expansion.experiment.read_json') as read:
            with self.assertRaises(ValueError):
                load_frozen_split(None, 'test')
            read.assert_not_called()

    def test_report_preserves_a_frozen_arm_win_without_file_io(self):
        row = {'model': 'three_fixture', 'mae_cm': .1, 'mae_wet_cm': .2,
               METRIC: .15, 'fp': 2, 'fn': 3}
        summary = {'counts': {'validation': 2}, 'selected_additions': [],
            'core_stations': ['a', 'b', 'c'], 'station_audits_train_only': [],
            'validation_metrics': [row], 'best_three': 'three_fixture',
            'best_expanded': 'expanded_fixture', 'selected_forecaster': 'three_fixture',
            'expanded_vs_three_balanced_mae_change_cm': .1, 'matched_config_comparison': []}
        output = MagicMock()
        write_report(output, summary)
        text = output.__truediv__.return_value.write_text.call_args.args[0]
        self.assertIn('three_fixture', text)
        self.assertIn('+0.100000', text)


if __name__ == '__main__':
    unittest.main()
