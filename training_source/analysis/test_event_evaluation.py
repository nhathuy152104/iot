import unittest

import numpy as np
import pandas as pd

from event_evaluation import evaluate_station


def fixture(y, p, times=None):
    index = pd.date_range('2023-02-01', periods=len(y), freq='5min', tz='UTC') if times is None else pd.to_datetime(times, utc=True)
    return pd.DataFrame({'target': y, 'prediction': p,
                         'target_time': index + pd.Timedelta(minutes=9)}, index=index)


class EventTests(unittest.TestCase):
    def test_one_false_episode_not_six_notifications(self):
        events, alarms, result = evaluate_station(fixture([0] * 8, [0, 11, 11, 11, 11, 11, 11, 0]))
        self.assertEqual(result['false_alarm_episodes'], 1)
        self.assertEqual(result['false_alarm_windows'], 6)
        self.assertEqual(result['events'], 0)
        self.assertIsNone(result['event_recall'])

    def test_missing_grid_splits_events_and_marks_censoring(self):
        f = fixture([0, 12, 12, 0], [0, 12, 12, 0],
                    ['2023-02-01T00:00Z', '2023-02-01T00:05Z', '2023-02-01T00:20Z', '2023-02-01T00:25Z'])
        events, _, result = evaluate_station(f)
        self.assertEqual(result['events'], 2)
        self.assertTrue(events[0]['right_censored'])
        self.assertTrue(events[1]['left_censored'])
        self.assertEqual(result['fully_observed_events'], 0)

    def test_lead_is_interval_not_alarm_start_or_fixed_ten_minutes(self):
        events, alarms, result = evaluate_station(fixture([0, 0, 12, 12, 0], [11, 11, 11, 11, 0]))
        self.assertEqual(events[0]['lead_lower_minutes'], 4)
        self.assertEqual(events[0]['lead_upper_minutes'], 9)
        self.assertTrue(alarms[0]['left_censored'])
        self.assertEqual(result['false_alarm_windows'], 2)

    def test_two_bin_confirmation_only_activates_second_bin(self):
        f = fixture([0, 12, 12, 0], [0, 12, 12, 0])
        events, alarms, result = evaluate_station(f, confirm_bins=2)
        self.assertEqual(alarms[0]['start_decision_utc'], str(f.index[2]))
        self.assertEqual(events[0]['lead_upper_minutes'], 4)
        self.assertEqual(result['missed_wet_windows'], 1)

    def test_singleton_high_kept_and_no_cross_event_warning_credit(self):
        _, _, result = evaluate_station(fixture([0, 12, 0, 0], [12, 0, 0, 0]))
        self.assertEqual(result['events'], 1)
        self.assertEqual(result['missed_events'], 1)
        self.assertEqual(result['false_alarm_episodes'], 1)

    def test_multiple_episodes_per_event_are_not_multiple_detected_events(self):
        events, alarms, result = evaluate_station(fixture([0, 12, 12, 12, 0], [0, 12, 0, 12, 0]))
        self.assertEqual(result['detected_events'], 1)
        self.assertEqual(result['alarm_episodes'], 2)
        self.assertEqual(events[0]['matching_alarm_episodes'], 2)
        self.assertEqual(result['fragmented_detected_events'], 1)

    def test_empty_and_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_station(fixture([], []))
        f = fixture([0, 12], [0, 12])
        with self.assertRaises(ValueError):
            evaluate_station(pd.concat([f, f]))


if __name__ == '__main__':
    unittest.main()
