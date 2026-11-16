import unittest
from pathlib import Path
import tempfile

import pandas as pd

from prepare_data import minute_grid, split_masks
import prepare_data


class PreparationTests(unittest.TestCase):
    def test_grid_does_not_use_future_or_fill_missing(self):
        times = pd.to_datetime(['2022-01-01T10:00:30Z', '2022-01-01T10:03:10Z'])
        frame = pd.DataFrame({c: [10., 50.] for c in ('depth_raw_mm', 'depth_filt_mm', 'depth_proc_mm')}, index=times)
        grid = minute_grid(frame)
        self.assertEqual(grid.loc['2022-01-01T10:01:00Z', 'depth_proc_mm'], 10)
        self.assertTrue(pd.isna(grid.loc['2022-01-01T10:02:00Z', 'depth_proc_mm']))
        self.assertEqual(grid.loc['2022-01-01T10:04:00Z', 'depth_proc_mm'], 50)

    def test_split_purges_overlapping_support(self):
        times = pd.date_range('2022-06-30T23:45Z', '2022-07-01T00:15Z', freq='1min')
        masks = split_masks(times)
        self.assertEqual(times[masks['train']].max(), pd.Timestamp('2022-06-30T23:49Z'))
        self.assertEqual(times[masks['validation']].min(), pd.Timestamp('2022-07-01T00:11Z'))
        self.assertFalse((masks['train'] & masks['validation']).any())

    def test_windows_require_full_history_and_exact_future_bin(self):
        times = pd.date_range('2022-01-01T00:00Z', periods=31, freq='1min')
        grid = pd.DataFrame({c: [float(i * 10) for i in range(31)]
                            for c in ('depth_raw_mm', 'depth_filt_mm', 'depth_proc_mm')}, index=times)
        with tempfile.TemporaryDirectory() as tmp:
            old_out = prepare_data.OUT
            try:
                prepare_data.OUT = Path(tmp)
                result = prepare_data.build_windows('synthetic', grid)
                windows = pd.read_parquet(Path(tmp) / 'provisional_windows' / 'synthetic_train.parquet')
                self.assertEqual(result['eligible_before_split_purge'], 11)
                self.assertEqual(windows.index.min(), times[10])
                self.assertEqual(windows.iloc[0]['depth_lag_0_cm'], 10)
                self.assertEqual(windows.iloc[0]['depth_lag_10_cm'], 0)
                self.assertEqual(windows.iloc[0]['target_depth_10min_cm'], 20)
                grid.loc[times[5], 'depth_proc_mm'] = float('nan')
                result = prepare_data.build_windows('synthetic', grid)
                self.assertEqual(result['eligible_before_split_purge'], 5)
            finally:
                prepare_data.OUT = old_out


if __name__ == '__main__':
    unittest.main()
