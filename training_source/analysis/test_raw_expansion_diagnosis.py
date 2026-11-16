import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from archive_paths import resolve_archive_path
from diagnose_raw_expansion import composition, paired_groups, transition_counts, verified_windows
from run_relocated import relocate_values


class DiagnosisTests(unittest.TestCase):
    def test_test_split_rejected_before_manifest_access(self):
        with self.assertRaises(ValueError):
            verified_windows(Path('does-not-exist.json'), 'test', {})

    def test_in_memory_paths_do_not_change_hashes_or_original(self):
        original = {'path': r'D:\CODE\IOT training\analysis\a.json', 'sha256': 'abc',
                    'other': r'D:\CODE\IOT training-other\x'}
        moved = relocate_values(original)
        self.assertNotEqual(moved['path'], original['path'])
        self.assertEqual(moved['sha256'], 'abc')
        self.assertEqual(moved['other'], original['other'])
        self.assertEqual(original['path'], r'D:\CODE\IOT training\analysis\a.json')
        with self.assertRaises(ValueError):
            relocate_values(r'D:\CODE\IOT training\..\outside')

    def test_relocated_path_and_missing_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'analysis').mkdir()
            target = root / 'analysis' / 'sample.json'
            target.write_text('{}')
            self.assertEqual(resolve_archive_path(r'D:\CODE\IOT training\analysis\sample.json', root), target)
            with self.assertRaises(FileNotFoundError):
                resolve_archive_path(r'D:\CODE\IOT training\analysis\missing.json', root)

    def test_weight_share_is_not_wet_row_share(self):
        frame = pd.DataFrame({'y': [0., 0., 10.], 'depth_lag_0_cm': [0., 0., 9.]})
        result = composition(frame, 'y')
        self.assertAlmostEqual(result['wet_share'], 1 / 3)
        self.assertAlmostEqual(result['wet_weight20_share'], 20 / 22)

    def test_paired_partition_reconciles_balanced_score(self):
        frame = pd.DataFrame({'y': [0., 5., 10., 20.], 'depth_lag_0_cm': [0., 4., 9., 20.]})
        rows = paired_groups(frame, 'y', np.array([1., 4., 9., 20.]),
                             np.array([0., 5., 8., 18.]), np.array(['a', 'a', 'b', 'b']), 'fixture')
        self.assertEqual(sum(r['windows'] for r in rows), 4)
        self.assertAlmostEqual(sum(r['balanced_change_contribution_cm'] for r in rows), 0.25)

    def test_threshold_transitions_are_paired(self):
        result = transition_counts(np.array([0., 10., 0., 10.]),
                                   np.array([11., 11., 0., 0.]), np.array([0., 0., 11., 11.]))
        self.assertEqual(result, {'fp_resolved': 1, 'fp_introduced': 1, 'fn_resolved': 1, 'fn_introduced': 1})


if __name__ == '__main__':
    unittest.main()
