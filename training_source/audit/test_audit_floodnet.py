"""Synthetic-only tests: no FloodNet observations are embedded here."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import audit_floodnet as audit


class AuditTests(unittest.TestCase):
    def test_episode_duplicates_missing_and_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = ['deployment_id', 'time', 'depth_proc_mm', 'depth_raw_mm',
                      'depth_filt_mm', 'deploy_type', 'sensor_display_name']
            samples = [(0, 0), (1, 100), (2, 110), (2, 111), (3, 120),
                       (4, ''), (5, 150), (20, 200), (21, 210), (22, 220), (23, 0)]
            with (root / 'floodnet-flood-sensor-jan-2021.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for minute, value in samples:
                    w.writerow(dict(zip(fields, ['synthetic', f'2021-01-01T00:{minute:02}:00+00:00',
                        value, value, value, 'pluvial', 'Synthetic test station'])))
            old_root, old_out = audit.ROOT, audit.OUT
            try:
                audit.ROOT = audit.OUT = root
                with contextlib.redirect_stdout(io.StringIO()):
                    audit.main()
            finally:
                audit.ROOT, audit.OUT = old_root, old_out
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual(summary['input_rows'], 11)
            self.assertEqual(summary['totals']['duplicate_timestamps'], 1)
            self.assertEqual(summary['totals']['conflicting_duplicate_timestamps'], 1)
            self.assertEqual(summary['totals']['missing_proc'], 1)
            self.assertEqual(summary['totals']['gaps_gt_10min'], 1)
            self.assertEqual(summary['sustained_counts']['pluvial']['100'], 2)
            self.assertEqual(summary['pluvial_100mm_temporal_clusters_6h'], 1)

    def test_number_rejects_nonfinite_and_missing(self):
        for value in ('', 'null', 'nan', 'inf', '-inf', None):
            self.assertIsNone(audit.number(value))
        self.assertEqual(audit.number('-5'), -5)


if __name__ == '__main__':
    unittest.main()
