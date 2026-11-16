"""Read-only integrity and ledger reconciliation for part-B event evaluation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from delta_experiment import read_json
from event_evaluation import summarize
from train_baseline_xgboost import BASE, digest


def validate(output):
    manifest = read_json(output / 'artifact_validation.json')
    summary = read_json(output / 'summary.json')
    assert manifest['status'] == summary['status'] == 'PASS'
    assert not summary['test_loaded'] and not summary['train_loaded'] and not summary['fit_called']
    for name, expected in manifest['output_sha256'].items():
        assert digest(output / name) == expected, name
    for key in ('input_sha256', 'frozen_sha256'):
        for name, expected in manifest[key].items():
            assert digest(Path(name)) == expected, name
    for name, expected in manifest['code_sha256'].items():
        assert digest(BASE / name) == expected, name
    table = pd.read_csv(output / 'event_metrics.csv')
    events = pd.read_csv(output / 'events_local.csv')
    alarms = pd.read_csv(output / 'alarms_local.csv')
    metrics = pd.read_csv(output / 'replayed_window_metrics.csv').set_index('model')
    for row in table.to_dict('records'):
        def selected(frame):
            mask = frame.model.eq(row['model']) & frame.confirm_bins.eq(row['confirm_bins'])
            if row['station'] != 'ALL':
                mask &= frame.station.eq(row['station'])
            return frame.loc[mask].to_dict('records')
        a, b = selected(events), selected(alarms)
        recomputed = summarize(a, b, row['windows'])
        for key, value in recomputed.items():
            if value is None:
                assert pd.isna(row[key]), key
            else:
                np.testing.assert_allclose(value, row[key], atol=1e-9, rtol=1e-9, err_msg=key)
        assert sum(r['high_windows'] for r in a) == row['wet_windows']
        assert sum(r['false_windows'] for r in b) == row['false_alarm_windows']
        if row['station'] == 'ALL':
            station_rows = table.loc[table.model.eq(row['model']) & table.confirm_bins.eq(row['confirm_bins']) & table.station.ne('ALL')]
            for key in ('windows', 'events', 'detected_events', 'missed_events', 'alarm_episodes', 'false_alarm_episodes',
                        'false_alarm_windows', 'missed_wet_windows', 'wet_windows'):
                assert station_rows[key].sum() == row[key], key
            if row['confirm_bins'] == 1:
                assert row['false_alarm_windows'] == metrics.loc[row['model'], 'fp']
                assert row['missed_wet_windows'] == metrics.loc[row['model'], 'fn']
    print(f'PASS: {len(table)} metric rows reconcile with event/alarm ledgers; '
          f'{len(manifest["frozen_sha256"])} frozen files unchanged; input/output/code checksums match')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    validate(parser.parse_args().run_directory.resolve())
