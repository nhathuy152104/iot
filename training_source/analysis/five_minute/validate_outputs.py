"""Check every exported station/split without training or fitting thresholds."""
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import OUT, POLICY, classify, load_json, save_json, split_masks


def main():
    manifest = load_json(OUT / 'selection_manifest.json')
    stations = pd.read_csv(OUT / 'station_selection.csv')
    counts = pd.read_csv(OUT / 'window_counts.csv')
    assert stations['deployment_id'].is_unique
    assert len(stations) == manifest['stations_scanned']
    assert len(counts) == 3 * len(stations)
    assert len(manifest['selected_core']) <= POLICY['max_selected_stations']
    assert not any(c.startswith(('audit_', 'review_')) for c in manifest['feature_columns'])
    checked_rows = 0
    checked_files = 0
    for row in counts.to_dict(orient='records'):
        station, split = row['deployment_id'], row['split']
        path = OUT / 'provisional_windows' / f'{station}_{split}.parquet'
        windows = pd.read_parquet(path)
        assert len(windows) == row['windows'], str(path)
        assert windows.index.is_unique and windows.index.is_monotonic_increasing, str(path)
        assert split_masks(windows.index)[split].all(), str(path)
        assert windows['deployment_id'].eq(station).all(), str(path)
        assert np.isfinite(windows[manifest['feature_columns']].to_numpy()).all(), str(path)
        assert windows[manifest['label_column']].ge(0).all(), str(path)
        assert windows['source_age_lag_0_seconds'].ge(0).all(), str(path)
        assert windows['source_age_lag_0_seconds'].lt(300).all(), str(path)
        assert windows['audit_realized_target_lead_seconds'].gt(300).all(), str(path)
        assert windows['audit_realized_target_lead_seconds'].le(600).all(), str(path)
        wet = windows[manifest['label_column']].ge(10)
        assert int(wet.sum()) == row['wet_windows_ge_10cm'], str(path)
        checked_files += 1
        checked_rows += len(windows)
    # Recompute categories from TRAIN statistics only, not validation/test columns.
    for station in stations.to_dict(orient='records'):
        train = counts.loc[(counts['deployment_id'] == station['deployment_id']) & counts['split'].eq('train')].iloc[0].to_dict()
        category, _ = classify(train, station['train_missing_grid_fraction'])
        assert category == station['category'], station['deployment_id']
        if station['selected_core']:
            assert category == 'candidate_core'
    actual_selected = set(stations.loc[stations['selected_core'], 'deployment_id'])
    assert actual_selected == set(manifest['selected_core'])
    result = {'status': 'PASS', 'files_checked': checked_files, 'windows_checked': checked_rows,
              'checks': ['finite whitelisted features', 'nonnegative labels', 'source freshness',
                         'target realized lead within (5min,10min]', 'no overlapping split supports',
                         '24h embargo each side', 'consistent counts', 'train-only category recomputation'],
              'limits': 'Structural validation only; does not validate floods or proc filter causality.'}
    save_json(OUT / 'output_validation.json', result)
    print(result)


if __name__ == '__main__':
    main()
