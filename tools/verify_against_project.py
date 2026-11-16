"""Read-only parity check against frozen validation and pre-July raw caches."""
import argparse
import calendar
import json
from pathlib import Path, PureWindowsPath
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from predict_depth import FEATURES, build_features, load_checked_model, sha256
from train_from_download import prepare_station


def resolve(value, training_root):
    relative = PureWindowsPath(value).relative_to(PureWindowsPath(r'D:\CODE\IOT training'))
    if '..' in relative.parts:
        raise ValueError('Unsafe provenance path')
    return training_root.joinpath(*relative.parts)


def verify(training_root):
    model, config = load_checked_model()
    manifest = json.loads((ROOT / 'model/window_provenance.json').read_text(encoding='utf-8'))
    source_manifest = json.loads((ROOT / 'model/source_provenance.json').read_text(encoding='utf-8'))
    parts, cache_count = [], 0
    for row in manifest:
        if row['split'] != 'validation' or row['rows'] == 0:
            continue  # Do not load train/test tables.
        path = resolve(row['path'], training_root)
        if sha256(path) != row['sha256']:
            raise ValueError(f'Window changed: {path}')
        expected = pd.read_parquet(path)
        assert len(expected) == row['rows']
        assert expected.deployment_id.eq(row['station']).all()
        sources = []
        for cache in source_manifest['cache_files']:
            if cache['station'] != row['station']:
                continue
            path = resolve(cache['path'], training_root)
            match = re.search(r'-([a-z]{3})-(\d{4})_', path.parent.name, re.IGNORECASE)
            if not match:
                raise ValueError(f'Unrecognized cache month: {path}')
            year = int(match.group(2))
            month = list(calendar.month_abbr).index(match.group(1).title())
            if (year, month) >= (2023, 7):
                continue  # Include past raw history for exact rolling numerical state; no test.
            if sha256(path) != cache['sha256']:
                raise ValueError(f'Cache changed: {path}')
            sources.append(pd.read_parquet(path, columns=['depth_raw_mm', 'depth_proc_mm']))
            cache_count += 1
        if not sources:
            raise ValueError('Required validation source caches are not available')
        raw = pd.concat(sources).sort_index(kind='stable')
        actual = build_features(raw, raw.index.max()).loc[expected.index, FEATURES]
        np.testing.assert_allclose(actual.to_numpy(), expected[FEATURES].to_numpy(), atol=1e-9, rtol=1e-9)
        prepared = prepare_station(raw)['validation']
        pd.testing.assert_index_equal(prepared.index, expected.index, check_names=False)
        np.testing.assert_allclose(prepared['label'], expected[config['label']], atol=0, rtol=0)
        prediction = np.maximum(0, actual.depth_lag_0_cm.to_numpy() + model.predict(actual))
        result = expected[['deployment_id', config['label']]].copy()
        result['prediction'] = prediction
        parts.append(result)
    validation = pd.concat(parts).sort_index(kind='stable')
    assert len(validation) == 92215
    original = training_root / 'analysis/training_runs/20261110T043429_532355Z_raw_delta'
    stored = pd.read_parquet(original / 'validation_predictions_local.parquet')
    assert stored.index.equals(validation.index)
    assert stored.deployment_id.equals(validation.deployment_id)
    np.testing.assert_allclose(validation.prediction, stored.prediction_raw_delta_d3_weight20_cm, atol=1e-6, rtol=1e-6)
    error = (validation.prediction - validation[config['label']]).abs()
    wet = validation[config['label']] >= 10
    balanced = float((error[wet].mean() + error[~wet].mean()) / 2)
    metrics = pd.DataFrame(json.loads((ROOT / 'model/metrics.json').read_text(encoding='utf-8'))).set_index('model')
    np.testing.assert_allclose(balanced, metrics.loc['raw_delta_d3_weight20', 'balanced_wet_nonwet_mae_cm'], atol=1e-9, rtol=1e-9)
    print(json.dumps({'status': 'PASS', 'validation_rows': len(validation), 'source_cache_files_checked': cache_count,
        'features_match_frozen_windows': True, 'predictions_match_original': True,
        'download_trainer_validation_membership_and_labels_match': True,
        'balanced_mae_cm': balanced, 'training_windows_loaded': False,
        'past_raw_history_included_for_exact_rolling_state': True, 'test_loaded': False}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    verify(parser.parse_args().training_root.resolve())
