"""Train a separate fixed three-core candidate from downloaded CSVs, no caches/test.

This is a handover convenience path, NOT a new model selection protocol or a
claim that a fresh download is byte-identical to the original research archive.
"""
import argparse
import calendar
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from predict_depth import FEATURES, build_features, load_checked_model, sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def month_key(name):
    parts = Path(name).stem.split('-')
    return int(parts[-1]), list(calendar.month_abbr).index(parts[-2].title())


def prepare_station(source):
    source = source.sort_index(kind='stable')
    source = source.loc[~source.index.duplicated(keep='first')].copy()
    features = build_features(source[['depth_raw_mm']], source.index.max())
    source['proc'] = pd.to_numeric(source.depth_proc_mm, errors='coerce').replace([np.inf, -np.inf], np.nan)
    source['bin'] = source.index.ceil('5min')
    # Last physical row, including null; never aggregate proc using skip-null last().
    proc = source.drop_duplicates('bin', keep='last').set_index('bin').proc.reindex(features.index)
    label = proc.shift(-2) / 10
    valid = np.isfinite(features).all(axis=1) & np.isfinite(label) & label.ge(0)
    index = features.index
    train_end, validation_end = pd.Timestamp('2023-01-01', tz='UTC'), pd.Timestamp('2023-07-01', tz='UTC')
    embargo = pd.Timedelta(hours=24)
    masks = {'train': (index + pd.Timedelta(minutes=10)) < train_end - embargo,
             'validation': ((index - pd.Timedelta(minutes=15)) >= train_end + embargo) &
                           ((index + pd.Timedelta(minutes=10)) < validation_end - embargo)}
    result = {}
    for split, mask in masks.items():
        kept = valid & mask
        frame = features.loc[kept].copy()
        frame['label'] = label.loc[kept]
        result[split] = frame
    return result


def metrics(y, p):
    error = np.abs(y - p)
    wet = y >= 10
    if not wet.any() or wet.all():
        raise ValueError('Both target groups required for balanced MAE')
    return {'windows': len(y), 'mae_cm': float(error.mean()), 'mae_wet_cm': float(error[wet].mean()),
            'mae_nonwet_cm': float(error[~wet].mean()),
            'balanced_mae_cm': float((error[wet].mean() + error[~wet].mean()) / 2)}


def run(data_dir, output_dir, prepare_only):
    frozen, config = load_checked_model()
    for name, version in config['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Pinned runtime mismatch: {name}; recreate environment before reproduction')
    inventory = json.loads((ROOT / 'DATA_INVENTORY.json').read_text(encoding='utf-8'))
    names = sorted([r['relative_path'] for r in inventory['csv_source_files']
                    if month_key(r['relative_path']) < (2023, 7)], key=month_key)
    if not names:
        raise ValueError('No pre-test source months in inventory')
    paths = [data_dir / name for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Download/extract required monthly CSV: {path.name}')
    output = output_dir / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_downloaded_raw_candidate')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol_before_preparation.json', {
        'policy': 'fixed d3_weight20 only; same raw preprocessing, features, future-proc label, temporal split, embargo; no model selection',
        'source_months': names, 'test_policy': 'do not read monthly CSV July-2023 onward',
        'limits': 'new download provenance; must compare before claiming exact reproduction; no promotion/deployment',
        'trainer_sha256': sha256(Path(__file__)), 'runtime_sha256': sha256(ROOT / 'runtime/predict_depth.py')})
    stations = config['original']['stations']
    parts = {station: [] for station in stations}
    provenance = []
    for path in paths:
        before = path.stat()
        with pd.read_csv(path, usecols=['time', 'deployment_id', 'depth_raw_mm', 'depth_proc_mm'], chunksize=200000) as reader:
            for chunk in reader:
                chunk = chunk.loc[chunk.deployment_id.isin(stations)].copy()
                if chunk.empty:
                    continue
                chunk.index = pd.to_datetime(chunk.time, format='mixed', utc=True, errors='raise')
                for station in stations:
                    selected = chunk.loc[chunk.deployment_id.eq(station), ['depth_raw_mm', 'depth_proc_mm']]
                    if not selected.empty:
                        parts[station].append(selected)
        checksum = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source file changed during preparation')
        provenance.append({'filename': path.name, 'bytes': after.st_size, 'sha256': checksum})
        print(f'Read pre-test month: {path.name}', flush=True)
    splits = {'train': [], 'validation': []}
    for station, sources in parts.items():
        if not sources:
            raise ValueError(f'No data for core station {station}')
        prepared = prepare_station(pd.concat(sources))
        for split, frame in prepared.items():
            splits[split].append(frame)
    train, validation = [pd.concat(splits[s]).sort_index(kind='stable') for s in ('train', 'validation')]
    counts = {'train': len(train), 'validation': len(validation)}
    write(output / 'preparation.json', {'counts': counts, 'sources': provenance})
    if counts != {'train': 270312, 'validation': 92215}:
        raise ValueError('Counts differ from frozen study; inspect data/version/policy instead of silently fitting')
    if prepare_only:
        print(f'PREPARED ONLY (no fit): {output}')
        return
    params = config['original']['common_xgboost_params']
    write(output / 'config_before_fit.json', {'parameters': params, 'max_depth': 3, 'n_estimators': 250,
        'seed': 42, 'wet_weight': 20, 'features': FEATURES, 'counts': counts})
    model = XGBRegressor(**params, max_depth=3, n_estimators=250, random_state=42, device='cpu', eval_metric='mae')
    model.fit(train[FEATURES], train.label - train.depth_lag_0_cm,
              sample_weight=np.where(train.label >= 10, 20., 1.), verbose=False)
    model.save_model(output / 'candidate.ubj')
    y = validation.label.to_numpy()
    current = validation.depth_lag_0_cm.to_numpy()
    predictions = {'persistence': current,
        'frozen_three_core': np.maximum(0, current + frozen.predict(validation[FEATURES])),
        'new_candidate': np.maximum(0, current + model.predict(validation[FEATURES]))}
    reloaded = XGBRegressor()
    reloaded.load_model(output / 'candidate.ubj')
    np.testing.assert_allclose(predictions['new_candidate'], np.maximum(0, current + reloaded.predict(validation[FEATURES])), atol=1e-6, rtol=1e-6)
    write(output / 'summary.json', {'status': 'NEW_RESEARCH_CANDIDATE_NOT_PROMOTED', 'test_loaded': False,
        'metrics': {name: metrics(y, p) for name, p in predictions.items()}, 'roundtrip': 'PASS'})
    print(f'NEW CANDIDATE SAVED (frozen model unchanged): {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output_dir.resolve(), args.prepare_only)
