"""Portable past-only RAW inference. No proc labels, fitting, API or test loading."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BUNDLE = Path(__file__).resolve().parents[1]
FEATURES = ['depth_lag_0_cm', 'depth_lag_5_cm', 'depth_lag_10_cm',
    'source_age_lag_0_seconds', 'source_age_lag_5_seconds', 'source_age_lag_10_seconds',
    'slope_5min_cm_per_min', 'slope_10min_cm_per_min',
    'mean_three_bins_cm', 'min_three_bins_cm', 'max_three_bins_cm', 'std_three_bins_cm']


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checked_time(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('Timestamp must include a timezone, e.g. 2026-01-01T10:10:00Z')
    return result.tz_convert('UTC')


def build_features(observations, as_of):
    """Match frozen raw pipeline, but never construct/require future labels."""
    cutoff = checked_time(as_of).floor('5min')
    if observations.index.tz is None or observations.index.hasnans:
        raise ValueError('Timezone-aware valid observation index required')
    source = observations.loc[observations.index <= cutoff].sort_index(kind='stable').copy()
    if source.empty:
        return pd.DataFrame(columns=FEATURES, dtype=float)
    source = source.loc[~source.index.duplicated(keep='first')]
    source['raw'] = pd.to_numeric(source.depth_raw_mm, errors='coerce').replace([np.inf, -np.inf], np.nan)
    source['bin'] = source.index.ceil('5min')
    source['source_time'] = source.index
    source['nonnegative_raw'] = source.raw.clip(lower=0)
    grouped = source.groupby('bin', sort=True)
    last = source.drop_duplicates('bin', keep='last').set_index('bin')
    last['depth'] = grouped.nonnegative_raw.median() / 10
    last.loc[last.raw.isna(), 'depth'] = np.nan
    if (cutoff - last.index.min()) / pd.Timedelta(minutes=5) > 2_000_000:
        raise ValueError('Time range too large; provide a recent single-station history')
    grid = last.reindex(pd.date_range(last.index.min(), cutoff, freq='5min', name='decision_time_utc'))
    depth = grid.depth
    age = (grid.index.to_series() - grid.source_time).dt.total_seconds()
    rolling = depth.rolling(3, min_periods=3)
    frame = pd.DataFrame(index=grid.index)
    for steps in (0, 1, 2):
        frame[f'depth_lag_{5 * steps}_cm'] = depth.shift(steps)
        frame[f'source_age_lag_{5 * steps}_seconds'] = age.shift(steps)
    frame['slope_5min_cm_per_min'] = (depth - depth.shift(1)) / 5
    frame['slope_10min_cm_per_min'] = (depth - depth.shift(2)) / 10
    for name in ('mean', 'min', 'max', 'std'):
        frame[f'{name}_three_bins_cm'] = getattr(rolling, name)()
    return frame[FEATURES]


def load_checked_model():
    manifest = json.loads((BUNDLE / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name in ('model/raw_delta_d3_weight20.ubj', 'model/config_before_fit.json'):
        if sha256(BUNDLE / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'Model/config checksum mismatch: {name}')
    config = json.loads((BUNDLE / 'model/config_before_fit.json').read_text(encoding='utf-8'))
    if config['features'] != FEATURES or config['fit_target'] != 'future proc label minus current RAW-DERIVED depth':
        raise ValueError('Unexpected model feature/target contract')
    model = XGBRegressor(n_jobs=4)
    model.load_model(BUNDLE / 'model/raw_delta_d3_weight20.ubj')
    model.set_params(n_jobs=4)
    if model.get_booster().feature_names != FEATURES:
        raise ValueError('Saved model feature order does not match raw inference')
    return model, config


def predict(observations, as_of, station):
    cutoff = checked_time(as_of)
    decision = cutoff.floor('5min')
    features = build_features(observations, cutoff)
    result = {'status': 'unavailable', 'mode': 'OFFLINE_RESEARCH_ONLY', 'station': station,
        'as_of_utc': cutoff.isoformat(), 'decision_time_utc': decision.isoformat(),
        'model_id': 'raw_delta_d3_weight20', 'target_bin_start_exclusive_utc': (decision + pd.Timedelta(minutes=5)).isoformat(),
        'target_bin_end_inclusive_utc': (decision + pd.Timedelta(minutes=10)).isoformat(),
        'warning': 'Archive observation clock only; not validated live ESP32 forecasting or road-safety advice.'}
    if decision not in features.index:
        return {**result, 'reason': 'no_completed_observed_bin'}
    row = features.loc[[decision]]
    missing = [name for name in FEATURES if not np.isfinite(row[name].iloc[0])]
    if missing:
        return {**result, 'reason': 'missing_raw_history_or_nonfinite_features', 'missing_features': missing}
    model, config = load_checked_model()
    delta = float(model.predict(row)[0])
    if not np.isfinite(delta):
        raise ValueError('Nonfinite model prediction')
    current = float(row.depth_lag_0_cm.iloc[0])
    recorded = config['packages']
    actual = {name: importlib.metadata.version(name) for name in ('pandas', 'numpy', 'xgboost')}
    return {**result, 'status': 'ok', 'current_depth_cm': current, 'predicted_delta_cm': delta,
        'predicted_depth_cm': max(0., current + delta), 'persistence_depth_cm': current,
        'current_bin_source_age_seconds': float(row.source_age_lag_0_seconds.iloc[0]),
        'runtime_packages': actual, 'runtime_matches_recorded_packages': all(actual[k] == recorded[k] for k in actual)}


def read_input(path, station=None):
    if Path(path).suffix.lower() == '.json':
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value, list):
            raise ValueError('JSON input must be an array of observation objects')
        frame = pd.DataFrame(value)
    else:
        frame = pd.read_csv(path, usecols=lambda name: name in ('time', 'depth_raw_mm', 'deployment_id'))
    if not {'time', 'depth_raw_mm'}.issubset(frame.columns):
        raise ValueError('CSV requires time and depth_raw_mm columns (mm, not cm)')
    if 'deployment_id' in frame:
        if frame.deployment_id.isna().any():
            raise ValueError('Null deployment_id; refuse ambiguous station history')
        identities = frame.deployment_id.astype(str).unique()
        if station is None:
            if len(identities) != 1:
                raise ValueError('Multiple/zero stations; choose --station explicitly')
            station = identities[0]
        frame = frame.loc[frame.deployment_id.astype(str).eq(station)].copy()
    elif not station:
        raise ValueError('CSV has no deployment_id; provide --station to label the single-station input')
    if frame.empty:
        raise ValueError('No rows for requested station')
    text = frame.time.astype(str)
    if not text.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', regex=True).all():
        raise ValueError('All CSV times must include a timezone')
    frame.index = pd.to_datetime(text, format='mixed', utc=True, errors='raise')
    if frame.index.hasnans:
        raise ValueError('Invalid CSV timestamps')
    return frame[['depth_raw_mm']], station


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--station')
    parser.add_argument('--as-of')
    args = parser.parse_args()
    try:
        frame, station = read_input(args.input, args.station)
        output = predict(frame, args.as_of if args.as_of else frame.index.max(), station)
        print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({'status': 'error', 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
