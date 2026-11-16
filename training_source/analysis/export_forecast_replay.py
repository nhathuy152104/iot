"""Export small LOCAL historical map replays, never into frontend/public.

Only frozen three-core raw model is exported, not the new MAE trial. No future
labels/observations are included in browser frames; diagnostics are metadata.
"""
import datetime as dt

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from archive_paths import ROOT, resolve_archive_path
from delta_experiment import depth_from_delta, read_json
from depth_accuracy import error_profile
from diagnose_raw_expansion import EXPANDED, verified_windows
from train_baseline_xgboost import BASE, digest, evaluate, json_write

CASES = [('ugliest_cyan_elephant', '2023-04-23', 'Diễn biến tăng/rút'),
         ('ugliest_cyan_elephant', '2023-06-02', 'Giữ mức cao và chuyển mức'),
         ('daily_new_falcon', '2023-04-30', 'Đối chiếu trạm thứ hai')]


def station_metadata(stations):
    """Read identity/location only from a pre-test source month; no geocoding."""
    source = ROOT / 'floodnet-flood-sensor-jan-2023.csv'
    stat = source.stat()
    found = {}
    with pd.read_csv(source, usecols=['deployment_id', 'lat', 'lon', 'sensor_display_name'], chunksize=10000) as reader:
        for chunk in reader:
            for row in chunk.loc[chunk.deployment_id.isin(stations)].itertuples(index=False):
                if row.deployment_id in found:
                    continue
                lat, lon = float(row.lat), float(row.lon)
                if not np.isfinite([lat, lon]).all() or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError('Invalid source location; refuse to invent coordinates')
                found[row.deployment_id] = {'id': row.deployment_id,
                    'name': str(row.sensor_display_name) if pd.notna(row.sensor_display_name) else row.deployment_id,
                    'latitude': lat, 'longitude': lon}
            if set(found) == set(stations):
                break
    if set(found) != set(stations):
        raise ValueError('Missing station metadata')
    return found, {'path': str(source), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                   'purpose': 'deployment identity/location columns only; fingerprint, not full CSV hash'}


def scenario_frames(frame, prediction, station, day):
    start = pd.Timestamp(day, tz='UTC')
    times = pd.date_range(start, periods=288, freq='5min')
    current = frame.depth_lag_0_cm.loc[frame.deployment_id.eq(station)].reindex(times)
    pred = pd.Series(prediction, index=frame.index).loc[frame.deployment_id.eq(station)].reindex(times)
    frames = []
    for at, now, future in zip(times, current, pred):
        valid = np.isfinite(now) and np.isfinite(future)
        frames.append({'decision_at': at.isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            'current_depth_cm': float(now) if valid else None,
            'predicted_depth_cm': float(future) if valid else None,
            'baseline_depth_cm': float(now) if valid else None})
    return frames


def run():
    config = read_json(EXPANDED / 'config_before_fit.json')
    core_run = resolve_archive_path(config['base_raw_run'])
    provenance = {}
    validation = verified_windows(core_run / 'window_provenance.json', 'validation', provenance)
    path = core_run / 'raw_delta_d3_weight20.ubj'
    model = XGBRegressor()
    model.load_model(path)
    prediction = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[config['features']]))
    expected = pd.read_parquet(core_run / 'validation_predictions_local.parquet')
    assert expected.index.equals(validation.index)
    assert expected.deployment_id.equals(validation.deployment_id)
    np.testing.assert_allclose(prediction, expected.prediction_raw_delta_d3_weight20_cm, rtol=1e-6, atol=1e-6)
    metrics = evaluate(validation[config['label']], prediction, validation.depth_lag_0_cm)
    near = (prediction >= 8) & (prediction < 12)
    errors = error_profile(validation[config['label']].to_numpy()[near], prediction[near])
    metadata, source_fingerprint = station_metadata({case[0] for case in CASES})
    bundle = {'schema_version': 1, 'mode': 'historical_replay', 'model_id': 'raw_delta_d3_weight20',
        'source': 'FloodNet NYC — New York University / City University of New York; local research derivative',
        'validation': {'windows': len(validation), 'balanced_mae_cm': metrics['balanced_wet_nonwet_mae_cm'],
            'near_prediction_p90_error_cm': errors['p90_abs_error_cm'] if errors['windows'] >= 30 else None},
        'scenarios': [{'id': station + '_' + day, 'title': day + ' · ' + title, 'station': metadata[station],
            'frames': scenario_frames(validation, prediction, station, day)} for station, day, title in CASES]}
    output = BASE / 'replay_exports' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    file = output / 'forecast-replay-local.json'
    json_write(file, bundle)
    json_write(output / 'export_provenance.json', {'status': 'LOCAL_ONLY_NOT_FOR_REDISTRIBUTION',
        'model_path': str(path), 'model_sha256': digest(path), 'windows': provenance,
        'source_metadata_fingerprint': source_fingerprint, 'export_sha256': digest(file),
        'limits': 'validation already inspected; illustrative dates not a representative performance test; no future labels in frame payload; not live prediction',
        'features_and_bin_policy': config['preprocessing']})
    print(f'LOCAL REPLAY EXPORTED: {file}', flush=True)
    return file


if __name__ == '__main__':
    run()
