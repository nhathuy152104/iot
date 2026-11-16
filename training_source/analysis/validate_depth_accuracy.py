"""Replay the fixed depth-objective trial; read-only, no train/test table loading."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from archive_paths import resolve_archive_path
from delta_experiment import depth_from_delta, read_json
from depth_accuracy import error_profile
from diagnose_raw_expansion import EXPANDED, verified_windows
from train_baseline_xgboost import baseline_predictions, digest, evaluate


def validate(output):
    check = read_json(output / 'artifact_validation.json')
    config = read_json(output / 'config_before_fit.json')
    for name, checksum in check['output_sha256'].items():
        assert digest(output / name) == checksum, name
    for name, checksum in check['frozen_sha256'].items():
        assert digest(Path(name)) == checksum, name
    core = resolve_archive_path(read_json(EXPANDED / 'config_before_fit.json')['base_raw_run'])
    frame = verified_windows(core / 'window_provenance.json', 'validation', {})
    predictions = baseline_predictions(frame)
    for name, path in [('three_d3_weight20', core / 'raw_delta_d3_weight20.ubj'),
                       ('trial_absolute_error', output / 'trial_absolute_error.ubj')]:
        model = XGBRegressor()
        model.load_model(path)
        predictions[name] = depth_from_delta(frame.depth_lag_0_cm, model.predict(frame[config['features']]))
    stored = pd.read_parquet(output / 'validation_predictions_local.parquet')
    assert stored.index.equals(frame.index) and stored.deployment_id.equals(frame.deployment_id)
    metrics = pd.read_csv(output / 'metrics.csv').set_index('model')
    profiles = pd.read_csv(output / 'depth_error_profiles.csv').set_index(['model', 'group'])
    y = frame[config['label']].to_numpy()
    for name, p in predictions.items():
        np.testing.assert_allclose(p, stored['prediction_' + name + '_cm'], atol=1e-6, rtol=1e-6)
        for key, value in evaluate(y, p, frame.depth_lag_0_cm).items():
            if value is not None:
                np.testing.assert_allclose(value, metrics.loc[name, key], atol=1e-8, rtol=1e-8)
        for group, mask in {'all': np.ones(len(y), bool), 'target_ge10': y >= 10,
                            'target_8_to12': (y >= 8) & (y < 12),
                            'prediction_8_to12': (p >= 8) & (p < 12)}.items():
            for key, value in error_profile(y[mask], p[mask]).items():
                if value is not None:
                    np.testing.assert_allclose(value, profiles.loc[(name, group), key], atol=1e-8, rtol=1e-8)
    print('PASS: four prediction methods, all metrics/error profiles replayed; frozen files/output checksums unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    validate(parser.parse_args().run_directory.resolve())
