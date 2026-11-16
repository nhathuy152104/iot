"""Recompute all delta validation predictions/metrics without fitting or test I/O."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from delta_experiment import (
    METRIC, depth_from_delta, original_config, read_json, select_forecaster, verified_split,
)
from train_baseline_xgboost import BASE, DATA, baseline_predictions, choose, digest, evaluate, json_write


def validate(output, require_improvement=False):
    before = read_json(output / 'config_before_fit.json')
    summary = read_json(output / 'summary.json')
    decision = read_json(output / 'selection_validation_only.json')
    provenance = read_json(output / 'input_provenance.json')
    origin = Path(before['origin'])
    original, manifest = original_config(origin)
    assert original == before['original']
    assert digest(origin / 'config_before_fit.json') == before['origin_config_sha256']
    assert digest(origin / 'xgboost_selected.ubj') == before['origin_model_sha256']
    assert digest(BASE / 'delta_experiment.py') == before['script_sha256']
    assert digest(BASE / 'train_baseline_xgboost.py') == before['reused_training_script_sha256']
    assert digest(DATA / 'pipeline.py') == before['reused_pipeline_sha256']
    assert {r['split'] for r in provenance} == {'train', 'validation'}
    assert summary['test_loaded'] is False and decision['test_loaded'] is False
    assert not list(output.glob('*test*.parquet'))
    expected_sources = [r for r in read_json(origin / 'input_provenance.json') if r['split'] != 'test']
    assert provenance == expected_sources
    for row in provenance:
        assert digest(Path(row['path'])) == row['sha256'], row['path']
    actual_sources = []
    frame = verified_split('validation', origin, manifest, actual_sources)
    stored = pd.read_parquet(output / 'validation_predictions_local.parquet')
    assert stored.index.equals(frame.index)
    for column in ('deployment_id', original['label'], 'depth_lag_0_cm'):
        assert stored[column].equals(frame[column]), column
    features = original['features']
    assert original['label'] not in features and 'deployment_id' not in features
    assert not any(c.startswith(('review_', 'audit_')) for c in features)
    predictions = baseline_predictions(frame)
    old = XGBRegressor()
    old.load_model(origin / 'xgboost_selected.ubj')
    predictions['original_absolute_selected'] = np.maximum(old.predict(frame[features]), 0)
    for candidate in original['candidate_configs']:
        name = 'delta_' + candidate['id']
        model = XGBRegressor()
        model.load_model(output / f'{name}.ubj')
        assert model.get_booster().feature_names == features
        predictions[name] = depth_from_delta(frame.depth_lag_0_cm, model.predict(frame[features]))
    actual_rows = []
    metrics = pd.read_csv(output / 'metrics.csv')
    assert len(metrics) == len(predictions) and set(metrics.split) == {'validation'}
    for name, prediction in predictions.items():
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
        np.testing.assert_allclose(stored[f'prediction_{name}_cm'], prediction, rtol=1e-6, atol=1e-6)
        actual = evaluate(frame[original['label']], prediction, frame.depth_lag_0_cm)
        expected = next(r for r in summary['validation_metrics'] if r['model'] == name)
        csv_row = metrics.loc[metrics.model.eq(name)].iloc[0]
        for key, value in actual.items():
            if value is None:
                assert expected[key] is None and pd.isna(csv_row[key]), (name, key)
            else:
                np.testing.assert_allclose(value, expected[key], rtol=1e-8, atol=1e-8)
                np.testing.assert_allclose(value, csv_row[key], rtol=1e-8, atol=1e-8)
        actual_rows.append({'model': name, **actual})
    best_delta = choose([r for r in actual_rows if r['model'].startswith('delta_')])
    selected = select_forecaster(actual_rows)
    assert best_delta == summary['best_delta'] == decision['best_delta']
    assert selected == summary['selected_forecaster'] == decision['selected_forecaster']
    assert decision['validation_candidates'] == summary['validation_metrics']
    best = next(r for r in actual_rows if r['model'] == best_delta)
    persistence = next(r for r in actual_rows if r['model'] == 'persistence')
    improvement = best[METRIC] < persistence[METRIC]
    assert improvement == summary['delta_beats_persistence_balanced_validation']
    np.testing.assert_allclose(best[METRIC] - persistence[METRIC], summary['balanced_mae_change_vs_persistence_cm'])
    assert len(frame) == summary['counts']['validation']
    assert sum(r['rows'] for r in provenance if r['split'] == 'train') == summary['counts']['train']
    if require_improvement:
        assert improvement, 'Delta does not improve balanced validation MAE over persistence'
    artifact_names = ['config_before_fit.json', 'summary.json', 'selection_validation_only.json',
        'metrics.csv', 'validation_predictions_local.parquet', 'input_provenance.json']
    artifact_names += [f'delta_{c["id"]}.ubj' for c in original['candidate_configs']]
    result = {'status': 'PASS', 'validation_rows_repredicted_per_model': len(frame),
        'models_repredicted': len(predictions), 'original_source_files_verified': len(provenance),
        'selected_forecaster': selected, 'test_loaded': False,
        'delta_balanced_validation_improvement_cm': persistence[METRIC] - best[METRIC],
        'checks': ['all 4 saved delta models reproduce all validation predictions',
            'frozen absolute model and both baselines reproduce', 'all metrics reproduce in JSON and CSV',
            'validation-only selection reproduces including baselines',
            'input files are identical to original train/validation', 'no new test evaluation'],
        'artifact_sha256': {name: digest(output / name) for name in artifact_names},
        'limits': 'Technical reproducibility only; not independent confirmation, causal-filter verification or operational approval.'}
    json_write(output / 'artifact_validation.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--require-improvement', action='store_true')
    args = parser.parse_args()
    validate(args.run_directory, args.require_improvement)
