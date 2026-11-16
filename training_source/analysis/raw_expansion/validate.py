"""Rebuild candidate train data and replay both arms on the frozen validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from delta_experiment import METRIC, depth_from_delta, read_json
from five_minute.pipeline import TRAIN_END, EMBARGO, month_key
from raw_branch.experiment import load_raw_sources
from raw_branch.validate import compare_metrics
from raw_expansion.experiment import REVIEW, candidate_train_manifest, load_frozen_split, load_prepared
from raw_expansion.selection import CANDIDATES, RULES, build_candidate_train, select_overall
from raw_expansion.report import write_report
from train_baseline_xgboost import DATA, baseline_predictions, choose, digest, evaluate, json_write


def validate(output):
    before, station_decision, core, train = load_prepared(output)
    config = read_json(output / 'config_before_fit.json')
    summary = read_json(output / 'summary.json')
    decision = read_json(output / 'selection_validation_only.json')
    for key, value in before.items():
        assert config[key] == value, key
    assert digest(output / 'station_selection_before_fit.json') == config['station_selection_sha256']
    assert digest(output / 'candidate_window_provenance.json') == config['candidate_windows_manifest_sha256']
    assert digest(REVIEW) == config['review_table_sha256']
    assert config['candidate_stations'] == list(CANDIDATES)
    assert config['coverage_rules_train_only'] == RULES
    assert summary['test_loaded'] is False and decision['test_loaded'] is False
    assert summary['new_station_validation_loaded'] is False
    assert decision['station_selection_used_validation'] is False
    provenance = read_json(output / 'candidate_window_provenance.json')
    assert len(provenance) == 7 and {r['station'] for r in provenance} == set(CANDIDATES)
    assert {r['split'] for r in provenance} == {'train'}
    assert not list((output / 'candidate_train').glob('*_validation.parquet'))
    assert not list((output / 'candidate_train').glob('*_test.parquet'))
    manifest = candidate_train_manifest(read_json(DATA / 'selection_manifest.json'))
    sources, csv_files, cache_files = load_raw_sources(manifest, list(CANDIDATES))
    expected_source = read_json(output / 'source_provenance.json')
    assert expected_source == {'csv_fingerprints': csv_files, 'cache_files': cache_files}
    for row in csv_files:
        year, month = month_key(Path(row['path']))
        assert pd.Timestamp(year=year, month=month, day=1, tz='UTC') < TRAIN_END
    features, label = config['features'], config['label']
    assert label not in features and 'deployment_id' not in features
    assert not any(c.startswith(('audit_', 'review_')) for c in features)
    audits, reasons = [], []
    for station in CANDIDATES:
        source = sources[station].loc[sources[station].index < TRAIN_END - EMBARGO]
        expected, audit, exclusion = build_candidate_train(station, source, features, label)
        row = next(r for r in provenance if r['station'] == station)
        stored = pd.read_parquet(row['path'])
        pd.testing.assert_frame_equal(expected, stored, check_freq=False)
        assert len(stored) == row['rows']
        assert row['selected'] == audit['selected_for_exploratory_expansion']
        audits.append(audit)
        reasons.extend(exclusion)
    assert audits == station_decision['station_audits_train_only'] == summary['station_audits_train_only']
    selected_stations = [r['station'] for r in audits if r['selected_for_exploratory_expansion']]
    assert selected_stations == station_decision['selected_additions'] == summary['selected_additions'] == config['selected_additions']
    audit_csv = pd.read_csv(output / 'station_audit_train.csv')
    audit_csv['failed_criteria'] = audit_csv.failed_criteria.fillna('')
    expected_csv = pd.DataFrame([{**r, 'failed_criteria': '|'.join(r['failed_criteria'])} for r in audits])
    pd.testing.assert_frame_equal(expected_csv, audit_csv, check_dtype=False)
    a = pd.DataFrame(reasons).set_index(['station', 'reason']).sort_index()
    b = pd.read_csv(output / 'candidate_exclusion_reasons.csv').set_index(['station', 'reason']).sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    base = Path(config['base_raw_run'])
    validation = load_frozen_split(base, 'validation')
    expected_counts = {'core_train': len(core), 'added_train': len(train) - len(core),
        'expanded_train': len(train), 'validation': len(validation),
        'core_train_wet': int(core[label].ge(10).sum()), 'expanded_train_wet': int(train[label].ge(10).sum())}
    assert expected_counts == config['counts'] == summary['counts']
    assert set(train.deployment_id.unique()) == set(config['core_stations']) | set(selected_stations)
    assert np.isfinite(train[features]).all().all()
    assert set(validation.deployment_id.unique()).issubset(config['core_stations'])
    stored = pd.read_parquet(output / 'validation_predictions_local.parquet')
    assert stored.index.equals(validation.index)
    frozen_stored = pd.read_parquet(base / 'validation_predictions_local.parquet')
    assert stored.index.equals(frozen_stored.index)
    for column in ('deployment_id', label, 'depth_lag_0_cm'):
        assert stored[column].equals(validation[column]) and stored[column].equals(frozen_stored[column]), column
    predictions = baseline_predictions(validation)
    for candidate in config['candidate_model_configs']:
        for arm in ('three', 'expanded'):
            name = arm + '_' + candidate['id']
            path = base / f'raw_delta_{candidate["id"]}.ubj' if arm == 'three' else output / f'{name}.ubj'
            model = XGBRegressor()
            model.load_model(path)
            assert model.get_booster().feature_names == features
            assert model.get_booster().num_boosted_rounds() == candidate['n_estimators']
            predictions[name] = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[features]))
    metrics = pd.read_csv(output / 'metrics.csv')
    assert len(metrics) == 10 and set(metrics.model) == set(predictions)
    actual_rows = []
    for name, prediction in predictions.items():
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
        np.testing.assert_allclose(stored[f'prediction_{name}_cm'], prediction, rtol=1e-6, atol=1e-6)
        actual = evaluate(validation[label], prediction, validation.depth_lag_0_cm)
        expected = next(r for r in summary['validation_metrics'] if r['model'] == name)
        compare_metrics(actual, expected, metrics.loc[metrics.model.eq(name)].iloc[0])
        arm = 'three' if name.startswith('three_') else 'expanded' if name.startswith('expanded_') else 'baseline'
        actual_rows.append({'model': name, 'arm': arm, **actual})
    best_three = choose([r for r in actual_rows if r['arm'] == 'three'])
    best_expanded = choose([r for r in actual_rows if r['arm'] == 'expanded'])
    selected = select_overall(actual_rows)
    for key, value in (('best_three', best_three), ('best_expanded', best_expanded), ('selected_forecaster', selected)):
        assert summary[key] == decision[key] == value
    assert summary['validation_metrics'] == decision['validation_metrics']
    by_name = {r['model']: r for r in actual_rows}
    change = by_name[best_expanded][METRIC] - by_name[best_three][METRIC]
    np.testing.assert_allclose(change, summary['expanded_vs_three_balanced_mae_change_cm'])
    paired = pd.read_csv(output / 'matched_config_comparison.csv')
    for candidate in config['candidate_model_configs']:
        a, b = by_name['three_' + candidate['id']], by_name['expanded_' + candidate['id']]
        row = paired.loc[paired.config.eq(candidate['id'])].iloc[0]
        expected = next(r for r in summary['matched_config_comparison'] if r['config'] == candidate['id'])
        for key, value in {'three_balanced_mae_cm': a[METRIC], 'expanded_balanced_mae_cm': b[METRIC],
            'change_cm': b[METRIC] - a[METRIC], 'three_fp': a['fp'], 'expanded_fp': b['fp'],
            'three_fn': a['fn'], 'expanded_fn': b['fn']}.items():
            np.testing.assert_allclose(value, row[key], rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(value, expected[key], rtol=1e-8, atol=1e-8)
    breakdown = pd.read_csv(output / 'validation_breakdown.csv')
    for station in sorted(validation.deployment_id.unique()):
        mask = validation.deployment_id.eq(station).to_numpy()
        for name, prediction in predictions.items():
            actual = evaluate(validation[label].to_numpy()[mask], prediction[mask], validation.depth_lag_0_cm.to_numpy()[mask])
            row = breakdown.loc[breakdown.station.eq(station) & breakdown.model.eq(name)].iloc[0]
            expected = {key: None if pd.isna(row[key]) else row[key] for key in actual}
            compare_metrics(actual, expected, row)
    for row in read_json(output / 'frozen_input_provenance.json'):
        assert digest(Path(row['path'])) == row['sha256'], row['path']
    names = ['config_before_preparation.json', 'config_before_fit.json', 'station_selection_before_fit.json',
        'frozen_input_provenance.json', 'source_provenance.json', 'candidate_window_provenance.json',
        'station_audit_train.csv', 'candidate_exclusion_reasons.csv', 'selection_validation_only.json',
        'summary.json', 'metrics.csv', 'validation_predictions_local.parquet', 'validation_breakdown.csv',
        'matched_config_comparison.csv']
    names += [f'expanded_{c["id"]}.ubj' for c in config['candidate_model_configs']]
    result = {'status': 'PASS', 'train_counts': expected_counts, 'selected_additions': selected_stations,
        'source_csv_fingerprints_verified': len(csv_files), 'source_cache_checksums_verified': len(cache_files),
        'candidate_train_files_rebuilt': len(provenance), 'validation_methods_repredicted': len(predictions),
        'selected_forecaster': selected, 'expanded_vs_three_balanced_mae_change_cm': change,
        'test_loaded': False, 'new_station_validation_loaded': False,
        'checks': ['seven candidate train windows/audits/reasons rebuilt from unchanged pre-2023 source caches',
            'selection reproduces from train coverage only; no offline quality masks',
            'core train and validation remain identical to frozen raw three-point arm',
            'all 8 saved delta models and 2 baselines reproduce all 92215 validation predictions/metrics',
            'per-station breakdown and same-config comparisons reproduce',
            'old model may win; expanded data are not automatically promoted',
            'all frozen original input/model checksums unchanged'],
        'artifact_sha256': {name: digest(output / name) for name in names},
        'limits': 'Technical reproducibility only, not clean-label certification, new-station generalization or live-warning approval.'}
    json_write(output / 'artifact_validation.json', result)
    write_report(output, summary)
    print(json.dumps({key: value for key, value in result.items() if key != 'artifact_sha256'}, indent=2))
    return result
