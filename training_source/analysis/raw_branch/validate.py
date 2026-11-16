"""Rebuild raw windows and independently replay saved models; never refit/read test."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from delta_experiment import depth_from_delta, original_config, read_json, select_forecaster, verified_split
from five_minute.pipeline import VALIDATION_END, month_key, split_masks
from raw_branch.experiment import load_raw_sources, paired_comparison
from raw_branch.pipeline import eligibility_reasons, make_raw_features, make_raw_grid, membership_summary
from raw_branch.report import write_report
from train_baseline_xgboost import BASE, DATA, baseline_predictions, choose, digest, evaluate, json_write


def compare_metrics(actual, expected, csv_row=None):
    for key, value in actual.items():
        if value is None:
            assert expected[key] is None, key
            if csv_row is not None:
                assert pd.isna(csv_row[key]), key
        else:
            np.testing.assert_allclose(value, expected[key], rtol=1e-8, atol=1e-8)
            if csv_row is not None:
                np.testing.assert_allclose(value, csv_row[key], rtol=1e-8, atol=1e-8)


def validate(output):
    config = read_json(output / 'config_before_fit.json')
    before = read_json(output / 'config_before_preparation.json')
    summary = read_json(output / 'summary.json')
    decision = read_json(output / 'selection_validation_only.json')
    for key, value in before.items():
        assert config[key] == value, key
    for path, checksum in config['code_sha256'].items():
        assert digest(BASE / path) == checksum, path
    assert digest(DATA / 'selection_manifest.json') == config['source_manifest_sha256']
    origin = Path(config['origin'])
    original, old_manifest = original_config(origin)
    assert original == config['original']
    assert digest(origin / 'config_before_fit.json') == config['origin_config_sha256']
    proc_origin = Path(config['proc_delta_origin'])
    assert digest(proc_origin / 'config_before_fit.json') == config['proc_delta_config_sha256']
    proc_name = read_json(proc_origin / 'summary.json')['best_delta']
    assert digest(proc_origin / f'{proc_name}.ubj') == config['proc_delta_model_sha256']
    assert summary['test_loaded'] is False and decision['test_loaded'] is False
    assert decision['paired_metrics_used_for_selection'] is False
    assert summary['paired_metrics_used_for_selection'] is False
    assert not list((output / 'windows').glob('*_test.parquet'))
    old_provenance = []
    old = {s: verified_split(s, origin, old_manifest, old_provenance) for s in ('train', 'validation')}
    assert old_provenance == read_json(output / 'proc_window_provenance.json')
    manifest = read_json(DATA / 'selection_manifest.json')
    sources, csv_files, cache_files = load_raw_sources(manifest, original['stations'])
    expected_source = read_json(output / 'source_provenance.json')
    assert csv_files == expected_source['csv_fingerprints']
    assert cache_files == expected_source['cache_files']
    for row in csv_files:
        year, month = month_key(Path(row['path']))
        assert pd.Timestamp(year=year, month=month, day=1, tz='UTC') < VALIDATION_END
    window_provenance = read_json(output / 'window_provenance.json')
    assert digest(output / 'window_provenance.json') == config['window_provenance_sha256']
    assert len(window_provenance) == 6
    assert set(r['station'] for r in window_provenance) == set(original['stations'])
    assert set(r['split'] for r in window_provenance) == {'train', 'validation'}
    for row in window_provenance:
        assert digest(Path(row['path'])) == row['sha256']
    features, label = config['features'], config['label']
    assert features == original['features'] and label not in features
    assert not any(c.startswith(('audit_', 'review_')) or c == 'deployment_id' for c in features)
    parts = {'train': [], 'validation': []}
    expected_coverage, expected_reasons, expected_bins = [], [], []
    members = pd.read_parquet(output / 'window_membership_local.parquet')
    prefix_records = read_json(output / 'prefix_causality_checks.json')
    for station, source in sources.items():
        grid = make_raw_grid(source)
        frame = make_raw_features(grid, station, label)
        reasons = eligibility_reasons(frame, label, features)
        masks = split_masks(frame.index)
        prefix_record = next(r for r in prefix_records if r['station'] == station)
        prefix = make_raw_features(make_raw_grid(source, as_of=pd.Timestamp(prefix_record['cutoff_utc'])), station, label)
        pd.testing.assert_frame_equal(frame.loc[prefix.index, features], prefix[features])
        assert len(prefix) == prefix_record['feature_rows_checked']
        for split in ('train', 'validation'):
            expected = frame.loc[masks[split] & reasons.eq('kept')]
            provenance = next(r for r in window_provenance if r['station'] == station and r['split'] == split)
            stored = pd.read_parquet(provenance['path'])
            # Parquet preserves timestamps but not DatetimeIndex.freq metadata.
            pd.testing.assert_frame_equal(expected, stored, check_freq=False)
            assert len(stored) == provenance['rows']
            assert split_masks(stored.index)[split].all()
            assert np.isfinite(stored[features]).all().all()
            parts[split].append(stored)
            previous = old[split].loc[old[split].deployment_id.eq(station)]
            expected_coverage.append(membership_summary(station, split, stored, previous, reasons, label))
            for reason, count in reasons.loc[masks[split]].value_counts().items():
                expected_reasons.append({'station': station, 'split': split, 'reason': reason, 'windows': int(count)})
            current = grid.loc[masks[split]]
            expected_bins.append({'station': station, 'split': split,
                'decision_bins_in_support_range': len(current),
                'observed_bins': int(current.audit_bin_observed.sum()),
                'empty_bins': int((~current.audit_bin_observed).sum()),
                'last_raw_nonfinite_bins': int(current.audit_last_raw_missing.sum()),
                'raw_input_missing_bins': int(current.raw_median_cm.isna().sum()),
                'source_rows': int(current.audit_source_rows.sum()),
                'finite_raw_rows': int(current.audit_finite_raw_rows.sum()),
                'negative_raw_rows_floored': int(current.audit_negative_raw_rows.sum()),
                'raw_ge_100cm_rows_retained': int(current.audit_high_raw_ge_100cm_rows.sum()),
                'duplicate_source_times_first_kept_entire_station': int(source.index.duplicated().sum())})
            union = stored.index.union(previous.index)
            member = members.loc[members.deployment_id.eq(station) & members.split.eq(split)]
            assert member.index.equals(union)
            np.testing.assert_array_equal(member.raw_eligible, union.isin(stored.index))
            np.testing.assert_array_equal(member.proc_eligible, union.isin(previous.index))
            np.testing.assert_array_equal(member.raw_reason, reasons.reindex(union))
            np.testing.assert_allclose(member[label], frame[label].reindex(union), equal_nan=True)
    for actual, filename, keys in (
        (expected_coverage, 'window_coverage.csv', ['station', 'split']),
        (expected_reasons, 'exclusion_reasons.csv', ['station', 'split', 'reason']),
        (expected_bins, 'raw_bin_audit.csv', ['station', 'split']),
    ):
        a = pd.DataFrame(actual).set_index(keys).sort_index().sort_index(axis=1)
        b = pd.read_csv(output / filename).set_index(keys).sort_index().sort_index(axis=1)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    raw = {split: pd.concat(frames).sort_index(kind='stable') for split, frames in parts.items()}
    assert {split: len(f) for split, f in raw.items()} == config['counts'] == summary['counts']
    validation = raw['validation']
    stored = pd.read_parquet(output / 'validation_predictions_local.parquet')
    assert stored.index.equals(validation.index)
    for column in ('deployment_id', label, 'depth_lag_0_cm', 'audit_negative_raw_history', 'audit_sparse_raw_history'):
        assert stored[column].equals(validation[column]), column
    predictions = baseline_predictions(validation)
    models = {}
    for candidate in config['candidate_configs']:
        name = 'raw_delta_' + candidate['id']
        model = XGBRegressor()
        model.load_model(output / f'{name}.ubj')
        assert model.get_booster().feature_names == features
        predictions[name] = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[features]))
        models[name] = model
    metrics = pd.read_csv(output / 'metrics.csv')
    assert len(metrics) == 6 and set(metrics.split) == {'validation_raw_eligible'}
    actual_rows = []
    for name, prediction in predictions.items():
        np.testing.assert_allclose(stored[f'prediction_{name}_cm'], prediction, rtol=1e-6, atol=1e-6)
        actual = evaluate(validation[label], prediction, validation.depth_lag_0_cm)
        expected = next(r for r in summary['validation_metrics'] if r['model'] == name)
        csv_row = metrics.loc[metrics.model.eq(name)].iloc[0]
        compare_metrics(actual, expected, csv_row)
        actual_rows.append({'model': name, **actual})
    best_delta = choose([r for r in actual_rows if r['model'].startswith('raw_delta_')])
    selected = select_forecaster(actual_rows)
    assert best_delta == summary['best_delta'] == decision['best_delta']
    assert selected == summary['selected_forecaster'] == decision['selected_forecaster']
    assert decision['validation_candidates'] == summary['validation_metrics']
    paired, paired_stored = paired_comparison(validation, old['validation'], models[best_delta], original, proc_origin)
    pd.testing.assert_frame_equal(paired_stored, pd.read_parquet(output / 'paired_validation_predictions_local.parquet'), check_freq=False)
    paired_csv = pd.read_csv(output / 'paired_validation_metrics.csv')
    for row in paired:
        expected = next(r for r in summary['paired_validation_metrics'] if r['model'] == row['model'])
        compare_metrics({k: v for k, v in row.items() if k not in ('model', 'split')}, expected,
                        paired_csv.loc[paired_csv.model.eq(row['model'])].iloc[0])
    artifact_names = ['config_before_preparation.json', 'config_before_fit.json', 'selection_validation_only.json',
        'summary.json', 'metrics.csv', 'window_coverage.csv', 'exclusion_reasons.csv', 'raw_bin_audit.csv',
        'source_provenance.json', 'proc_window_provenance.json', 'window_provenance.json',
        'window_membership_local.parquet', 'validation_predictions_local.parquet',
        'paired_validation_metrics.csv', 'paired_validation_predictions_local.parquet', 'prefix_causality_checks.json']
    artifact_names += [f'raw_delta_{c["id"]}.ubj' for c in config['candidate_configs']]
    result = {'status': 'PASS', 'raw_windows_rebuilt': summary['counts'],
        'source_csv_fingerprints_verified': len(csv_files), 'source_cache_checksums_verified': len(cache_files),
        'raw_window_files_verified': len(window_provenance), 'original_proc_window_files_verified': len(old_provenance),
        'methods_repredicted_on_validation': len(predictions), 'paired_methods_repredicted': len(paired),
        'selected_forecaster': selected, 'test_loaded': False,
        'checks': ['all raw windows reconstructed from checksummed source caches',
            'whole-source/prefix features equal at original train cutoffs',
            'all coverage, exclusion reasons and memberships reproduce',
            'all 4 saved delta models and 2 baselines reproduce every validation prediction/metric',
            'paired proc/raw labels, predictions and metrics reproduce',
            'selection uses full raw validation, not common-subset scores',
            'original data/config/code fingerprints match; source cache months precede July 2023'],
        'artifact_sha256': {name: digest(output / name) for name in artifact_names},
        'limits': 'Technical reproducibility only; not true-label, arrival-time, calibration, live-warning or transfer validation.'}
    json_write(output / 'artifact_validation.json', result)
    write_report(output, summary)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, indent=2))
    return result
