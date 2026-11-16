"""Prepare and fit raw-input delta models using only the original three stations.

No test files, no new station selection, no modification of source/cache/old runs.
Proc is used for future labels and explicit paired retrospective comparisons.
"""
import datetime as dt
import importlib.metadata
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from delta_experiment import (
    DEFAULT_ORIGIN, METRIC, delta_target, depth_from_delta, make_model,
    original_config, read_json, select_forecaster, verified_split,
)
from five_minute.pipeline import TRAIN_END, VALIDATION_END, EMBARGO, month_key, split_masks
from raw_branch.pipeline import POLICY, eligibility_reasons, keyed, make_raw_features, make_raw_grid, membership_summary
from train_baseline_xgboost import BASE, DATA, baseline_predictions, choose, digest, evaluate, json_write

PROC_DELTA_ORIGIN = BASE / 'training_runs' / '20261110T040220_424848Z_delta'


def source_months(manifest):
    """Do not even read cache files from July 2023 onward."""
    paths = []
    for filename in manifest['source_files']:
        source = DATA.parents[1] / filename
        year, month = month_key(source)
        if pd.Timestamp(year=year, month=month, day=1, tz='UTC') < VALIDATION_END:
            paths.append(source)
    return sorted(paths, key=month_key)


def load_raw_sources(manifest, stations):
    parts = {s: [] for s in stations}
    sources, provenance = [], []
    for source in source_months(manifest):
        stat = source.stat()
        candidates = []
        for marker_path in (DATA / 'local_cache').glob(f'{source.stem}_*/complete.json'):
            marker = read_json(marker_path)
            if (marker['size'] == stat.st_size and marker['mtime_ns'] == stat.st_mtime_ns
                    and marker['cache_schema'] == 1 and len(marker['station_ids']) == manifest['stations_scanned']
                    and set(stations).issubset(marker['station_ids'])):
                candidates.append(marker_path)
        if len(candidates) != 1:
            raise ValueError(f'Expected one complete matching existing cache for {source.name}; found {len(candidates)}')
        marker_path = candidates[0]
        sources.append({'path': str(source), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                        'cache_marker': str(marker_path), 'marker_sha256': digest(marker_path)})
        for station in stations:
            path = marker_path.parent / f'{station}.parquet'
            if not path.exists():
                continue  # Sensor not yet deployed/already retired in this month.
            frame = pd.read_parquet(path, columns=['depth_raw_mm', 'depth_proc_mm'])
            if frame.index.tz is None or frame.index.hasnans:
                raise ValueError(f'Invalid timestamps in {path}')
            frame = frame.loc[frame.index < VALIDATION_END]
            parts[station].append(frame)
            provenance.append({'station': station, 'path': str(path), 'sha256': digest(path), 'rows': len(frame)})
    result = {}
    for station, frames in parts.items():
        if not frames:
            raise ValueError(f'No source observations for {station}')
        result[station] = pd.concat(frames).sort_index(kind='stable')
    return result, sources, provenance


def prepare(output, original, old_splits, manifest):
    observations, source_files, cache_files = load_raw_sources(manifest, original['stations'])
    windows_dir = output / 'windows'
    windows_dir.mkdir()
    features, label = original['features'], original['label']
    if label in features or 'deployment_id' in features or any(c.startswith(('audit_', 'review_')) for c in features):
        raise ValueError('Feature whitelist contains forbidden metadata')
    raw_parts = {'train': [], 'validation': []}
    coverage, reason_rows, bin_rows, membership, window_files, prefix_checks = [], [], [], [], [], []
    for station, source in observations.items():
        duplicates = int(source.index.duplicated().sum())
        grid = make_raw_grid(source)
        frame = make_raw_features(grid, station, label)
        reasons = eligibility_reasons(frame, label, features)
        masks = split_masks(frame.index)
        # Verify whole-source versus prefix processing on a TRAIN cutoff only.
        train_index = grid.index[masks['train']]
        if train_index.empty:
            raise ValueError(f'No train grid for prefix replay: {station}')
        cutoff = train_index[len(train_index) // 2]
        prefix_grid = make_raw_grid(source, as_of=cutoff)
        prefix = make_raw_features(prefix_grid, station, label)
        pd.testing.assert_frame_equal(frame.loc[prefix.index, features], prefix[features])
        prefix_checks.append({'station': station, 'cutoff_utc': str(cutoff),
            'feature_rows_checked': len(prefix), 'status': 'PASS', 'checks_future_labels': False})
        for split in ('train', 'validation'):
            mask = masks[split]
            index = frame.index[mask]
            eligible = reasons.eq('kept') & mask
            windows = frame.loc[eligible].copy()
            assert np.isfinite(windows[features]).all().all()
            assert windows[label].ge(0).all()
            path = windows_dir / f'{station}_{split}.parquet'
            windows.to_parquet(path)
            window_files.append({'station': station, 'split': split, 'path': str(path),
                                 'rows': len(windows), 'sha256': digest(path)})
            raw_parts[split].append(windows)
            old = old_splits[split].loc[old_splits[split].deployment_id.eq(station)]
            coverage.append(membership_summary(station, split, windows, old, reasons, label))
            for reason, count in reasons.loc[index].value_counts().items():
                reason_rows.append({'station': station, 'split': split, 'reason': reason, 'windows': int(count)})
            current = grid.loc[index]
            bin_rows.append({'station': station, 'split': split, 'decision_bins_in_support_range': len(index),
                'observed_bins': int(current.audit_bin_observed.sum()),
                'empty_bins': int((~current.audit_bin_observed).sum()),
                'last_raw_nonfinite_bins': int(current.audit_last_raw_missing.sum()),
                'raw_input_missing_bins': int(current.raw_median_cm.isna().sum()),
                'source_rows': int(current.audit_source_rows.sum()),
                'finite_raw_rows': int(current.audit_finite_raw_rows.sum()),
                'negative_raw_rows_floored': int(current.audit_negative_raw_rows.sum()),
                'raw_ge_100cm_rows_retained': int(current.audit_high_raw_ge_100cm_rows.sum()),
                'duplicate_source_times_first_kept_entire_station': duplicates})
            union = windows.index.union(old.index)
            member = pd.DataFrame(index=union)
            member['deployment_id'] = station
            member['split'] = split
            member['raw_eligible'] = union.isin(windows.index)
            member['proc_eligible'] = union.isin(old.index)
            member['raw_reason'] = reasons.reindex(union)
            member[label] = frame[label].reindex(union)
            membership.append(member)
    raw = {split: pd.concat(parts).sort_index(kind='stable') for split, parts in raw_parts.items()}
    for split, frame in raw.items():
        if frame.empty:
            raise ValueError(f'No raw windows for {split}; do not auto-change split boundaries')
    pd.DataFrame(coverage).to_csv(output / 'window_coverage.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(reason_rows).to_csv(output / 'exclusion_reasons.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(bin_rows).to_csv(output / 'raw_bin_audit.csv', index=False, encoding='utf-8-sig')
    pd.concat(membership).to_parquet(output / 'window_membership_local.parquet')
    json_write(output / 'prefix_causality_checks.json', prefix_checks)
    json_write(output / 'source_provenance.json', {'csv_fingerprints': source_files, 'cache_files': cache_files})
    json_write(output / 'window_provenance.json', window_files)
    return raw, coverage, reason_rows, bin_rows


def paired_comparison(raw_validation, proc_validation, selected_delta, original, proc_origin):
    raw_key, proc_key = keyed(raw_validation), keyed(proc_validation)
    common = raw_key.index.intersection(proc_key.index).sort_values()
    if common.empty:
        return [], pd.DataFrame()
    a, b = raw_key.loc[common], proc_key.loc[common]
    label, features = original['label'], original['features']
    np.testing.assert_allclose(a[label], b[label], rtol=0, atol=0)
    proc_config = read_json(proc_origin / 'config_before_fit.json')
    if proc_config['original'] != original or proc_config['test_policy'] != 'DO NOT LOAD OR PREDICT TEST; old test is diagnostic only':
        raise ValueError('Proc delta comparison config differs from original')
    proc_name = read_json(proc_origin / 'summary.json')['best_delta']
    proc_model = XGBRegressor()
    proc_model.load_model(proc_origin / f'{proc_name}.ubj')
    predictions = {'raw_' + name: value for name, value in baseline_predictions(a).items()}
    predictions['raw_delta_selected'] = depth_from_delta(a.depth_lag_0_cm, selected_delta.predict(a[features]))
    predictions['proc_persistence'] = b.depth_lag_0_cm.to_numpy()
    predictions['proc_delta_selected'] = depth_from_delta(b.depth_lag_0_cm, proc_model.predict(b[features]))
    rows = [{'split': 'validation_common_only', 'model': name,
        **evaluate(a[label], prediction, a.depth_lag_0_cm if name.startswith('raw_') else b.depth_lag_0_cm)}
        for name, prediction in predictions.items()]
    stored = a[[label]].copy()
    stored['current_raw_cm'] = a.depth_lag_0_cm
    stored['current_proc_cm'] = b.depth_lag_0_cm
    for name, prediction in predictions.items():
        stored[f'prediction_{name}_cm'] = prediction
    return rows, stored


def run(origin=DEFAULT_ORIGIN, proc_origin=PROC_DELTA_ORIGIN):
    original, old_manifest = original_config(origin)
    source_manifest = read_json(DATA / 'selection_manifest.json')
    output = BASE / 'training_runs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_raw_delta')
    output.mkdir(parents=True, exist_ok=False)
    config = {'status': 'EXPLORATORY_RAW_INPUT_FUTURE_PROC_LABEL_NOT_DEPLOYMENT_APPROVED',
        'origin': str(origin.resolve()), 'proc_delta_origin': str(proc_origin.resolve()),
        'original': original, 'preprocessing': POLICY,
        'split_boundaries_utc': [str(TRAIN_END), str(VALIDATION_END)],
        'embargo_hours_each_side': EMBARGO.total_seconds() / 3600,
        'station_policy': 'exact original 3 core; no seven additional stations; no reranking',
        'training_rows': 'all eligible raw-history/future-proc-label windows; counts differ from proc branch',
        'selection': 'min balanced MAE on ALL raw-eligible validation rows; tie overall MAE, persistence, ID',
        'paired_policy': 'common validation rows only, reported AFTER selection; training memberships can differ',
        'test_policy': 'NO TEST FILES OR JULY-2023-ONWARD SOURCE CACHES LOADED',
        'features': original['features'], 'label': original['label'],
        'candidate_configs': original['candidate_configs'],
        'fit_target': 'future proc label minus current RAW-DERIVED depth',
        'weight_policy': 'unchanged: future proc label >=10cm receives candidate wet_weight',
        'model_and_validation': 'same 4 XGBoost configs, seed42, CPU<=4 threads; no early stopping/refit',
        'code_sha256': {str(path.relative_to(BASE)): digest(path) for path in (
            Path(__file__), Path(__file__).with_name('pipeline.py'),
            BASE / 'delta_experiment.py', BASE / 'train_baseline_xgboost.py', DATA / 'pipeline.py')},
        'source_manifest_sha256': digest(DATA / 'selection_manifest.json'),
        'origin_config_sha256': digest(origin / 'config_before_fit.json'),
        'proc_delta_config_sha256': digest(proc_origin / 'config_before_fit.json'),
        'proc_delta_model_sha256': digest(proc_origin / (read_json(proc_origin / 'summary.json')['best_delta'] + '.ubj')),
        'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('xgboost', 'scikit-learn', 'pandas', 'numpy', 'pyarrow')},
        'limits': [
            'Causal with respect to archive raw observation order, NOT verified arrival-time/clock/calibration behavior.',
            'Future processed labels are offline proxies, not independently verified flood depths.',
            'Median can attenuate fast changes; one-sample bins cannot reject blips; persistent objects can survive.',
            'Negative raw is floored, not repaired calibration or proof of dry conditions.',
            'Station selection originally used proc train statistics; no claim of raw-only station selection.',
            'Raw/proc training rows differ; common validation does not remove that training-data confound.',
            'Repeatedly inspected validation is exploratory; no claim of real-time 10-minute accuracy or local transfer.',
            'No ESP32, API, frontend, routing or live alerts modified; artifacts local under FloodNet terms.',
        ]}
    for package, version in config['packages'].items():
        if version != original['packages'][package]:
            raise ValueError(f'Runtime differs from original for {package}')
    json_write(output / 'config_before_preparation.json', config)
    proc_provenance = []
    old = {split: verified_split(split, origin, old_manifest, proc_provenance) for split in ('train', 'validation')}
    json_write(output / 'proc_window_provenance.json', proc_provenance)
    raw, coverage, reasons, bin_audit = prepare(output, original, old, source_manifest)
    train, validation = raw['train'], raw['validation']
    features, label = original['features'], original['label']
    json_write(output / 'config_before_fit.json', {**config, 'counts': {s: len(f) for s, f in raw.items()},
        'window_provenance_sha256': digest(output / 'window_provenance.json')})
    predictions = baseline_predictions(validation)
    rows = [{'split': 'validation_raw_eligible', 'model': name,
        **evaluate(validation[label], value, validation.depth_lag_0_cm)} for name, value in predictions.items()]
    models = {}
    for candidate in original['candidate_configs']:
        name = 'raw_delta_' + candidate['id']
        model = make_model(original, candidate)
        weights = np.where(train[label].to_numpy() >= 10, candidate['wet_weight'], 1.)
        start = time.perf_counter()
        model.fit(train[features], delta_target(train[label], train.depth_lag_0_cm), sample_weight=weights, verbose=False)
        predictions[name] = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[features]))
        rows.append({'split': 'validation_raw_eligible', 'model': name,
            **evaluate(validation[label], predictions[name], validation.depth_lag_0_cm),
            'fit_seconds': time.perf_counter() - start})
        model.save_model(output / f'{name}.ubj')
        reloaded = XGBRegressor()
        reloaded.load_model(output / f'{name}.ubj')
        np.testing.assert_allclose(predictions[name], depth_from_delta(validation.depth_lag_0_cm,
            reloaded.predict(validation[features])), rtol=1e-6, atol=1e-6)
        models[name] = model
        print(f'{name}: validation balanced_MAE={rows[-1][METRIC]:.6f} cm', flush=True)
    best_delta = choose([r for r in rows if r['model'].startswith('raw_delta_')])
    selected = select_forecaster(rows)
    # Lock decision before reading old proc model outputs on the common subset.
    json_write(output / 'selection_validation_only.json', {'best_delta': best_delta,
        'selected_forecaster': selected, 'validation_candidates': rows, 'test_loaded': False,
        'selection_rule': config['selection'], 'paired_metrics_used_for_selection': False})
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False, encoding='utf-8-sig')
    stored = validation[['deployment_id', label, 'depth_lag_0_cm',
        'audit_negative_raw_history', 'audit_sparse_raw_history']].copy()
    for name, prediction in predictions.items():
        stored[f'prediction_{name}_cm'] = prediction
    stored.to_parquet(output / 'validation_predictions_local.parquet')
    paired_rows, paired_predictions = paired_comparison(validation, old['validation'], models[best_delta], original, proc_origin)
    pd.DataFrame(paired_rows).to_csv(output / 'paired_validation_metrics.csv', index=False, encoding='utf-8-sig')
    paired_predictions.to_parquet(output / 'paired_validation_predictions_local.parquet')
    breakdown = []
    groups = {'all': np.ones(len(validation), dtype=bool)}
    groups.update({f'station:{s}': validation.deployment_id.eq(s).to_numpy() for s in validation.deployment_id.unique()})
    groups['negative_raw_history'] = validation.audit_negative_raw_history.to_numpy()
    groups['no_negative_raw_history'] = ~validation.audit_negative_raw_history.to_numpy()
    groups['sparse_raw_history'] = validation.audit_sparse_raw_history.to_numpy()
    for group, mask in groups.items():
        if not mask.any():
            continue
        for name in ('persistence', 'linear_trend_5min', best_delta):
            breakdown.append({'group': group, 'model': name,
                **evaluate(validation[label].to_numpy()[mask], np.asarray(predictions[name])[mask],
                           validation.depth_lag_0_cm.to_numpy()[mask])})
    pd.DataFrame(breakdown).to_csv(output / 'validation_breakdown.csv', index=False, encoding='utf-8-sig')
    summary = {'status': config['status'], 'output': str(output), 'counts': {s: len(f) for s, f in raw.items()},
        'coverage': coverage, 'exclusion_reasons': reasons, 'bin_audit': bin_audit,
        'best_delta': best_delta, 'selected_forecaster': selected, 'validation_metrics': rows,
        'paired_validation_metrics': paired_rows, 'paired_metrics_used_for_selection': False,
        'test_loaded': False, 'model_roundtrip': 'PASS on all validation rows, all four models', 'limits': config['limits']}
    json_write(output / 'summary.json', summary)
    from raw_branch.report import write_report
    write_report(output, summary)
    print(f'RAW EXPERIMENT COMPLETED: {output}', flush=True)
    return output
