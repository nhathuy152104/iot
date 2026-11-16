"""Add train-only candidate stations, keep the original raw validation unchanged."""
import datetime as dt
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from delta_experiment import METRIC, delta_target, depth_from_delta, make_model, read_json
from five_minute.pipeline import TRAIN_END, VALIDATION_END, EMBARGO, month_key, split_masks
from raw_branch.experiment import load_raw_sources
from raw_branch.pipeline import POLICY
from raw_expansion.selection import CANDIDATES, RULES, build_candidate_train, select_overall
from train_baseline_xgboost import BASE, DATA, baseline_predictions, choose, digest, evaluate, json_write

RAW_ORIGIN = BASE / 'training_runs' / '20261110T043429_532355Z_raw_delta'
REVIEW = DATA / 'review_18' / 'station_review.csv'


def candidate_train_manifest(manifest):
    result = dict(manifest)
    result['source_files'] = [name for name in manifest['source_files']
        if pd.Timestamp(year=month_key(Path(name))[0], month=month_key(Path(name))[1], day=1, tz='UTC') < TRAIN_END]
    return result


def frozen_inputs(base):
    config = read_json(base / 'config_before_fit.json')
    for name, checksum in config['code_sha256'].items():
        if digest(BASE / name) != checksum:
            raise ValueError(f'Original raw processing code changed: {name}')
    rows = read_json(base / 'window_provenance.json')
    if len(rows) != 6 or {r['split'] for r in rows} != {'train', 'validation'}:
        raise ValueError('Unexpected frozen raw window manifest')
    paths = [base / name for name in ('config_before_fit.json', 'summary.json', 'metrics.csv',
        'selection_validation_only.json', 'validation_predictions_local.parquet')]
    paths += [base / f'raw_delta_{c["id"]}.ubj' for c in config['candidate_configs']]
    for row in rows:
        path = Path(row['path'])
        if digest(path) != row['sha256']:
            raise ValueError(f'Frozen source changed: {path}')
        paths.append(path)
    return config, [{'path': str(path.resolve()), 'sha256': digest(path)} for path in paths]


def load_frozen_split(base, split):
    if split not in ('train', 'validation'):
        raise ValueError('Test is forbidden')
    rows = [r for r in read_json(base / 'window_provenance.json') if r['split'] == split]
    parts = []
    for row in rows:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f'Frozen window changed: {row["path"]}')
        frame = pd.read_parquet(row['path'])
        assert len(frame) == row['rows'] and split_masks(frame.index)[split].all()
        assert frame.deployment_id.eq(row['station']).all()
        parts.append(frame)
    return pd.concat(parts).sort_index(kind='stable')


def prepare(base=RAW_ORIGIN):
    base = base.resolve()
    prior, frozen = frozen_inputs(base)
    review = pd.read_csv(REVIEW)
    proposals = review.loc[review.proposal.eq('candidate_event_segments'), 'deployment_id'].tolist()
    if set(proposals) != set(CANDIDATES):
        raise ValueError('Seven-station candidate list changed; explicitly revise the protocol first')
    output = BASE / 'training_runs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_raw_expanded')
    output.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), Path(__file__).with_name('selection.py'),
        BASE / 'raw_branch' / 'pipeline.py', BASE / 'raw_branch' / 'experiment.py',
        BASE / 'delta_experiment.py', BASE / 'train_baseline_xgboost.py', DATA / 'pipeline.py']
    config = {'status': 'EXPLORATORY_RAW_TRAIN_EXPANSION_NOT_DEPLOYMENT_APPROVED',
        'base_raw_run': str(base), 'original_model_config': prior['original'],
        'core_stations': prior['original']['stations'], 'candidate_stations': list(CANDIDATES),
        'coverage_rules_train_only': RULES, 'preprocessing': POLICY,
        'features': prior['features'], 'label': prior['label'],
        'candidate_model_configs': prior['candidate_configs'],
        'split_boundaries_utc': [str(TRAIN_END), str(VALIDATION_END)],
        'embargo_hours_each_side': EMBARGO.total_seconds() / 3600,
        'candidate_source_policy': 'read only pre-January-2023 monthly candidate caches; trim observations before train embargo boundary',
        'training_policy': 'core train unchanged + ALL raw-eligible train windows of each coverage-qualified candidate; no offline masks, sampling, upper cuts or station weights',
        'quality_policy': 'coverage qualification is NOT clean-label approval; report flags, retain difficult cases, do not rank/select by validation or model scores',
        'validation_policy': 'exact frozen 92215 core raw windows; no new candidate validation, no label/input mutation',
        'selection_rule': 'min balanced validation MAE; tie overall MAE, persistence, trend, frozen-three arm, model ID',
        'test_policy': 'NO TEST; no new-station validation or July-2023-onward caches',
        'fit_policy': 'same four configs, future-proc minus current-raw target, same future>=10cm weights, CPU<=4 threads, seed42, no refit/early stopping',
        'code_sha256': {str(path.relative_to(BASE)): digest(path) for path in code_paths},
        'source_manifest_sha256': digest(DATA / 'selection_manifest.json'),
        'review_table_sha256': digest(REVIEW),
        'packages': {p: importlib.metadata.version(p) for p in prior['packages']},
        'limits': prior['limits'] + [
            'All seven candidates have prior quality concerns; full eligible training windows are an explicit stress test, not clean-label certification.',
            'Groups separated by six hours are coverage proxies, not independent verified storms.',
            'Frozen validation covers two original stations; does not establish generalization to new stations.',
            'Selection of a better model on repeatedly used validation is exploratory, not independent confirmation.',
        ]}
    if config['packages'] != prior['packages']:
        raise ValueError('Runtime changed since the frozen raw experiment')
    if prior['preprocessing'] != POLICY:
        raise ValueError('Raw preprocessing policy changed')
    json_write(output / 'config_before_preparation.json', config)
    json_write(output / 'frozen_input_provenance.json', frozen)
    manifest = candidate_train_manifest(read_json(DATA / 'selection_manifest.json'))
    sources, csv_files, cache_files = load_raw_sources(manifest, list(CANDIDATES))
    folder = output / 'candidate_train'
    folder.mkdir()
    audits, exclusions, windows_provenance = [], [], []
    for station in CANDIDATES:
        source = sources[station].loc[sources[station].index < TRAIN_END - EMBARGO]
        windows, audit, reasons = build_candidate_train(station, source, config['features'], config['label'])
        path = folder / f'{station}_train.parquet'
        windows.to_parquet(path)
        audits.append(audit)
        exclusions.extend(reasons)
        windows_provenance.append({'station': station, 'split': 'train', 'path': str(path),
            'rows': len(windows), 'sha256': digest(path),
            'selected': audit['selected_for_exploratory_expansion']})
        print(f'{station}: train={len(windows):,}, wet={audit["wet_ge_10cm"]:,}, '
              f'selected={audit["selected_for_exploratory_expansion"]}, failed={audit["failed_criteria"]}', flush=True)
    selected = [r['station'] for r in audits if r['selected_for_exploratory_expansion']]
    decision = {'selected_additions': selected, 'core_stations': config['core_stations'],
        'coverage_rules': RULES, 'station_audits_train_only': audits,
        'validation_used': False, 'offline_masks_used': False,
        'quality_status': 'Coverage-qualified exploratory data, NOT verified clean flood labels'}
    json_write(output / 'station_selection_before_fit.json', decision)
    json_write(output / 'source_provenance.json', {'csv_fingerprints': csv_files, 'cache_files': cache_files})
    json_write(output / 'candidate_window_provenance.json', windows_provenance)
    table = [{**r, 'failed_criteria': '|'.join(r['failed_criteria'])} for r in audits]
    pd.DataFrame(table).to_csv(output / 'station_audit_train.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(exclusions).to_csv(output / 'candidate_exclusion_reasons.csv', index=False, encoding='utf-8-sig')
    print(f'EXPANSION PREPARED (no fit, no validation scoring/selection): {output}', flush=True)
    return output


def load_prepared(output):
    config = read_json(output / 'config_before_preparation.json')
    for name, checksum in config['code_sha256'].items():
        if digest(BASE / name) != checksum:
            raise ValueError(f'Code changed after preparation: {name}')
    if digest(DATA / 'selection_manifest.json') != config['source_manifest_sha256'] or digest(REVIEW) != config['review_table_sha256']:
        raise ValueError('Source manifest/review changed')
    for row in read_json(output / 'frozen_input_provenance.json'):
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f'Frozen raw arm changed: {row["path"]}')
    decision = read_json(output / 'station_selection_before_fit.json')
    assert decision['validation_used'] is False and decision['offline_masks_used'] is False
    provenance = read_json(output / 'candidate_window_provenance.json')
    parts = []
    for row in provenance:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f'Prepared window changed: {row["path"]}')
        assert row['selected'] == (row['station'] in decision['selected_additions'])
        if row['selected']:
            frame = pd.read_parquet(row['path'])
            assert split_masks(frame.index)['train'].all()
            assert frame.deployment_id.eq(row['station']).all()
            parts.append(frame)
    base = Path(config['base_raw_run'])
    core = load_frozen_split(base, 'train')
    train = pd.concat([core, *parts]).sort_index(kind='stable')
    return config, decision, core, train


def fit(output):
    import time
    if (output / 'config_before_fit.json').exists():
        raise ValueError('This prepared run has already entered fitting; create a new run instead of overwriting')
    config, station_decision, core, train = load_prepared(output)
    if not station_decision['selected_additions']:
        raise ValueError('No candidates pass declared train-only coverage; do not auto-lower thresholds')
    base = Path(config['base_raw_run'])
    validation = load_frozen_split(base, 'validation')
    features, label = config['features'], config['label']
    counts = {'core_train': len(core), 'added_train': len(train) - len(core),
        'expanded_train': len(train), 'validation': len(validation),
        'core_train_wet': int(core[label].ge(10).sum()),
        'expanded_train_wet': int(train[label].ge(10).sum())}
    json_write(output / 'config_before_fit.json', {**config, 'counts': counts,
        'selected_additions': station_decision['selected_additions'],
        'station_selection_sha256': digest(output / 'station_selection_before_fit.json'),
        'candidate_windows_manifest_sha256': digest(output / 'candidate_window_provenance.json')})
    old_metrics = pd.read_csv(base / 'metrics.csv')
    predictions = baseline_predictions(validation)
    rows = [{'model': name, 'arm': 'baseline', **evaluate(validation[label], p, validation.depth_lag_0_cm)}
            for name, p in predictions.items()]
    models = {}
    # Frozen three-station models: no retraining or changed target rows.
    for candidate in config['candidate_model_configs']:
        name = 'three_' + candidate['id']
        model = XGBRegressor()
        model.load_model(base / f'raw_delta_{candidate["id"]}.ubj')
        predictions[name] = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[features]))
        row = {'model': name, 'arm': 'three', **evaluate(validation[label], predictions[name], validation.depth_lag_0_cm)}
        expected = old_metrics.loc[old_metrics.model.eq('raw_delta_' + candidate['id'])].iloc[0]
        for key, value in evaluate(validation[label], predictions[name], validation.depth_lag_0_cm).items():
            if value is None:
                assert pd.isna(expected[key])
            else:
                np.testing.assert_allclose(value, expected[key], rtol=1e-8, atol=1e-8)
        rows.append(row)
        models[name] = model
    for candidate in config['candidate_model_configs']:
        name = 'expanded_' + candidate['id']
        model = make_model(config['original_model_config'], candidate)
        weights = np.where(train[label].to_numpy() >= 10, candidate['wet_weight'], 1.)
        started = time.perf_counter()
        model.fit(train[features], delta_target(train[label], train.depth_lag_0_cm), sample_weight=weights, verbose=False)
        predictions[name] = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[features]))
        row = {'model': name, 'arm': 'expanded',
            **evaluate(validation[label], predictions[name], validation.depth_lag_0_cm),
            'fit_seconds': time.perf_counter() - started}
        rows.append(row)
        model.save_model(output / f'{name}.ubj')
        reloaded = XGBRegressor()
        reloaded.load_model(output / f'{name}.ubj')
        np.testing.assert_allclose(predictions[name], depth_from_delta(validation.depth_lag_0_cm,
            reloaded.predict(validation[features])), rtol=1e-6, atol=1e-6)
        models[name] = model
        print(f'{name}: validation balanced_MAE={row[METRIC]:.6f}cm FP={row["fp"]} FN={row["fn"]}', flush=True)
    best_three = choose([r for r in rows if r['arm'] == 'three'])
    best_expanded = choose([r for r in rows if r['arm'] == 'expanded'])
    selected = select_overall(rows)
    decision = {'best_three': best_three, 'best_expanded': best_expanded,
        'selected_forecaster': selected, 'validation_metrics': rows,
        'station_selection_used_validation': False, 'test_loaded': False,
        'selection_rule': config['selection_rule']}
    json_write(output / 'selection_validation_only.json', decision)
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False, encoding='utf-8-sig')
    stored = validation[['deployment_id', label, 'depth_lag_0_cm']].copy()
    for name, p in predictions.items():
        stored[f'prediction_{name}_cm'] = p
    stored.to_parquet(output / 'validation_predictions_local.parquet')
    breakdown = []
    for station in sorted(validation.deployment_id.unique()):
        mask = validation.deployment_id.eq(station).to_numpy()
        for name, p in predictions.items():
            breakdown.append({'station': station, 'model': name,
                **evaluate(validation[label].to_numpy()[mask], np.asarray(p)[mask], validation.depth_lag_0_cm.to_numpy()[mask])})
    pd.DataFrame(breakdown).to_csv(output / 'validation_breakdown.csv', index=False, encoding='utf-8-sig')
    summaries = {r['model']: r for r in rows}
    paired = []
    for candidate in config['candidate_model_configs']:
        a, b = summaries['three_' + candidate['id']], summaries['expanded_' + candidate['id']]
        paired.append({'config': candidate['id'], 'three_balanced_mae_cm': a[METRIC],
            'expanded_balanced_mae_cm': b[METRIC], 'change_cm': b[METRIC] - a[METRIC],
            'three_fp': a['fp'], 'expanded_fp': b['fp'], 'three_fn': a['fn'], 'expanded_fn': b['fn']})
    pd.DataFrame(paired).to_csv(output / 'matched_config_comparison.csv', index=False, encoding='utf-8-sig')
    summary = {'status': config['status'], 'output': str(output), 'counts': counts,
        'core_stations': config['core_stations'], 'selected_additions': station_decision['selected_additions'],
        'station_audits_train_only': station_decision['station_audits_train_only'],
        'best_three': best_three, 'best_expanded': best_expanded, 'selected_forecaster': selected,
        'validation_metrics': rows, 'matched_config_comparison': paired,
        'expanded_vs_three_balanced_mae_change_cm': summaries[best_expanded][METRIC] - summaries[best_three][METRIC],
        'test_loaded': False, 'new_station_validation_loaded': False,
        'roundtrip': 'PASS for all expanded models on all frozen validation rows', 'limits': config['limits']}
    json_write(output / 'summary.json', summary)
    from raw_expansion.report import write_report
    write_report(output, summary)
    print(f'EXPANSION COMPLETED: {output}', flush=True)
    return output
