"""Controlled delta-target experiment. Train/validation only; no test evaluation.

Reuse the original rows, feature whitelist, four configurations and metric.
The sole modeling change is y_fit = future_depth - current_depth.
Artifacts are local research derivatives, not deployment-approved models.
"""
import argparse
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from train_baseline_xgboost import (
    BASE, DATA, baseline_predictions, choose, digest, evaluate, json_write, load_split,
)

DEFAULT_ORIGIN = BASE / 'training_runs' / '20261110T033843_342867Z'
METRIC = 'balanced_wet_nonwet_mae_cm'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def original_config(origin):
    config = read_json(origin / 'config_before_fit.json')
    if digest(DATA / 'selection_manifest.json') != config['source_manifest_sha256']:
        raise ValueError('Manifest changed since original run; refusing an uncontrolled comparison')
    manifest = {'selected_core': config['stations'], 'feature_columns': config['features'],
                'label_column': config['label']}
    return config, manifest


def verified_split(split, origin, manifest, provenance, allowed=('train', 'validation')):
    if split not in allowed:
        raise ValueError(f'Split {split!r} is forbidden in this experiment')
    source = [r for r in read_json(origin / 'input_provenance.json') if r['split'] == split]
    if not source:
        raise ValueError(f'No original provenance for {split}')
    for row in source:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError(f'Original source changed: {row["path"]}')
    actual = []
    frame = load_split(split, manifest, actual)
    if actual != source:
        raise ValueError(f'Source rows/paths/order differ from original {split}')
    provenance.extend(actual)
    return frame


def delta_target(future, current):
    future, current = np.asarray(future, dtype=float), np.asarray(current, dtype=float)
    if future.shape != current.shape or not (np.isfinite(future).all() and np.isfinite(current).all()):
        raise ValueError('Finite, equally shaped future/current arrays required')
    return future - current


def depth_from_delta(current, delta):
    current, delta = np.asarray(current, dtype=float), np.asarray(delta, dtype=float)
    if current.shape != delta.shape or not (np.isfinite(current).all() and np.isfinite(delta).all()):
        raise ValueError('Finite, equally shaped current/delta arrays required')
    # Negative CHANGES are valid. Clip only the reconstructed physical depth.
    return np.maximum(current + delta, 0)


def select_forecaster(rows):
    """Keep persistence on an exact tie; do not force selection of an ML model."""
    usable = [r for r in rows if r[METRIC] is not None and np.isfinite(r[METRIC])]
    if not usable:
        raise ValueError('No finite balanced validation metric')
    return min(usable, key=lambda r: (r[METRIC], r['mae_cm'],
               r['model'] != 'persistence', r['model']))['model']


def make_model(config, candidate):
    return XGBRegressor(**config['common_xgboost_params'],
        n_estimators=candidate['n_estimators'], max_depth=candidate['max_depth'],
        random_state=config['random_seed'], device=config['device'], eval_metric='mae')


def run(origin):
    origin = origin.resolve()
    original, manifest = original_config(origin)
    output = BASE / 'training_runs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_delta')
    output.mkdir(parents=True, exist_ok=False)
    config = {
        'status': 'EXPLORATORY_VALIDATION_ONLY_NOT_DEPLOYMENT_APPROVED',
        'origin': str(origin), 'original': original,
        'only_modeling_change': 'fit future_depth - current_depth; predict max(0, current_depth + fitted_delta)',
        'sample_weights': 'unchanged: ORIGINAL FUTURE DEPTH >=10cm receives candidate wet_weight',
        'rows': 'all original core train/validation windows; no offline quality masks or resampling',
        'test_policy': 'DO NOT LOAD OR PREDICT TEST; old test is diagnostic only',
        'selection': 'min balanced validation MAE, tie overall MAE then prefer persistence then ID',
        'modeling_constraints': 'no new features, stations, hyperparameters, refit, or early stopping',
        'origin_config_sha256': digest(origin / 'config_before_fit.json'),
        'origin_model_sha256': digest(origin / 'xgboost_selected.ubj'),
        'script_sha256': digest(Path(__file__)),
        'reused_training_script_sha256': digest(BASE / 'train_baseline_xgboost.py'),
        'reused_pipeline_sha256': digest(DATA / 'pipeline.py'),
        'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('xgboost', 'scikit-learn', 'pandas', 'numpy', 'pyarrow')},
    }
    for package, version in config['packages'].items():
        if version != original['packages'][package]:
            raise ValueError(f'Runtime changed for {package}; not a one-factor comparison')
    json_write(output / 'config_before_fit.json', config)
    provenance = []
    train = verified_split('train', origin, manifest, provenance)
    validation = verified_split('validation', origin, manifest, provenance)
    features, label = original['features'], original['label']
    current = validation.depth_lag_0_cm.to_numpy()
    predictions = baseline_predictions(validation)
    old = XGBRegressor()
    old.load_model(origin / 'xgboost_selected.ubj')
    predictions['original_absolute_selected'] = np.maximum(old.predict(validation[features]), 0)
    rows = [{'split': 'validation', 'model': name, **evaluate(validation[label], p, current)}
            for name, p in predictions.items()]
    original_metrics = pd.read_csv(origin / 'metrics.csv')
    selected_old = read_json(origin / 'selection_before_test.json')['selected_xgboost']
    for name in predictions:
        old_name = selected_old if name == 'original_absolute_selected' else name
        expected = original_metrics.loc[(original_metrics.split == 'validation') & (original_metrics.model == old_name)].iloc[0]
        actual = next(r for r in rows if r['model'] == name)
        for key in (METRIC, 'mae_cm', 'mae_wet_cm', 'mae_nonwet_cm', 'fp', 'fn'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-8, atol=1e-8)
    target = delta_target(train[label], train.depth_lag_0_cm)
    models = {}
    for candidate in original['candidate_configs']:
        name = 'delta_' + candidate['id']
        model = make_model(original, candidate)
        # Keep weighting based on FUTURE DEPTH, not on the magnitude/sign of delta.
        weights = np.where(train[label].to_numpy() >= 10, candidate['wet_weight'], 1.)
        started = time.perf_counter()
        model.fit(train[features], target, sample_weight=weights, verbose=False)
        predictions[name] = depth_from_delta(current, model.predict(validation[features]))
        rows.append({'split': 'validation', 'model': name,
            **evaluate(validation[label], predictions[name], current),
            'fit_seconds': time.perf_counter() - started})
        model.save_model(output / f'{name}.ubj')
        reloaded = XGBRegressor()
        reloaded.load_model(output / f'{name}.ubj')
        np.testing.assert_allclose(predictions[name], depth_from_delta(current, reloaded.predict(validation[features])), rtol=1e-6, atol=1e-6)
        models[name] = model
        print(f'{name}: validation balanced_MAE={rows[-1][METRIC]:.6f}cm', flush=True)
    best_delta = choose([r for r in rows if r['model'].startswith('delta_')])
    selected = select_forecaster(rows)
    decision = {'best_delta': best_delta, 'selected_forecaster': selected,
        'validation_candidates': rows, 'test_loaded': False, 'selection_rule': config['selection']}
    json_write(output / 'selection_validation_only.json', decision)
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False, encoding='utf-8-sig')
    stored = validation[['deployment_id', label, 'depth_lag_0_cm']].copy()
    for name, prediction in predictions.items():
        stored[f'prediction_{name}_cm'] = prediction
    stored.to_parquet(output / 'validation_predictions_local.parquet')
    json_write(output / 'input_provenance.json', provenance)
    best = next(r for r in rows if r['model'] == best_delta)
    baseline = next(r for r in rows if r['model'] == 'persistence')
    summary = {'status': config['status'], 'output': str(output),
        'counts': {'train': len(train), 'validation': len(validation)},
        'best_delta': best_delta, 'selected_forecaster': selected,
        'delta_beats_persistence_balanced_validation': best[METRIC] < baseline[METRIC],
        'balanced_mae_change_vs_persistence_cm': best[METRIC] - baseline[METRIC],
        'validation_metrics': rows, 'test_loaded': False,
        'roundtrip': 'PASS on all validation rows, all four delta models',
        'limits': original['scientific_limits']}
    json_write(output / 'summary.json', summary)
    write_report(output, summary)
    print(json.dumps({k: summary[k] for k in ('output', 'best_delta', 'selected_forecaster',
        'delta_beats_persistence_balanced_validation', 'balanced_mae_change_vs_persistence_cm', 'test_loaded')}, indent=2), flush=True)
    return output


def write_report(output, summary):
    lines = ['# Thí nghiệm XGBoost dự báo mức thay đổi', '',
        '**Chỉ đánh giá validation. Không nạp/dự đoán test; chưa duyệt triển khai.**', '',
        'Giữ nguyên dữ liệu, 3 trạm core, 12 feature, 4 cấu hình, seed42, trọng số và hàm mục tiêu squared error. '
        'Chỉ đổi đích học từ độ sâu thành `độ sâu tương lai − độ sâu hiện tại`; cộng lại độ sâu hiện tại trước khi chấm điểm.', '',
        f'Số cửa sổ: {summary["counts"]}. Không lọc bằng mask offline; không thêm 7 trạm.', '',
        '| Model | MAE toàn tập | MAE ≥10cm | MAE <10cm | MAE cân bằng | Recall | Precision | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['validation_metrics']:
        lines.append(f'| {r["model"]} | {r["mae_cm"]:.4f} | {r["mae_wet_cm"]:.4f} | {r["mae_nonwet_cm"]:.4f} | '
            f'{r[METRIC]:.4f} | {r["recall_wet"]:.4f} | {r["precision_wet"]:.4f} | {r["fp"]} | {r["fn"]} |')
    lines += ['', f'Ứng viên delta tốt nhất: **{summary["best_delta"]}**. '
        f'Phương pháp tốt nhất trong toàn bộ so sánh: **{summary["selected_forecaster"]}**.',
        f'Chênh lệch MAE cân bằng của delta tốt nhất so với persistence: **{summary["balanced_mae_change_vs_persistence_cm"]:+.6f} cm** (âm tốt hơn).', '',
        'Đây là lựa chọn thăm dò trên validation đã dùng nhiều lần, không phải chứng minh tổng quát hóa. '
        'Không coi giảm một chỉ số là đã đạt yêu cầu cảnh báo; phải xem báo giả, bỏ sót và đánh giá theo sự kiện độc lập.', '',
        '## Giới hạn', '',
        '- Nhãn là số đo cuối bin (t+5,t+10], không phải chính xác t+10 hoặc cực đại 10 phút.',
        '- Proc chưa được xác minh causal; nhãn chưa đối chiếu hiện trường. Không dùng kết quả để quyết định đường an toàn.',
        '- Test cũ chỉ dùng trong báo cáo chẩn đoán riêng, không dùng chọn/tinh chỉnh thí nghiệm này. Cần bộ xác nhận độc lập sau này.',
        '- Chưa chạy ESP32, API hoặc frontend; không thay cảnh báo theo ngưỡng hiện tại.', '',
        '## Artifacts', '',
        '- `config_before_fit.json`: cấu hình và checksum trước fit.',
        '- `selection_validation_only.json`, `metrics.csv`, `summary.json`: quyết định và chỉ số.',
        '- `delta_*.ubj`: cả bốn model nghiên cứu, không tự động duyệt deployment.',
        '- `validation_predictions_local.parquet`, `input_provenance.json`: giữ cục bộ.', '',
        'Nguồn: FloodNet (New York University and The City University of New York). '
        'Không phân phối dữ liệu, dự đoán hoặc trọng số khi chưa làm rõ quyền. '
        'Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0 cùng Data Access License Agreement.', '']
    (output / 'BAO_CAO_DELTA.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, default=DEFAULT_ORIGIN)
    run(parser.parse_args().origin)
