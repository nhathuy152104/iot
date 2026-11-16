"""Depth-first diagnostics and ONE fixed MAE-objective trial; no test or deployment."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from archive_paths import resolve_archive_path
from delta_experiment import delta_target, depth_from_delta, make_model, read_json
from diagnose_raw_expansion import EXPANDED, snapshot, verified_windows
from train_baseline_xgboost import BASE, baseline_predictions, digest, evaluate, json_write


def error_profile(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Matching finite arrays required')
    if not len(y):
        return {'windows': 0, 'mae_cm': None, 'p90_abs_error_cm': None, 'p99_abs_error_cm': None,
                'max_abs_error_cm': None, 'bias_cm': None}
    error = prediction - y
    return {'windows': len(y), 'mae_cm': float(np.abs(error).mean()),
        'p90_abs_error_cm': float(np.quantile(np.abs(error), .90)),
        'p99_abs_error_cm': float(np.quantile(np.abs(error), .99)),
        'max_abs_error_cm': float(np.abs(error).max()), 'bias_cm': float(error.mean())}


def run():
    output = BASE / 'training_runs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_depth_mae_trial')
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / 'protocol_before_fit.json', {
        'scope': 'one fixed hypothesis: absolute-error objective may better match MAE and reduce outlier influence',
        'change': 'reg:squarederror -> reg:absoluteerror only, fixed d3_weight20; no grid search',
        'unchanged': 'core raw train, validation, labels, features, preprocessing, weights20, seed42, 250 trees, depth3',
        'selection': 'EXPLORATORY ONLY; do not replace frozen model or replay export regardless of validation result',
        'test': 'do not load test or new-station validation; no refit using validation',
        'strata': 'all, label>=10cm, 8<=label<12cm, 8<=prediction<12cm; never eligibility or online filters',
        'uncertainty': 'empirical validation error summaries, NOT calibrated prediction intervals or guarantees'})
    config = read_json(EXPANDED / 'config_before_fit.json')
    core_run = resolve_archive_path(config['base_raw_run'])
    frozen = snapshot((core_run, EXPANDED))
    provenance = {}
    train = verified_windows(core_run / 'window_provenance.json', 'train', provenance)
    validation = verified_windows(core_run / 'window_provenance.json', 'validation', provenance)
    assert len(train) == 270312 and len(validation) == 92215
    label, features = config['label'], config['features']
    candidate = next(c for c in config['candidate_model_configs'] if c['id'] == 'd3_weight20')
    old = XGBRegressor()
    model_path = core_run / 'raw_delta_d3_weight20.ubj'
    old.load_model(model_path)
    predictions = baseline_predictions(validation)
    predictions['three_d3_weight20'] = depth_from_delta(validation.depth_lag_0_cm, old.predict(validation[features]))
    prior = pd.read_csv(core_run / 'metrics.csv').set_index('model')
    old_result = evaluate(validation[label], predictions['three_d3_weight20'], validation.depth_lag_0_cm)
    np.testing.assert_allclose(old_result['balanced_wet_nonwet_mae_cm'], prior.loc['raw_delta_d3_weight20', 'balanced_wet_nonwet_mae_cm'])
    model = make_model(config['original_model_config'], candidate)
    model.set_params(objective='reg:absoluteerror')
    json_write(output / 'config_before_fit.json', {'features': features, 'label': label, 'candidate': candidate,
        'parameters': {k: v for k, v in model.get_params().items() if k != 'missing'},
        'missing_parameter': 'NaN (XGBoost default); eligible feature windows are finite',
        'train_rows': len(train), 'validation_rows': len(validation),
        'frozen_model_sha256': digest(model_path), 'code_sha256': digest(Path(__file__))})
    model.fit(train[features], delta_target(train[label], train.depth_lag_0_cm),
              sample_weight=np.where(train[label].to_numpy() >= 10, 20., 1.), verbose=False)
    model.save_model(output / 'trial_absolute_error.ubj')
    prediction = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[features]))
    predictions['trial_absolute_error'] = prediction
    reloaded = XGBRegressor()
    reloaded.load_model(output / 'trial_absolute_error.ubj')
    np.testing.assert_allclose(prediction, depth_from_delta(validation.depth_lag_0_cm,
                               reloaded.predict(validation[features])), atol=1e-6, rtol=1e-6)
    metrics, profiles = [], []
    y = validation[label].to_numpy()
    for name, p in predictions.items():
        metrics.append({'model': name, **evaluate(y, p, validation.depth_lag_0_cm)})
        for group, mask in {'all': np.ones(len(y), bool), 'target_ge10': y >= 10,
                            'target_8_to12': (y >= 8) & (y < 12),
                            'prediction_8_to12': (p >= 8) & (p < 12)}.items():
            profiles.append({'model': name, 'group': group, **error_profile(y[mask], p[mask])})
    pd.DataFrame(metrics).to_csv(output / 'metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(profiles).to_csv(output / 'depth_error_profiles.csv', index=False, encoding='utf-8-sig')
    stored = validation[['deployment_id', label, 'depth_lag_0_cm']].copy()
    for name, p in predictions.items():
        stored['prediction_' + name + '_cm'] = p
    stored.to_parquet(output / 'validation_predictions_local.parquet')
    top = np.argsort(np.abs(predictions['three_d3_weight20'] - y))[-20:][::-1]
    stored.iloc[top].to_csv(output / 'largest_frozen_model_errors_local.csv', encoding='utf-8-sig')
    assert snapshot((core_run, EXPANDED)) == frozen
    json_write(output / 'input_provenance.json', {**provenance,
        str(model_path): {'sha256': digest(model_path), 'purpose': 'frozen comparison'}})
    json_write(output / 'summary.json', {'status': 'PASS', 'test_loaded': False,
        'promoted': False, 'metrics': metrics, 'profiles': profiles, 'frozen_files_unchanged': len(frozen)})
    lines = ['# Thí nghiệm độ sâu: đổi hàm học sang MAE', '',
        '**Một thí nghiệm cố định, hồi cứu trên validation đã dùng nhiều lần; không phê duyệt model mới.**', '',
        '| Model | MAE cân bằng cm | MAE toàn tập cm | MAE nhãn ≥10cm |', '|---|---:|---:|---:|']
    for r in metrics:
        lines.append(f'| {r["model"]} | {r["balanced_wet_nonwet_mae_cm"]:.6f} | {r["mae_cm"]:.6f} | {r["mae_wet_cm"]:.6f} |')
    lines += ['', 'Chỉ đổi objective reg:squarederror → reg:absoluteerror; giữ d3/250 cây/weight20, '
        'train raw 270.312 và validation 92.215, feature/nhãn/mốc chia nguyên bản. Không thêm dữ liệu hoặc lọc ca khó.', '',
        'Xem `depth_error_profiles.csv` để kiểm tra quanh ngưỡng 8–12 cm và P90/P99/sai lớn nhất. '
        'Nhóm theo nhãn và nhóm theo dự báo là hai tập khác nhau; không so các MAE nhóm theo dự báo như trên cùng cửa sổ. '
        'P90 lỗi hồi cứu không phải khoảng tin cậy cá nhân đã hiệu chuẩn.', '',
        'Không thay model ba mã hoặc artifact demo theo kết quả thăm dò. Cần tập xác nhận độc lập cho kết luận cải thiện.', '',
        'Nguồn: FloodNet NYU/CUNY. Artifacts giữ cục bộ; tuân giấy phép nguồn.']
    (output / 'BAO_CAO_THI_NGHIEM_DO_SAU.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    json_write(output / 'artifact_validation.json', {'status': 'PASS', 'model_roundtrip': True,
        'frozen_sha256': frozen, 'output_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()}})
    print(f'DEPTH TRIAL COMPLETED: {output}', flush=True)
    print(pd.DataFrame(metrics)[['model', 'mae_cm', 'balanced_wet_nonwet_mae_cm']].to_string(index=False))
    return output


if __name__ == '__main__':
    run()
