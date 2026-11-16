"""Reproducible, CPU-only, exploratory regression on provisional FloodNet bins.

No random split, no offline masks, no station IDs or label audit metadata in X.
Validation selects a fixed small candidate set; test is read only after choice.
Artifacts stay local and are NOT approved for public or safety-critical use.
"""
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from five_minute.pipeline import split_masks

BASE = Path(__file__).resolve().parent
DATA = BASE / 'five_minute'
SEED = 42
THRESHOLD_CM = 10.0
CANDIDATES = [
    {'id': 'd3_uniform', 'max_depth': 3, 'n_estimators': 250, 'wet_weight': 1.0},
    {'id': 'd3_weight20', 'max_depth': 3, 'n_estimators': 250, 'wet_weight': 20.0},
    {'id': 'd4_uniform', 'max_depth': 4, 'n_estimators': 400, 'wet_weight': 1.0},
    {'id': 'd4_weight20', 'max_depth': 4, 'n_estimators': 400, 'wet_weight': 20.0},
]


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def baseline_predictions(frame):
    persistence = np.maximum(frame['depth_lag_0_cm'].to_numpy(), 0)
    trend = np.maximum(persistence + 10 * frame['slope_5min_cm_per_min'].to_numpy(), 0)
    return {'persistence': persistence, 'linear_trend_5min': trend}


def evaluate(y, predictions, current):
    y, predictions, current = map(np.asarray, (y, predictions, current))
    wet = y >= THRESHOLD_CM
    dry = ~wet
    pred_wet = predictions >= THRESHOLD_CM
    tp = int(np.sum(wet & pred_wet))
    fn = int(np.sum(wet & ~pred_wet))
    fp = int(np.sum(dry & pred_wet))
    tn = int(np.sum(dry & ~pred_wet))
    newly_wet = wet & (current < THRESHOLD_CM)
    # Equal mean of high-target and low-target MAEs avoids all-dry dominance.
    wet_mae = float(mean_absolute_error(y[wet], predictions[wet])) if wet.any() else None
    dry_mae = float(mean_absolute_error(y[dry], predictions[dry])) if dry.any() else None
    balanced_mae = (wet_mae + dry_mae) / 2 if wet_mae is not None and dry_mae is not None else None
    return {'windows': len(y), 'wet_targets_ge_10cm': int(wet.sum()),
        'mae_cm': float(mean_absolute_error(y, predictions)),
        'rmse_cm': float(np.sqrt(mean_squared_error(y, predictions))),
        'mae_wet_cm': wet_mae, 'mae_nonwet_cm': dry_mae,
        'balanced_wet_nonwet_mae_cm': balanced_mae,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'recall_wet': tp / (tp + fn) if tp + fn else None,
        'precision_wet': tp / (tp + fp) if tp + fp else None,
        'false_positive_rate': fp / (fp + tn) if fp + tn else None,
        'new_crossing_windows': int(newly_wet.sum()),
        'new_crossing_detected': int(np.sum(newly_wet & pred_wet)),
        'new_crossing_recall': float(pred_wet[newly_wet].mean()) if newly_wet.any() else None}


def choose(validation_rows):
    eligible = [r for r in validation_rows if r['balanced_wet_nonwet_mae_cm'] is not None]
    if not eligible:
        raise ValueError('Validation lacks both target groups; cannot use declared selection rule')
    return min(eligible, key=lambda r: (r['balanced_wet_nonwet_mae_cm'], r['mae_cm'], r['model']))['model']


def load_split(split, manifest, provenance):
    features = manifest['feature_columns']
    assert not any(c.startswith(('audit_', 'review_')) for c in features)
    assert manifest['label_column'] not in features and 'deployment_id' not in features
    parts = []
    for station in manifest['selected_core']:
        path = DATA / 'provisional_windows' / f'{station}_{split}.parquet'
        frame = pd.read_parquet(path)
        if not len(frame):
            continue
        assert split_masks(frame.index)[split].all(), str(path)
        assert frame['deployment_id'].eq(station).all(), str(path)
        assert np.isfinite(frame[features].to_numpy()).all(), str(path)
        assert frame[manifest['label_column']].ge(0).all(), str(path)
        provenance.append({'split': split, 'station': station, 'path': str(path),
                           'rows': len(frame), 'sha256': digest(path)})
        parts.append(frame)
    if not parts:
        raise ValueError(f'No windows for {split}')
    return pd.concat(parts).sort_index(kind='stable')


def plot_test(test, predictions, label, output):
    paths = []
    # Selected AFTER model choice, for diagnosis only, not tuning.
    for station in sorted(test.deployment_id.unique()):
        positions = np.flatnonzero(test.deployment_id.eq(station).to_numpy())
        frame = test.iloc[positions]
        if not len(frame):
            continue
        center = frame[label].idxmax()
        keep = (frame.index >= center-pd.Timedelta(hours=3)) & (frame.index <= center+pd.Timedelta(hours=3))
        fig, ax = plt.subplots(figsize=(12, 4.5), layout='constrained')
        ax.scatter(frame.index[keep], frame.loc[keep, label], s=15, c='black', label='Archived target proc bin')
        for model, p in predictions.items():
            ax.scatter(frame.index[keep], np.asarray(p)[positions][keep], s=9, alpha=.65, label=model)
        ax.axhline(10, color='red', linestyle='--', linewidth=.7)
        ax.set_title(f'{station} | exploratory TEST case | {center:%Y-%m-%d %H:%M} UTC\nTarget belongs to a later 5-minute bin, not current observed depth')
        ax.set_xlabel('Prediction decision time UTC; scatter does not bridge absent windows')
        ax.set_ylabel('Depth (cm)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        path = output / f'test_{station}.png'
        fig.savefig(path, dpi=125)
        plt.close(fig)
        paths.append(path.name)
    return paths


def main():
    started = time.perf_counter()
    manifest = json.loads((DATA/'selection_manifest.json').read_text(encoding='utf-8'))
    run_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = BASE / 'training_runs' / run_id
    output.mkdir(parents=True, exist_ok=False)
    config = {'status': 'EXPLORATORY ONLY; noncausal archive filtering unverified',
        'stations': manifest['selected_core'], 'features': manifest['feature_columns'],
        'label': manifest['label_column'], 'target_semantics': manifest['target'],
        'candidate_configs': CANDIDATES, 'random_seed': SEED, 'device': 'cpu',
        'common_xgboost_params': {'learning_rate': .05, 'min_child_weight': 10, 'reg_lambda': 10,
            'subsample': 1, 'colsample_bytree': 1, 'objective': 'reg:squarederror',
            'tree_method': 'hist', 'n_jobs': min(4, os.cpu_count() or 1)},
        'selection_rule': 'min validation (MAE target>=10cm + MAE target<10cm)/2; tie overall MAE then ID',
        'baseline_linear_trend': 'max(0, current depth + 10 * past5min slope); does not use future source age',
        'training_row_policy': 'ALL original selected-core eligible windows; no offline review mask and no sampling',
        'wet_weight_policy': 'TRAIN label>=10cm gets fixed weight20 in two candidates only; labels not features',
        'prediction_postprocessing': 'clip negative predictions to zero, no upper clipping',
        'test_policy': 'load once after validation choice; never early stopping or candidate selection on test',
        'test_history_caveat': manifest['test_status'],
        'scientific_limits': ['No verified ground-truth flood labels or field validation.',
            'Processed depths can contain unverified noncausal filtering or sensor noise.',
            'Core station widely_proud_lizard has a visually ambiguous possible-box case.',
            'Overlapping windows are correlated; counts are not counts of independent events.',
            'New crossing recall is window-level and not event-level warning lead time.',
            'Different seasons/station availability across validation/test can shift metrics.',
            'No claim of transferability to ESP32 or local flood dynamics.',
            'Model/data-derived artifacts stay local; clarify FloodNet rights before distribution.'],
        'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('xgboost','scikit-learn','pandas','numpy','pyarrow','matplotlib')},
        'source_manifest_sha256': digest(DATA/'selection_manifest.json'),
        'training_script_sha256': digest(Path(__file__))}
    json_write(output/'config_before_fit.json', config)
    provenance = []
    train = load_split('train', manifest, provenance)
    validation = load_split('validation', manifest, provenance)
    features, label = config['features'], config['label']
    models = {}
    candidate_rows, validation_metrics = [], []
    for name, prediction in baseline_predictions(validation).items():
        validation_metrics.append({'split':'validation','model':name,
            **evaluate(validation[label], prediction, validation.depth_lag_0_cm)})
    for candidate in CANDIDATES:
        model = XGBRegressor(n_estimators=candidate['n_estimators'], max_depth=candidate['max_depth'],
            learning_rate=.05, min_child_weight=10, reg_lambda=10, subsample=1, colsample_bytree=1,
            objective='reg:squarederror', eval_metric='mae', tree_method='hist',
            device='cpu', random_state=SEED, n_jobs=min(4, os.cpu_count() or 1))
        weights = np.where(train[label].to_numpy() >= 10, candidate['wet_weight'], 1.)
        t0 = time.perf_counter()
        model.fit(train[features], train[label], sample_weight=weights, verbose=False)
        predictions = np.maximum(model.predict(validation[features]), 0)
        row = {'split':'validation','model':candidate['id'],
            **evaluate(validation[label], predictions, validation.depth_lag_0_cm),
            'fit_seconds':time.perf_counter()-t0}
        candidate_rows.append(row)
        validation_metrics.append(row)
        models[candidate['id']] = model
        print(f'Fitted {candidate["id"]}: validation balanced_MAE={row["balanced_wet_nonwet_mae_cm"]:.4f}cm', flush=True)
    selected_name = choose(candidate_rows)
    selected = models[selected_name]
    json_write(output/'selection_before_test.json', {'selected_xgboost':selected_name,
        'selection_rule':config['selection_rule'], 'validation_candidates':candidate_rows})
    selected.save_model(output/'xgboost_selected.ubj')
    # No refit on validation, so identical model is used in validation and test.
    reloaded = XGBRegressor()
    reloaded.load_model(output/'xgboost_selected.ubj')
    np.testing.assert_allclose(selected.predict(validation[features].iloc[:100]),
                               reloaded.predict(validation[features].iloc[:100]))
    test = load_split('test', manifest, provenance)
    test_predictions = baseline_predictions(test)
    test_predictions['xgboost_selected'] = np.maximum(reloaded.predict(test[features]), 0)
    test_metrics = [{'split':'test','model':name, **evaluate(test[label],p,test.depth_lag_0_cm)}
                    for name,p in test_predictions.items()]
    metrics = pd.DataFrame(validation_metrics+test_metrics)
    metrics.to_csv(output/'metrics.csv', index=False, encoding='utf-8-sig')
    breakdown = []
    for dimension, labels in [('station', test.deployment_id), ('quarter', test.index.to_period('Q').astype(str))]:
        labels = np.asarray(labels)
        for group in sorted(set(labels)):
            mask = labels == group
            for name, prediction in test_predictions.items():
                breakdown.append({'dimension':dimension,'group':group,'model':name,
                    **evaluate(test[label].to_numpy()[mask],np.asarray(prediction)[mask],test.depth_lag_0_cm.to_numpy()[mask])})
    pd.DataFrame(breakdown).to_csv(output/'test_breakdown.csv',index=False,encoding='utf-8-sig')
    result = test[['deployment_id',label,'depth_lag_0_cm']].copy()
    for name,p in test_predictions.items():
        result[f'prediction_{name}_cm'] = p
    result.to_parquet(output/'test_predictions_local.parquet')
    pd.DataFrame({'feature':features,'gain_based_importance':selected.feature_importances_}).sort_values(
        'gain_based_importance',ascending=False).to_csv(output/'feature_importance.csv',index=False,encoding='utf-8-sig')
    plots = plot_test(test,test_predictions,label,output)
    json_write(output/'input_provenance.json',provenance)
    totals = {'train':len(train),'validation':len(validation),'test':len(test)}
    summary = {'status':'COMPLETED_EXPLORATORY_NOT_DEPLOYMENT_APPROVED', 'run_id':run_id,
        'selected_xgboost':selected_name,'counts':totals,'test_metrics':test_metrics,
        'validation_metrics':validation_metrics,'fit_and_report_seconds':time.perf_counter()-started,
        'model_roundtrip_check':'PASS','plots':plots,'config':config}
    json_write(output/'summary.json',summary)
    write_report(output,summary)
    print(json.dumps({'output':str(output),'selected':selected_name,'test':test_metrics},ensure_ascii=False,indent=2),flush=True)


def write_report(output,summary):
    lines=['# Kết quả train thử nghiệm baseline và XGBoost','',
        '**Chỉ nghiên cứu hồi cứu. Chưa được duyệt để cảnh báo thực tế; chưa kiểm chứng proc causal hoặc nhãn ngập thật.**','',
        f'Model được chọn bằng validation: **{summary["selected_xgboost"]}**. Không refit bằng validation; model đó được lưu và kiểm tra nạp lại trước dự đoán test.',
        f'Số cửa sổ: {summary["counts"]}. Dùng toàn bộ cửa sổ gốc của 3 trạm ứng viên, không dùng mask đối soát offline để loại ca khó.', '',
        '## Validation','',
        '| Model | MAE toàn tập (cm) | MAE nhãn >=10cm | MAE nhãn <10cm | MAE cân bằng |',
        '|---|---:|---:|---:|---:|']
    def fmt(v):
        return 'N/A' if v is None else f'{v:.4f}'
    for r in summary['validation_metrics']:
        lines.append(f'| {r["model"]} | {fmt(r["mae_cm"])} | {fmt(r["mae_wet_cm"])} | {fmt(r["mae_nonwet_cm"])} | {fmt(r["balanced_wet_nonwet_mae_cm"])} |')
    lines+=['','Tiêu chí chọn chỉ so 4 cấu hình XGBoost: MAE cân bằng nhỏ nhất trên validation, hòa thì MAE toàn tập và tên ID. Baseline vẫn có thể tốt hơn model được chọn; không ép kết luận XGBoost thắng. Trọng số 20 chỉ dùng train cho hai cấu hình, không đổi tỷ lệ mẫu validation/test.','',
        '## Test thăm dò — không dùng chọn cấu hình','',
        '| Model | MAE toàn tập (cm) | MAE nhãn >=10cm | MAE cân bằng | Recall >=10cm | Precision >=10cm | FP | FN | Recall chuyển từ <10 sang >=10cm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['test_metrics']:
        lines.append(f'| {r["model"]} | {fmt(r["mae_cm"])} | {fmt(r["mae_wet_cm"])} | {fmt(r["balanced_wet_nonwet_mae_cm"])} | {fmt(r["recall_wet"])} | {fmt(r["precision_wet"])} | {r["fp"]} | {r["fn"]} | {fmt(r["new_crossing_recall"])} |')
    first=summary['test_metrics'][0]
    lines += ['',f'Test có {first["wet_targets_ge_10cm"]} cửa sổ nhãn >=10 cm và {first["new_crossing_windows"]} cửa sổ hiện tại <10 cm nhưng nhãn tương lai >=10 cm. Đây không phải số trận ngập độc lập.',
        'Recall/precision/FP/FN là theo cửa sổ tại một bin tương lai. Không phải lead time cảnh báo trước khởi phát và không phải xác suất model đã được hiệu chuẩn.', '',
        '## Giới hạn bắt buộc','',
        '- Nhãn là số đo cuối trong (t+5,t+10 phút], không phải độ sâu tức thời đúng t+10. Baseline xu hướng ngoại suy tới nhãn bin +10 phút, có biến động lead time thực tế.',
        '- Bộ lọc proc chưa được xác minh tính nhân quả. Tổng hợp dữ liệu causal không khắc phục được bộ lọc archive không causal.',
        '- Trạm widely_proud_lizard có ca nghi box; raw/proc và nhãn chưa được xác minh hiện trường. Không tự coi nhãn xử lý là ground truth.',
        '- Dữ liệu 2023 đã từng được xem thống kê/biểu đồ. Test này chỉ thăm dò, không là holdout xác nhận độc lập chưa từng xem.',
        '- Phân chia thời gian và embargo giữ nguyên từ pipeline 5 phút; tính mùa vụ, khác trạm và số nhãn cao ít có thể chi phối kết quả.',
        '- Chưa bổ sung 7 trạm cần xem xét, chưa dùng dữ liệu ESP32, chưa tích hợp API/frontend.',
        '- Không chỉ báo cáo MAE ngày khô; xem cả MAE lúc cao, bỏ sót, báo giả, các lần chuyển ngưỡng và breakdown theo trạm/quý.', '',
        '## Artifacts','',
        '- `xgboost_selected.ubj`: model cục bộ, không tự động phê duyệt deploy.',
        '- `config_before_fit.json`, `selection_before_test.json`: cấu hình chốt trước fit và lựa chọn trước test.',
        '- `metrics.csv`, `test_breakdown.csv`, `summary.json`: kết quả tổng hợp.',
        '- `test_predictions_local.parquet`: đầu ra dẫn xuất dữ liệu; không công khai.',
        '- `input_provenance.json`: đường dẫn/số hàng/checksum các file đã dùng.',
        '- `feature_importance.csv`: mức quan trọng mô hình, không phải quan hệ nhân quả.',
        '- `test_*.png`: biểu đồ trường hợp test chọn sau khi chốt model, chỉ dùng chẩn đoán.', '',
        'Nguồn: FloodNet (New York University and The City University of New York). Giữ artifacts cục bộ. Không phân phối dữ liệu/trích xuất hoặc trọng số khi chưa làm rõ quyền. Phân tích/biểu diễn phái sinh được phép tuân CC BY-NC-SA 4.0 cùng Data Access License Agreement.', '']
    (output/'BAO_CAO_TRAIN.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
