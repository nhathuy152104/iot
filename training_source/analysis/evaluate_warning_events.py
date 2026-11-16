"""Part B: fixed-policy warning-event evaluation on frozen core validation only."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from archive_paths import resolve_archive_path
from delta_experiment import depth_from_delta, read_json
from diagnose_raw_expansion import EXPANDED, snapshot, verified_windows
from event_evaluation import POLICY, evaluate_station, summarize
from train_baseline_xgboost import BASE, baseline_predictions, digest, evaluate, json_write

MODELS = ['persistence', 'linear_trend_5min', 'three_d3_weight20', 'expanded_d3_weight20']


def write_report(output, table):
    primary = table.loc[(table.station == 'ALL') & (table.confirm_bins == 1)]
    sensitivity = table.loc[(table.station == 'ALL') & (table.confirm_bins == 2)]
    lines = ['# Đánh giá cảnh báo theo đợt — phần B', '',
        '**Hồi cứu trên validation cũ; không train, không đọc test, không đổi model được chọn.**', '',
        '## 1. Quy tắc chốt trước khi tính', '',
        '- Ngưỡng nghiên cứu 10 cm. Một đợt gồm ≥1 nhãn proc tương lai cao liên tiếp trên lưới quyết định 5 phút.',
        '- Không gộp qua mẫu thấp hoặc khoảng thiếu. Báo rõ đợt thiếu quan sát đầu/cuối; không coi là trận mưa độc lập.',
        '- Cảnh báo chính bật ngay khi dự báo ≥10 cm, tắt ở mẫu thấp/thiếu. Không cooldown, hysteresis hoặc gộp tùy ý.',
        '- Đối chiếu đúng slot dự báo: cảnh báo chỉ khớp đợt nếu hoạt động ở cửa sổ có nhãn cao thuộc đợt. '
        'Một cảnh báo dài có thể khớp nhiều đợt và một đợt có thể có nhiều lần bật; báo riêng độ phân mảnh.',
        '- Đợt cảnh báo không khớp nhãn cao nào là đợt báo giả theo archive. Cảnh báo vừa đúng vừa sai báo riêng số cửa sổ sai.',
        '- Mẫu số là tổng cửa sổ hợp lệ ×5 phút quy ra ngày-trạm quan sát tương đương, không phải tổng ngày lịch.', '',
        '## 2. Kết quả chính — bật từ một mẫu cao', '',
        '| Phương pháp | Đợt nhãn cao | Phát hiện | Bỏ sót | Đợt báo giả | Báo giả/100 ngày-trạm quan sát | Đợt phát hiện bị bật/tắt nhiều lần |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in primary.itertuples():
        lines.append(f'| {r.model} | {r.events} | {r.detected_events} | {r.missed_events} | {r.false_alarm_episodes} | '
                     f'{r.false_episodes_per_100_observed_station_days:.2f} | {r.fragmented_detected_events} |')
    first = primary.iloc[0]
    lines += ['', f'Độ phủ: **{first.windows:,} cửa sổ**, **{first.observed_station_day_equivalents:.2f} ngày-trạm quan sát tương đương**.',
        f'Trong {first.events} đợt, có **{first.onset_observed_events}** đợt có mẫu thấp liền trước và '
        f'**{first.fully_observed_events}** đợt quan sát đủ cả hai đầu. '
        'Không tính một mẫu đầu chuỗi đang cao là một lần khởi phát đã biết.', '',
        '## 3. Phát hiện sớm: chỉ ước lượng theo archive', '',
        'Với đợt có mẫu thấp liền trước, thời điểm vượt ngưỡng nằm giữa timestamp nhãn thấp cuối và nhãn cao đầu. '
        'Lấy hai đầu khoảng đó trừ thời điểm quyết định khớp đầu tiên để ra khoảng báo trước. '
        'Không lấy lúc bắt đầu một cảnh báo vốn đã báo giả trước đó để kéo dài lead time.', '',
        '“Sớm chắc theo khoảng” là cận dưới >0; vẫn chỉ theo đồng hồ proc archive, không phải xác nhận cảnh báo sớm ngoài thực tế. '
        'Trung vị chỉ ở các đợt phát hiện được có onset quan sát, không đại diện những đợt bỏ sót.', '',
        '| Phương pháp | Onset quan sát được / phát hiện | Sớm chắc theo khoảng | Có thể sớm | Muộn chắc | Trung vị cận dưới / trên (phút, chỉ nhóm phát hiện) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in primary.itertuples():
        lines.append(f'| {r.model} | {r.onset_observed_events} / {r.onset_observed_detected} | {r.certain_early_events} | '
                     f'{r.possibly_early_events} | {r.certain_late_events} | '
                     f'{r.lead_lower_median_minutes_detected_only:.2f} / {r.lead_upper_median_minutes_detected_only:.2f} |')
    lines += ['', '## 4. Độ nhạy cố định: chờ hai dự báo cao liên tiếp', '',
        'Chỉ bật ở dự báo cao thứ hai, không ghi lùi thời điểm cảnh báo về mẫu thứ nhất. '
        'Đây là đối chiếu chính sách định trước, không tự chọn cấu hình sau khi xem điểm.', '',
        '| Phương pháp | Phát hiện / tổng đợt | Bỏ sót | Đợt báo giả | Báo giả/100 ngày-trạm quan sát |',
        '|---|---:|---:|---:|---:|']
    for r in sensitivity.itertuples():
        lines.append(f'| {r.model} | {r.detected_events} / {r.events} | {r.missed_events} | {r.false_alarm_episodes} | '
                     f'{r.false_episodes_per_100_observed_station_days:.2f} |')
    lines += ['', '## 5. Cách đọc và giới hạn', '',
        '- Các đợt một mẫu không bị xóa. `event_metrics.csv` có thêm số đợt ≥2 mẫu và mức phát hiện của tập con đó.',
        '- Báo giả ở đây là so với proc archive, không có kiểm định thực địa. Sự kiện cùng trạm/thời tiết có thể tương quan.',
        '- Khoảng thiếu tách đợt và cảnh báo; tỷ lệ theo độ phủ không sửa được thiên lệch do mất mẫu. '
        'Đợt bị cắt có thể là mảnh của cùng một lần ngập; xem cờ censoring trong ledger.',
        '- Không có arrival time/bản sao pipeline lọc nguồn nên lead time không xác nhận dự báo trực tiếp 10 phút hoặc chuyển sang ESP32.',
        '- Validation đã dùng nhiều lần; không dùng kết quả phần B để tự đổi tiêu chí chọn model đã chốt bằng MAE.', '',
        '## 6. Artifacts và kiểm tra', '',
        '- `protocol_before_evaluation.json`: quy tắc trước nạp dữ liệu.',
        '- `event_metrics.csv`: theo phương pháp, chính sách và trạm, kèm tổng pooled.',
        '- `events_local.csv`, `alarms_local.csv`: ledger giữ cục bộ; có cờ đầu/cuối bị cắt, match và khoảng lead.',
        '- `replayed_window_metrics.csv`: tái lập chỉ số cửa sổ cũ; kết quả chính một mẫu khớp TP/FP/FN cũ.',
        '- `input_provenance.json`, `artifact_validation.json`: checksum và kiểm tra bảo toàn dữ liệu/model.', '',
        '**Giữ model raw ba mã làm ứng viên theo tiêu chí cũ; không triển khai hoặc đổi cấu hình cảnh báo thật.** '
        'Trước bước vận hành cần chốt mức bỏ sót/báo giả chấp nhận được và đánh giá trên dữ liệu xác nhận độc lập. '
        'Demo lịch sử có thể trình bày AI cạnh baseline, tách khỏi cảnh báo theo số đo ESP32.', '',
        'Nguồn: FloodNet (NYU/CUNY). Dữ liệu và artifacts giữ cục bộ, tuân giấy phép nguồn.']
    (output / 'BAO_CAO_CANH_BAO_THEO_DOT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run():
    output = BASE / 'diagnostic_runs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_warning_events')
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / 'protocol_before_evaluation.json', {**POLICY, 'models': MODELS, 'confirm_bins': [1, 2],
        'input_scope': 'only frozen 92215 validation windows at original two core stations; no train/test/cache table loading; hashes of existing raw run files (including train artifacts) checked for immutability'})
    config = read_json(EXPANDED / 'config_before_fit.json')
    core_run = resolve_archive_path(config['base_raw_run'])
    before = snapshot((core_run, EXPANDED))
    provenance = {}
    validation = verified_windows(core_run / 'window_provenance.json', 'validation', provenance)
    label, features = config['label'], config['features']
    assert len(validation) == 92215
    stored_path = EXPANDED / 'validation_predictions_local.parquet'
    known_hash = read_json(EXPANDED / 'artifact_validation.json')['artifact_sha256']
    assert digest(stored_path) == known_hash[stored_path.name]
    stored = pd.read_parquet(stored_path)
    assert stored.index.equals(validation.index)
    for name in ('deployment_id', label, 'depth_lag_0_cm'):
        assert stored[name].equals(validation[name])
    predictions = baseline_predictions(validation)
    for arm in ('three', 'expanded'):
        path = core_run / 'raw_delta_d3_weight20.ubj' if arm == 'three' else EXPANDED / 'expanded_d3_weight20.ubj'
        if arm == 'expanded':
            assert digest(path) == known_hash[path.name]
        model = XGBRegressor()
        model.load_model(path)
        assert model.get_booster().feature_names == features
        predictions[f'{arm}_d3_weight20'] = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[features]))
        provenance[str(path)] = {'sha256': digest(path), 'purpose': 'frozen-model replay only'}
    old_metrics = pd.read_csv(EXPANDED / 'metrics.csv').set_index('model')
    rows, event_rows, alarm_rows, replayed = [], [], [], []
    for name in MODELS:
        prediction = predictions[name]
        np.testing.assert_allclose(prediction, stored[f'prediction_{name}_cm'], atol=1e-6, rtol=1e-6)
        actual = evaluate(validation[label], prediction, validation.depth_lag_0_cm)
        for key, value in actual.items():
            if value is not None:
                np.testing.assert_allclose(value, old_metrics.loc[name, key], atol=1e-8, rtol=1e-8)
        replayed.append({'model': name, **actual})
        for confirmation in (1, 2):
            all_events, all_alarms, station_stats = [], [], []
            for station in sorted(validation.deployment_id.unique()):
                mask = validation.deployment_id.eq(station).to_numpy()
                frame = pd.DataFrame({'target': validation.loc[mask, label], 'prediction': prediction[mask],
                    'target_time': validation.loc[mask, 'audit_target_source_time_utc']})
                events, alarms, stats = evaluate_station(frame, confirmation)
                common = {'model': name, 'confirm_bins': confirmation, 'station': station}
                all_events.extend([{**common, **r} for r in events])
                all_alarms.extend([{**common, **r} for r in alarms])
                rows.append({**common, **stats})
                station_stats.append(stats)
            pooled = summarize(all_events, all_alarms, len(validation))
            for key in ('false_alarm_windows', 'missed_wet_windows', 'wet_windows'):
                pooled[key] = sum(s[key] for s in station_stats)
            if confirmation == 1:
                assert pooled['false_alarm_windows'] == actual['fp']
                assert pooled['missed_wet_windows'] == actual['fn']
            rows.append({'model': name, 'confirm_bins': confirmation, 'station': 'ALL', **pooled})
            event_rows.extend(all_events)
            alarm_rows.extend(all_alarms)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'event_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(event_rows).to_csv(output / 'events_local.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(alarm_rows).to_csv(output / 'alarms_local.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(replayed).to_csv(output / 'replayed_window_metrics.csv', index=False, encoding='utf-8-sig')
    for path in [stored_path, EXPANDED / 'config_before_fit.json', EXPANDED / 'metrics.csv',
                 core_run / 'window_provenance.json', EXPANDED / 'artifact_validation.json']:
        provenance[str(path)] = {'sha256': digest(path), 'purpose': 'frozen validation provenance'}
    json_write(output / 'input_provenance.json', provenance)
    assert snapshot((core_run, EXPANDED)) == before
    summary = {'status': 'PASS', 'test_loaded': False, 'train_loaded': False, 'fit_called': False,
        'frozen_files_unchanged': len(before), 'models': MODELS,
        'counts': {'validation_windows': len(validation), 'stations': sorted(validation.deployment_id.unique())},
        'metrics_all': [r for r in rows if r['station'] == 'ALL']}
    json_write(output / 'summary.json', summary)
    write_report(output, table)
    json_write(output / 'artifact_validation.json', {'status': 'PASS',
        'checks': ['frozen validation checksums', 'four methods replay stored predictions/window metrics',
            'one-bin event evaluation reproduces old FP/FN windows', 'per-station pooled counts reconcile',
            'frozen original run snapshots unchanged', 'no train/test/cache table loading or fitting'],
        'frozen_sha256': before,
        'input_sha256': {p: item['sha256'] for p, item in provenance.items()},
        'output_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        'code_sha256': {name: digest(BASE / name) for name in ['event_evaluation.py', 'evaluate_warning_events.py']}})
    print(f'EVENT EVALUATION COMPLETED: {output}', flush=True)
    print(table.loc[table.station == 'ALL', ['model', 'confirm_bins', 'events', 'detected_events',
        'false_alarm_episodes', 'false_episodes_per_100_observed_station_days']].to_string(index=False))
    return output


if __name__ == '__main__':
    run()
