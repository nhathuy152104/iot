"""Read-only replay and error diagnosis of the ORIGINAL, frozen training run.

Writes a new local diagnostic directory, never mutates original artifacts/rows.
Old test is explicitly exploratory. No fitting, filtering or model selection.
"""
import argparse
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from delta_experiment import DEFAULT_ORIGIN, METRIC, original_config, read_json, verified_split
from train_baseline_xgboost import BASE, DATA, baseline_predictions, digest, evaluate, json_write


def stages(frame, label):
    """Retrospective, mutually exclusive error strata; NEVER online features."""
    current, future = frame.depth_lag_0_cm, frame[label]
    change = future - current
    return np.select([
        (current < 10) & (future >= 10), (current >= 10) & (future < 10),
        change > 1, change < -1, (current >= 10) & (future >= 10),
    ], ['new_crossing', 'down_crossing', 'rising_gt_1cm', 'falling_gt_1cm', 'stable_high'], default='stable_low')


def support_masks(frame, train, label):
    train_max = float(train[label].max())
    maxima = train.groupby('deployment_id')[label].max()
    station_max = frame.deployment_id.map(maxima).to_numpy()
    return {
        'all': np.ones(len(frame), dtype=bool),
        'future_gt_pooled_train_max': frame[label].to_numpy() > train_max,
        'future_le_pooled_train_max': frame[label].to_numpy() <= train_max,
        'future_gt_own_station_train_max': frame[label].to_numpy() > station_max,
        'history_disagreement_flag': frame.review_raw_proc_diff_ge_10cm_history.to_numpy(dtype=bool),
        'history_no_disagreement_flag': ~frame.review_raw_proc_diff_ge_10cm_history.to_numpy(dtype=bool),
    }


def error_rows(frame, label, predictions, masks, dimension, split):
    result = []
    y = frame[label].to_numpy()
    current = frame.depth_lag_0_cm.to_numpy()
    for group, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        for name, prediction in predictions.items():
            prediction = np.asarray(prediction)
            error = np.abs(y[mask] - prediction[mask])
            wet = y[mask] >= 10
            result.append({'split': split, 'dimension': dimension, 'group': group, 'model': name,
                **evaluate(y[mask], prediction[mask], current[mask]),
                'absolute_error_sum_cm': float(error.sum()),
                'wet_absolute_error_sum_cm': float(error[wet].sum()),
                'target_max_cm': float(y[mask].max()),
                'prediction_max_cm': float(prediction[mask].max())})
    return result


def candidate_segments(frame, label, predictions, split):
    """Group active decision windows by station and <=30min gap, not true storms.

    Activation uses future labels and frozen predictions solely for diagnosis.
    Missing/inactive windows do not become fabricated observations.
    """
    original = np.asarray(predictions['original_absolute_selected'])
    active = (frame[label].to_numpy() >= 10) | (frame.depth_lag_0_cm.to_numpy() >= 10) | (original >= 10)
    rows, definitions = [], []
    for station in sorted(frame.deployment_id.unique()):
        positions = np.flatnonzero(frame.deployment_id.eq(station).to_numpy() & active)
        if not len(positions):
            continue
        times = frame.index[positions]
        groups = (times.to_series().diff() > pd.Timedelta(minutes=30)).cumsum().to_numpy()
        for group in np.unique(groups):
            selected = positions[groups == group]
            block = frame.iloc[selected]
            segment_id = f'{split}_{station}_{block.index[0]:%Y%m%dT%H%M}'
            definition = {'split': split, 'segment_id': segment_id, 'station': station,
                'start_utc': str(block.index.min()), 'end_utc': str(block.index.max()),
                'active_windows': len(block), 'label_max_cm': float(block[label].max()),
                'current_max_cm': float(block.depth_lag_0_cm.max()),
                'history_disagreement_windows': int(block.review_raw_proc_diff_ge_10cm_history.sum()),
                'history_jump_windows': int(block.review_proc_jump_history.sum()),
                'stage_counts': pd.Series(stages(block, label)).value_counts().to_dict()}
            definitions.append(definition)
            mask = np.zeros(len(frame), dtype=bool)
            mask[selected] = True
            for row in error_rows(frame, label, predictions, {segment_id: mask}, 'candidate_segment', split):
                row.update({'station': station, 'start_utc': definition['start_utc'], 'end_utc': definition['end_utc']})
                rows.append(row)
    return rows, definitions


def cached_context(station, start, end, cache_provenance):
    """Read only matching, complete existing cache months; never rebuild cache."""
    root = DATA.parents[1]
    manifest = read_json(DATA / 'selection_manifest.json')
    from five_minute.pipeline import month_key
    parts = []
    for filename in manifest['source_files']:
        source = root / filename
        year, month = month_key(source)
        month_start = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
        month_end = month_start + pd.offsets.MonthBegin(1)
        if month_end <= start or month_start > end:
            continue
        paths = list((DATA / 'local_cache').glob(f'{source.stem}_*/{station}.parquet'))
        if len(paths) != 1:
            raise ValueError(f'Expected exactly one existing cache for {station}/{filename}; got {len(paths)}')
        path = paths[0]
        marker = read_json(path.parent / 'complete.json')
        if (marker['size'] != source.stat().st_size or marker['mtime_ns'] != source.stat().st_mtime_ns
                or station not in marker['station_ids']):
            raise ValueError(f'Stale cache: {path}')
        frame = pd.read_parquet(path)
        parts.append(frame.loc[(frame.index >= start) & (frame.index <= end)])
        cache_provenance[str(path)] = {'path': str(path), 'sha256': digest(path),
            'source': str(source), 'fingerprint_verified': True,
            'complete_marker_sha256': digest(path.parent / 'complete.json')}
    if not parts:
        raise ValueError(f'No source context for {station}/{start}')
    return pd.concat(parts).sort_index(kind='stable')


def plot_case(definition, frame, label, predictions, output, cache_provenance):
    start = pd.Timestamp(definition['start_utc']) - pd.Timedelta(minutes=30)
    end = pd.Timestamp(definition['end_utc']) + pd.Timedelta(minutes=40)
    station = definition['station']
    source = cached_context(station, start, end, cache_provenance)
    mask = frame.deployment_id.eq(station).to_numpy() & (frame.index >= start) & (frame.index <= end)
    view = frame.loc[mask]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout='constrained')
    for column in ('depth_raw_mm', 'depth_filt_mm', 'depth_proc_mm'):
        axes[0].scatter(source.index, source[column] / 10, s=9, alpha=.6, label=column)
    axes[0].set_title(f'{definition["split"].upper()} DIAGNOSTIC ONLY | {station}\nRaw/filt/proc at actual source time; no interpolation, no truth verification')
    axes[0].set_xlabel('Actual source timestamp UTC')
    axes[1].scatter(view.index, view[label], s=22, color='black', label='future archived target BIN')
    for name, prediction in predictions.items():
        axes[1].scatter(view.index, np.asarray(prediction)[mask], s=11, alpha=.65, label=name)
    axes[1].set_xlabel('Decision timestamp UTC (future target in (t+5,t+10])')
    for ax in axes:
        ax.axhline(10, color='red', linestyle='--', linewidth=.7)
        ax.set_ylabel('Depth cm')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    filename = definition['segment_id'] + '.png'
    fig.savefig(output / filename, dpi=120)
    plt.close(fig)
    raw_proc_diff = (source.depth_raw_mm - source.depth_proc_mm).abs() / 10
    return {'segment_id': definition['segment_id'], 'plot': filename,
        'source_rows': len(source), 'source_proc_max_cm': float(source.depth_proc_mm.max() / 10),
        'raw_proc_diff_ge_10cm_rows': int(raw_proc_diff.ge(10).sum()),
        'null_proc_source_rows': int(source.depth_proc_mm.isna().sum()),
        'quality_conclusion': 'UNVERIFIED: raw/proc agreement or plausible shape is not independent flood confirmation'}


def diagnose(origin, require_improvement=False):
    origin = origin.resolve()
    config, manifest = original_config(origin)
    provenance = []
    train = verified_split('train', origin, manifest, provenance)
    model = XGBRegressor()
    model.load_model(origin / 'xgboost_selected.ubj')
    replay = {}
    for split in ('validation', 'test'):
        frame = verified_split(split, origin, manifest, provenance, allowed=('validation', 'test'))
        predictions = baseline_predictions(frame)
        predictions['original_absolute_selected'] = np.maximum(model.predict(frame[config['features']]), 0)
        replay[split] = (frame, predictions)
    # First assert the exact persisted symptom, not just successful loading.
    old_summary = read_json(origin / 'summary.json')
    old_decision = read_json(origin / 'selection_before_test.json')['selected_xgboost']
    for split, (frame, predictions) in replay.items():
        for name, prediction in predictions.items():
            old_name = ('xgboost_selected' if split == 'test' else old_decision) if name == 'original_absolute_selected' else name
            expected = next(r for r in old_summary[f'{split}_metrics'] if r['model'] == old_name)
            actual = evaluate(frame[config['label']], prediction, frame.depth_lag_0_cm)
            for metric, value in actual.items():
                if value is None:
                    assert expected[metric] is None
                else:
                    np.testing.assert_allclose(value, expected[metric], rtol=1e-8, atol=1e-8)
    validation, predictions = replay['validation']
    baseline = evaluate(validation[config['label']], predictions['persistence'], validation.depth_lag_0_cm)[METRIC]
    original = evaluate(validation[config['label']], predictions['original_absolute_selected'], validation.depth_lag_0_cm)[METRIC]
    print(f'Frozen-model replay PASS; validation persistence={baseline:.9f}, original={original:.9f}', flush=True)
    if require_improvement:
        assert original < baseline, 'REPRODUCED: original XGBoost does not beat persistence'
        return None
    output = BASE / 'diagnostic_runs' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    all_rows, segments, segment_rows, distributions, plots = [], [], [], [], []
    cache_provenance = {}
    label = config['label']
    for split, frame in [('train', train)] + [(s, f) for s, (f, _) in replay.items()]:
        for station, part in [('ALL_CORE', frame)] + list(frame.groupby('deployment_id')):
            for column in ('depth_lag_0_cm', label):
                distributions.append({'split': split, 'station': station, 'column': column,
                    'rows': len(part), 'wet_ge_10cm': int(part[column].ge(10).sum()),
                    'max_cm': float(part[column].max()), 'q99_cm': float(part[column].quantile(.99)),
                    'q999_cm': float(part[column].quantile(.999))})
    for split, (frame, predictions) in replay.items():
        all_rows.extend(error_rows(frame, label, predictions, support_masks(frame, train, label), 'support', split))
        stage = stages(frame, label)
        all_rows.extend(error_rows(frame, label, predictions, {s: stage == s for s in np.unique(stage)}, 'stage', split))
        all_rows.extend(error_rows(frame, label, predictions,
            {s: frame.deployment_id.eq(s).to_numpy() for s in frame.deployment_id.unique()}, 'station', split))
        rows, definitions = candidate_segments(frame, label, predictions, split)
        segment_rows.extend(rows)
        segments.extend(definitions)
        ranked = sorted([r for r in rows if r['model'] == 'original_absolute_selected'],
            key=lambda r: r['absolute_error_sum_cm'], reverse=True)
        for row in ranked[:3]:
            definition = next(d for d in definitions if d['segment_id'] == row['group'])
            plots.append(plot_case(definition, frame, label, predictions, output, cache_provenance))
    pd.DataFrame(all_rows).to_csv(output / 'error_breakdown.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(segment_rows).to_csv(output / 'candidate_segment_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(distributions).to_csv(output / 'depth_distributions.csv', index=False, encoding='utf-8-sig')
    json_write(output / 'candidate_segments.json', segments)
    json_write(output / 'input_provenance.json', provenance)
    json_write(output / 'cache_provenance.json', list(cache_provenance.values()))
    json_write(output / 'case_context.json', plots)
    summary = {'status': 'DIAGNOSTIC_ONLY_NO_FIT_NO_SELECTION_NO_ROWS_REMOVED',
        'origin': str(origin), 'output': str(output),
        'script_sha256': digest(Path(__file__)), 'config_sha256': digest(origin / 'config_before_fit.json'),
        'model_sha256': digest(origin / 'xgboost_selected.ubj'),
        'replay': 'PASS: all original validation/test metrics reproduced from saved model and checksummed inputs',
        'validation_gap_cm': original - baseline,
        'candidate_segment_definition': 'same-station active windows (current/target/original prediction >=10cm), gaps <=30min; retrospective, not independent verified storms',
        'plots': plots, 'limits': config['scientific_limits']}
    json_write(output / 'summary.json', summary)
    write_report(output, summary, all_rows, segment_rows, distributions, origin)
    print(f'Diagnostic report: {output}', flush=True)
    return output


def write_report(output, summary, rows, segment_rows, distributions, origin):
    lines = ['# Chẩn đoán lượt train đầu tiên', '',
        '**Không fit lại, không chọn model, không loại hàng. Test cũ chỉ dùng chẩn đoán hồi cứu.**', '',
        summary['replay'], '',
        f'Khoảng thua baseline trên validation: **{summary["validation_gap_cm"]:+.6f} cm MAE cân bằng**.', '',
        '## 1. Phạm vi độ sâu', '',
        '| Tập | Trạm | Số nhãn ≥10cm | Nhãn lớn nhất cm |', '|---|---|---:|---:|']
    for r in distributions:
        if r['column'] == 'target_depth_bin_t_plus_10min_cm':
            lines.append(f'| {r["split"]} | {r["station"]} | {r["wet_ge_10cm"]} | {r["max_cm"]:.2f} |')
    lines += ['', 'Nhãn vượt cực đại train là bằng chứng thiếu độ phủ, không chứng minh nhãn sai. '
        'Cây hồi quy dự báo độ sâu tuyệt đối khó ngoại suy mức chưa gặp; kiểm tra bảng support bên dưới thay vì mặc định đó là nguyên nhân duy nhất.', '',
        '## 2. Phân rã sai số model cũ', '',
        '| Tập | Nhóm | Model | Hàng | Nhãn ≥10 | MAE ≥10cm | Tổng lỗi nhãn ≥10cm | FP | FN |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['dimension'] not in ('support', 'stage') or r['group'].startswith('history_'):
            continue
        wet_mae = 'N/A' if r['mae_wet_cm'] is None else f'{r["mae_wet_cm"]:.3f}'
        lines.append(f'| {r["split"]} | {r["group"]} | {r["model"]} | {r["windows"]} | {r["wet_targets_ge_10cm"]} | '
            f'{wet_mae} | {r["wet_absolute_error_sum_cm"]:.2f} | {r["fp"]} | {r["fn"]} |')
    lines += ['', 'Các pha dùng chênh lệch NHÃN TƯƠNG LAI để phân tích: chuyển ngưỡng lên/xuống trước; '
        'sau đó tăng/giảm >1cm; còn lại giữ mức cao/thấp. Chỉ là phân tầng hồi cứu, không phải feature hay bộ lọc online.', '',
        '## 3. Tác động của trọng số — chỉ đối chiếu validation cũ', '',
        '| Cấu hình cũ | MAE cân bằng | MAE <10cm | FP | FN |', '|---|---:|---:|---:|---:|']
    old = pd.read_csv(origin / 'metrics.csv')
    for _, r in old.loc[(old.split == 'validation') & old.model.str.startswith('d')].iterrows():
        lines.append(f'| {r.model} | {r[METRIC]:.4f} | {r.mae_nonwet_cm:.4f} | {int(r.fp)} | {int(r.fn)} |')
    lines += ['', 'Đối chiếu từng cặp cùng depth/số cây để thấy đánh đổi của weight20. Không điều chỉnh trọng số theo test.', '',
        '## 4. Các đoạn đóng góp lỗi lớn nhất', '',
        'Nhóm các cửa sổ hoạt động cùng trạm, cách nhau không quá 30 phút. Đây KHÔNG phải trận mưa/ngập độc lập; '
        'số hàng là cửa sổ hoạt động, không phải toàn bộ độ phủ thời gian.', '',
        '| Tập | Đoạn | Hàng | Tổng lỗi XGB cm | Tổng lỗi persistence cm |', '|---|---|---:|---:|---:|']
    for split in ('validation', 'test'):
        ranked = sorted([r for r in segment_rows if r['split'] == split and r['model'] == 'original_absolute_selected'],
            key=lambda r: r['absolute_error_sum_cm'], reverse=True)
        for r in ranked[:5]:
            p = next(p for p in segment_rows if p['group'] == r['group'] and p['model'] == 'persistence')
            lines.append(f'| {split} | {r["group"]} | {r["windows"]} | {r["absolute_error_sum_cm"]:.2f} | {p["absolute_error_sum_cm"]:.2f} |')
    lines += ['', '## 5. Biểu đồ nguồn và dự đoán', '']
    for case in summary['plots']:
        lines.append(f'- [{case["segment_id"]}]({case["plot"]}): {case["source_rows"]} hàng nguồn; '
            f'{case["raw_proc_diff_ge_10cm_rows"]} hàng raw/proc lệch ≥10cm. Chưa xác minh ngập thật.')
    lines += ['', 'Không suy ra nhãn đúng chỉ vì raw/filt/proc trùng nhau. Không có dữ liệu hiện trường/thời tiết độc lập '
        'trong phân tích này. Không loại ca cực trị để làm đẹp điểm.', '',
        '## 6. Hành động có kiểm soát', '',
        '- Thử delta target trên cùng train/validation; giữ nguyên 4 cấu hình, feature, trọng số và quy tắc chọn.',
        '- Không dùng các phân tầng tương lai, nhóm đoạn hoặc biểu đồ test làm feature/bộ lọc dữ liệu.',
        '- Xác minh causality/độ trễ của bộ lọc FloodNet trước tuyên bố hiệu quả thời gian thực.',
        '- Bổ sung trạm/mưa/lịch sử dài hơn phải là thí nghiệm khác. Cần tập xác nhận độc lập trước khẳng định tổng quát hóa.', '',
        'Nguồn: FloodNet (New York University and The City University of New York). '
        'Giữ toàn bộ artifacts cục bộ; không phân phối nguồn/trích xuất/trọng số khi chưa làm rõ quyền. '
        'Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0 cùng Data Access License Agreement.', '']
    (output / 'BAO_CAO_CHAN_DOAN.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, default=DEFAULT_ORIGIN)
    parser.add_argument('--require-improvement', action='store_true',
        help='Replay-only red-capable check: fail if original model does not beat persistence; writes nothing')
    args = parser.parse_args()
    diagnose(args.origin, args.require_improvement)
