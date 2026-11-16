"""Part A: frozen raw expansion diagnosis; NO fit, test or new-station validation."""
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from archive_paths import resolve_archive_path
from delta_experiment import depth_from_delta, read_json
from diagnose_training import stages
from five_minute.pipeline import TRAIN_END, EMBARGO, split_masks, month_key
from raw_branch.pipeline import make_raw_grid
from train_baseline_xgboost import BASE, baseline_predictions, digest, evaluate, json_write

EXPANDED = BASE / 'training_runs' / '20261110T050356_109133Z_raw_expanded'


def composition(frame, label):
    wet = frame[label].ge(10)
    count = int(wet.sum())
    return {'windows': len(frame), 'wet_windows': count, 'wet_share': float(wet.mean()),
        'weight20_sum': len(frame) + 19 * count,
        'wet_weight20_share': 20 * count / (len(frame) + 19 * count),
        'label_max_cm': float(frame[label].max()),
        'label_p99_cm': float(frame[label].quantile(.99)),
        'delta_mean_cm': float((frame[label] - frame.depth_lag_0_cm).mean()),
        'delta_abs_p99_cm': float((frame[label] - frame.depth_lag_0_cm).abs().quantile(.99))}


def transition_counts(y, three, expanded):
    wet, a, b = np.asarray(y) >= 10, np.asarray(three) >= 10, np.asarray(expanded) >= 10
    return {'fp_resolved': int((~wet & a & ~b).sum()),
        'fp_introduced': int((~wet & ~a & b).sum()),
        'fn_resolved': int((wet & ~a & b).sum()),
        'fn_introduced': int((wet & a & ~b).sum())}


def paired_groups(frame, label, three, expanded, groups, dimension):
    y = frame[label].to_numpy()
    wet = y >= 10
    change = np.abs(y - expanded) - np.abs(y - three)
    rows = []
    for group in sorted(set(groups)):
        mask = np.asarray(groups) == group
        high, low = mask & wet, mask & ~wet
        contribution = .5 * (change[high].sum() / wet.sum() + change[low].sum() / (~wet).sum())
        rows.append({'dimension': dimension, 'group': str(group), 'windows': int(mask.sum()),
            'wet_windows': int(high.sum()), 'three_absolute_error_sum_cm': float(np.abs(y[mask] - three[mask]).sum()),
            'expanded_absolute_error_sum_cm': float(np.abs(y[mask] - expanded[mask]).sum()),
            'absolute_error_change_cm': float(change[mask].sum()),
            'balanced_change_contribution_cm': float(contribution),
            'mean_predicted_delta_change_cm': float((expanded[mask] - three[mask]).mean()),
            **transition_counts(y[mask], three[mask], expanded[mask])})
    return rows


def verified_windows(manifest, split, provenance):
    if split not in ('train', 'validation'):
        raise ValueError('Test is forbidden')
    rows = [r for r in read_json(manifest) if r['split'] == split]
    if not rows:
        raise ValueError('Only explicit train/validation windows allowed')
    parts = []
    for row in rows:
        path = resolve_archive_path(row['path'])
        if digest(path) != row['sha256']:
            raise ValueError(f'Window checksum mismatch: {path}')
        frame = pd.read_parquet(path)
        assert len(frame) == row['rows']
        assert frame.deployment_id.eq(row['station']).all()
        assert split_masks(frame.index)[split].all()
        provenance[str(path)] = {'sha256': row['sha256'], 'purpose': split}
        parts.append(frame)
    return pd.concat(parts).sort_index(kind='stable')


def train_disagreement(core_run, expanded_run, train, provenance):
    """Contemporaneous raw median vs last proc, on train only; never features."""
    records = {}
    for run in (core_run, expanded_run):
        for row in read_json(run / 'source_provenance.json')['cache_files']:
            path = resolve_archive_path(row['path'])
            year, month = month_key(Path(path.parent.name.rsplit('_', 1)[0] + '.csv'))
            if (year, month) >= (2023, 1):
                continue  # Reject by month BEFORE reading content/checksum.
            records[str(path)] = row
    parts = {}
    for path_text, row in records.items():
        path = Path(path_text)
        if digest(path) != row['sha256']:
            raise ValueError(f'Cache checksum mismatch: {path}')
        source = pd.read_parquet(path, columns=['depth_raw_mm', 'depth_proc_mm'])
        source = source.loc[source.index < TRAIN_END - EMBARGO]
        provenance[str(path)] = {'sha256': row['sha256'], 'purpose': 'train-only raw/proc comparison'}
        parts.setdefault(row['station'], []).append(source)
    results = []
    for station, frames in parts.items():
        grid = make_raw_grid(pd.concat(frames).sort_index(kind='stable'))
        windows = train.loc[train.deployment_id.eq(station)]
        current_proc = grid.depth_proc_mm.reindex(windows.index) / 10
        difference = windows.depth_lag_0_cm - current_proc
        valid = np.isfinite(current_proc) & current_proc.ge(0)
        results.append({'station': station, 'eligible_train_windows': len(windows),
            'valid_contemporaneous_proc_windows': int(valid.sum()),
            'raw_median_minus_last_proc_mean_cm': float(difference[valid].mean()),
            'raw_proc_abs_diff_p99_cm': float(difference[valid].abs().quantile(.99)),
            'raw_proc_abs_diff_ge10cm_windows': int((valid & difference.abs().ge(10)).sum()),
            'negative_raw_history_share': float(windows.audit_negative_raw_history.mean()),
            'sparse_raw_history_share': float(windows.audit_sparse_raw_history.mean()),
            'current_source_rows_median': float(windows.audit_source_rows_current_bin.median())})
    return results


def snapshot(runs):
    return {str(p): digest(p) for run in runs for p in run.rglob('*') if p.is_file()}


def run():
    config = read_json(EXPANDED / 'config_before_fit.json')
    core_run = resolve_archive_path(config['base_raw_run'])
    frozen = snapshot((core_run, EXPANDED))
    output = BASE / 'diagnostic_runs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_raw_expansion')
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / 'protocol_before_analysis.json', {
        'scope': 'A only; retrospective exploratory; no fitting or model/station selection',
        'hypotheses': ['wet sample/weight composition shift', 'station distribution/raw-proc differences',
            'regression localized by validation station, target range or phase'],
        'phase': 'new/down crossing first; else future-current >1cm/<-1cm; else stable high/low',
        'depth_bands_cm': '[0,1),[1,10),[10,20),[20,50),[50,inf)',
        'test_policy': 'no test; no new-station validation; cache months strictly before 2023',
        'limits': 'Descriptive associations, not causal attribution; no event-warning evaluation (part B).'})
    provenance = {}
    core = verified_windows(core_run / 'window_provenance.json', 'train', provenance)
    added = verified_windows(EXPANDED / 'candidate_window_provenance.json', 'train', provenance)
    train = pd.concat([core, added]).sort_index(kind='stable')
    validation = verified_windows(core_run / 'window_provenance.json', 'validation', provenance)
    label, features = config['label'], config['features']
    assert (len(core), len(added), len(validation)) == (270312, 271905, 92215)
    counts = []
    for name, frame in [('core', core), ('added', added), ('expanded', train), ('validation_core', validation)]:
        counts.append({'group': name, **composition(frame, label)})
    for station, frame in train.groupby('deployment_id'):
        counts.append({'group': station, **composition(frame, label)})
    pd.DataFrame(counts).to_csv(output / 'train_composition.csv', index=False, encoding='utf-8-sig')
    distributions = []
    for name, frame in [('core', core), ('added', added), ('validation_core', validation)]:
        for column in [*features, label]:
            values = frame[column]
            distributions.append({'group': name, 'variable': column, 'mean': values.mean(),
                'min': values.min(), 'p01': values.quantile(.01), 'p50': values.median(),
                'p95': values.quantile(.95), 'p99': values.quantile(.99), 'max': values.max()})
    pd.DataFrame(distributions).to_csv(output / 'feature_distributions.csv', index=False, encoding='utf-8-sig')
    quality = train_disagreement(core_run, EXPANDED, train, provenance)
    pd.DataFrame(quality).to_csv(output / 'train_raw_proc_comparison.csv', index=False, encoding='utf-8-sig')
    stored = pd.read_parquet(EXPANDED / 'validation_predictions_local.parquet')
    assert stored.index.equals(validation.index)
    for column in ('deployment_id', label, 'depth_lag_0_cm'):
        assert stored[column].equals(validation[column])
    predictions = baseline_predictions(validation)
    for candidate in config['candidate_model_configs']:
        for arm in ('three', 'expanded'):
            name = arm + '_' + candidate['id']
            path = core_run / f'raw_delta_{candidate["id"]}.ubj' if arm == 'three' else EXPANDED / f'{name}.ubj'
            model = XGBRegressor()
            model.load_model(path)
            assert model.get_booster().feature_names == features
            predictions[name] = depth_from_delta(validation.depth_lag_0_cm, model.predict(validation[features]))
    metrics = []
    expected = pd.read_csv(EXPANDED / 'metrics.csv').set_index('model')
    for name, prediction in predictions.items():
        np.testing.assert_allclose(prediction, stored[f'prediction_{name}_cm'], atol=1e-6, rtol=1e-6)
        actual = evaluate(validation[label], prediction, validation.depth_lag_0_cm)
        for key, value in actual.items():
            if value is not None:
                np.testing.assert_allclose(value, expected.loc[name, key], atol=1e-8, rtol=1e-8)
        metrics.append({'model': name, **actual})
    pd.DataFrame(metrics).to_csv(output / 'replayed_metrics.csv', index=False, encoding='utf-8-sig')
    a, b = predictions['three_d3_weight20'], predictions['expanded_d3_weight20']
    target = validation[label].to_numpy()
    groups = {'station': validation.deployment_id.to_numpy(), 'phase': stages(validation, label),
        'target_band': pd.cut(target, [0, 1, 10, 20, 50, np.inf], right=False).astype(str),
        'month': validation.index.strftime('%Y-%m'),
        'negative_raw_history': validation.audit_negative_raw_history.astype(str).to_numpy(),
        'sparse_raw_history': validation.audit_sparse_raw_history.astype(str).to_numpy()}
    rows = []
    balanced_change = expected.loc['expanded_d3_weight20', 'balanced_wet_nonwet_mae_cm'] - expected.loc['three_d3_weight20', 'balanced_wet_nonwet_mae_cm']
    for dimension, values in groups.items():
        partition = paired_groups(validation, label, a, b, values, dimension)
        assert sum(r['windows'] for r in partition) == len(validation)
        np.testing.assert_allclose(sum(r['balanced_change_contribution_cm'] for r in partition), balanced_change, atol=1e-10)
        rows.extend(partition)
    pd.DataFrame(rows).to_csv(output / 'paired_error_decomposition.csv', index=False, encoding='utf-8-sig')
    # Rank station-days, not independent flood events. All windows remain in metrics.
    daily = paired_groups(validation, label, a, b,
        validation.deployment_id.to_numpy() + '_' + validation.index.strftime('%Y-%m-%d').to_numpy(), 'station_day')
    daily = pd.DataFrame(daily).sort_values('balanced_change_contribution_cm', ascending=False)
    daily.to_csv(output / 'station_day_changes.csv', index=False, encoding='utf-8-sig')
    chosen = pd.concat([daily.head(3), daily.tail(3)]).drop_duplicates('group')
    plot_names = []
    for _, row in chosen.iterrows():
        station, day = row['group'].rsplit('_', 1)
        mask = validation.deployment_id.eq(station).to_numpy() & (validation.index.strftime('%Y-%m-%d') == day)
        view = validation.loc[mask]
        fig, ax = plt.subplots(figsize=(12, 4), layout='constrained')
        ax.plot(view.index, view.depth_lag_0_cm, label='current raw median (persistence)')
        ax.plot(view.index, view[label], label='future proc label (decision clock)')
        ax.plot(view.index, a[mask], label='three d3 weight20')
        ax.plot(view.index, b[mask], label='expanded d3 weight20')
        ax.axhline(10, color='grey', linestyle='--')
        ax.set(title=f'{station} / {day} UTC - eligible windows only; gaps not interpolated', ylabel='cm')
        # Break lines across missing eligible windows instead of bridging gaps.
        for line in ax.lines[:4]:
            line.set_linestyle('None')
            line.set_marker('.')
            line.set_markersize(3)
        ax.legend(fontsize=8)
        name = row['group'] + '.png'
        fig.savefig(output / name, dpi=130)
        plt.close(fig)
        plot_names.append(name)
    assert snapshot((core_run, EXPANDED)) == frozen, 'Frozen run mutated'
    json_write(output / 'input_provenance.json', provenance)
    summary = {'status': 'PASS', 'counts': {r['group']: r for r in counts[:4]},
        'balanced_change_cm': balanced_change, 'threshold_transitions': transition_counts(target, a, b),
        'decomposition': rows, 'plots': plot_names, 'test_loaded': False, 'fit_called': False,
        'frozen_run_files_unchanged': len(frozen), 'verified_input_files': len(provenance)}
    json_write(output / 'summary.json', summary)
    write_report(output, summary, daily, quality)
    json_write(output / 'artifact_validation.json', {'status': 'PASS',
        'checks': ['window/cache checksums', 'all 10 validation methods replayed',
            'all partition contributions sum to pooled balanced MAE change', 'frozen run snapshots unchanged'],
        'input_sha256': {p: digest(Path(p)) for p in provenance},
        'output_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()}})
    print(f'DIAGNOSIS COMPLETED: {output}', flush=True)
    return output


def write_report(output, summary, daily, quality):
    c = summary['counts']
    lines = ['# Phân tích tác động mở rộng dữ liệu raw — phần A', '',
        '**Chỉ phân tích hồi cứu; không train lại, không đọc test, không chọn lại model.**', '',
        '## 1. Tỷ trọng train và trọng số', '',
        '| Nhóm | Cửa sổ | Nhãn ≥10 cm | Tỷ lệ nhãn cao | Phần tổng trọng số ×20 của nhãn cao |',
        '|---|---:|---:|---:|---:|']
    for name in ('core', 'added', 'expanded'):
        r = c[name]
        lines.append(f'| {name} | {r["windows"]:,} | {r["wet_windows"]:,} | {r["wet_share"]:.2%} | {r["wet_weight20_share"]:.2%} |')
    lines += ['', 'Trọng số ×20 giữ nguyên nhưng phân bố đóng góp vào hàm mất mát đổi đáng kể. '
        'Không suy ra quan hệ nhân quả hoặc hướng thay đổi dự báo chỉ từ tỷ trọng này. '
        'Các cấu hình uniform cũng kém hơn sau mở rộng, nên không thể quy toàn bộ suy giảm cho ×20.', '',
        '## 2. Phân rã sai số trên cùng validation', '',
        f'Chênh MAE cân bằng (mở rộng − ba mã): **{summary["balanced_change_cm"]:.6f} cm**.', '',
        'Đóng góp dưới đây dùng mẫu số toàn validation trong từng nhóm nhãn cao/thấp; '
        'cộng các hàng của mỗi chiều sẽ khớp chênh MAE cân bằng. Không lấy trung bình MAE cân bằng các trạm.', '',
        '| Chiều | Nhóm | Cửa sổ | Nhãn cao | Đóng góp chênh MAE cân bằng (cm) |',
        '|---|---|---:|---:|---:|']
    for r in summary['decomposition']:
        if r['dimension'] in ('station', 'phase', 'target_band'):
            lines.append(f'| {r["dimension"]} | {r["group"]} | {r["windows"]:,} | {r["wet_windows"]:,} | {r["balanced_change_contribution_cm"]:+.6f} |')
    t = summary['threshold_transitions']
    lines += ['', '## 3. Giảm báo giả có đổi lấy bỏ sót không?', '',
        f'- Báo giả được sửa: **{t["fp_resolved"]}**; báo giả mới: **{t["fp_introduced"]}**.',
        f'- Bỏ sót được sửa: **{t["fn_resolved"]}**; bỏ sót mới: **{t["fn_introduced"]}**.',
        'Đây là ghép cặp đúng cùng cửa sổ, không phải số đợt hay số thông báo.', '',
        '## 4. Khác biệt dữ liệu bổ sung', '',
        'Xem `feature_distributions.csv`, `train_composition.csv` và `train_raw_proc_comparison.csv`. '
        'So sánh raw/proc ở cùng bin train: raw là trung vị bin, proc là hàng cuối; '
        'chênh lệch phản ánh cả tổng hợp theo thời gian lẫn xử lý nguồn, không tự chứng minh lỗi hiệu chuẩn/nhãn.', '',
        '| Mã | Tỷ lệ lịch sử có raw âm | Tỷ lệ lịch sử thưa | P99 chênh raw/proc tuyệt đối (cm) |',
        '|---|---:|---:|---:|']
    for r in quality:
        lines.append(f'| {r["station"]} | {r["negative_raw_history_share"]:.2%} | {r["sparse_raw_history_share"]:.2%} | {r["raw_proc_abs_diff_p99_cm"]:.3f} |')
    lines += ['', '## 5. Các ngày đóng góp tăng sai số nhiều nhất', '',
        'Đây là xếp hạng hồi cứu để xem lỗi, không phải sự kiện ngập độc lập và không dùng để xóa mẫu.', '']
    for _, r in daily.head(5).iterrows():
        lines.append(f'- `{r["group"]}`: {r["balanced_change_contribution_cm"]:+.6f} cm.')
    lines += ['', '## 6. Biểu đồ và kiểm tra', '']
    lines += [f'- [{name}]({name})' for name in summary['plots']]
    lines += ['', f'- Tái lập 8 model + 2 baseline trên 92.215 cửa sổ: PASS.',
        f'- {summary["frozen_run_files_unchanged"]} file trong hai run cũ không đổi checksum.',
        '- Không đọc test hoặc validation trạm mới; chỉ đọc cache các tháng trước 01/2023.',
        '- Các ngưỡng pha/nhóm chỉ để mô tả hồi cứu; không là feature hoặc bộ lọc vận hành.', '',
        '## 7. Kết luận và hướng tiếp', '',
        'Giữ model ba mã theo tiêu chí cũ. Đã định vị các nhóm đóng góp tăng/giảm sai số; '
        'chưa tách được nguyên nhân do thành phần trạm, nhiễu, mốc đo hay động lực nước. '
        'Muốn kiểm chứng nhân quả cần thí nghiệm thay một yếu tố có kiểm soát, không chọn trạm bằng điểm validation. '
        'Bước tiếp theo là phần B: định nghĩa và đánh giá cảnh báo theo đợt. Chưa triển khai cảnh báo AI thật.', '',
        'Nguồn: FloodNet (NYU/CUNY). Dữ liệu và artifacts giữ cục bộ; tuân giấy phép nguồn.']
    (output / 'BAO_CAO_PHAN_TICH_A.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    run()
