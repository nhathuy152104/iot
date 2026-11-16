"""Provisional local-only FloodNet 5-minute preparation; never trains a model.

Station selection uses train data only. Preserve nulls on the last source row,
do not interpolate, and include entire bin support in split/embargo checks.
"""
import calendar
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DEPTHS = ['depth_raw_mm', 'depth_filt_mm', 'depth_proc_mm']
TRAIN_END = pd.Timestamp('2023-01-01', tz='UTC')
VALIDATION_END = pd.Timestamp('2023-07-01', tz='UTC')
EMBARGO = pd.Timedelta(hours=24)
POLICY = {
    'min_train_windows': 1000, 'min_train_wet_windows': 20,
    'min_train_nonwet_windows': 1000, 'min_train_decision_days': 30,
    'min_train_wet_days': 3, 'min_train_wet_temporal_groups_6h': 3,
    'max_train_history_disagreement_fraction': 0.10,
    'max_train_history_jump_fraction': 0.005,
    'max_train_missing_grid_fraction': 0.30, 'max_selected_stations': 20,
}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def month_key(path):
    return (int(path.stem.split('-')[-1]), list(calendar.month_abbr).index(path.stem.split('-')[-2].title()))


def make_grid(frame):
    """Last physical observation, not last non-null independently per column."""
    frame = frame.sort_index(kind='stable')
    frame = frame.loc[~frame.index.duplicated(keep='first')].copy()
    frame['source_time_utc'] = frame.index
    frame['bin_end_utc'] = frame.index.ceil('5min')
    frame = frame.drop_duplicates('bin_end_utc', keep='last').set_index('bin_end_utc')
    index = pd.date_range(frame.index.min(), frame.index.max(), freq='5min', name='decision_time_utc')
    grid = frame.reindex(index)
    grid['source_age_seconds'] = (grid.index.to_series() - grid['source_time_utc']).dt.total_seconds()
    return grid


def split_masks(index):
    earliest_support = index - pd.Timedelta(minutes=15)
    last_support = index + pd.Timedelta(minutes=10)
    return {
        'train': last_support < TRAIN_END - EMBARGO,
        'validation': (earliest_support >= TRAIN_END + EMBARGO) & (last_support < VALIDATION_END - EMBARGO),
        'test': earliest_support >= VALIDATION_END + EMBARGO,
    }


def make_features(grid, station):
    depth = grid['depth_proc_mm'] / 10
    rolling = depth.rolling(3, min_periods=3)
    eligible = rolling.count().eq(3) & rolling.min().ge(0) & depth.shift(-2).ge(0)
    x = pd.DataFrame(index=grid.index)
    for steps in (0, 1, 2):
        x[f'depth_lag_{steps * 5}_cm'] = depth.shift(steps)
        x[f'source_age_lag_{steps * 5}_seconds'] = grid['source_age_seconds'].shift(steps)
    x['slope_5min_cm_per_min'] = (depth - depth.shift(1)) / 5
    x['slope_10min_cm_per_min'] = (depth - depth.shift(2)) / 10
    for name in ('mean', 'min', 'max', 'std'):
        x[f'{name}_three_bins_cm'] = getattr(rolling, name)()
    x['target_depth_bin_t_plus_10min_cm'] = depth.shift(-2)
    # Future source timestamps are label audit metadata, NEVER model features.
    x['audit_target_source_time_utc'] = grid['source_time_utc'].shift(-2)
    x['audit_realized_target_lead_seconds'] = (
        x['audit_target_source_time_utc'] - grid.index.to_series()).dt.total_seconds()
    x['deployment_id'] = station
    mismatch = (grid['depth_raw_mm'] - grid['depth_proc_mm']).abs().ge(100)
    x['review_raw_proc_diff_ge_10cm_history'] = mismatch.rolling(3, min_periods=1).max().astype(bool)
    # For the three history bins, only two internal transitions are assessed.
    jumps = depth.diff().abs().div(5).gt(25.4)
    x['review_proc_jump_history'] = jumps.rolling(2, min_periods=1).max().astype(bool)
    x['review_raw_missing_history'] = grid['depth_raw_mm'].isna().rolling(3, min_periods=1).max().astype(bool)
    return x, eligible


def temporal_groups(times):
    """A descriptive proxy only; NOT verified independent storms."""
    if len(times) == 0:
        return 0
    return int((times.to_series().diff() > pd.Timedelta(hours=6)).sum()) + 1


def summarize_split(station, split, windows):
    wet = windows['target_depth_bin_t_plus_10min_cm'].ge(10)
    wet_times = windows.index[wet]
    return {
        'deployment_id': station, 'split': split, 'windows': len(windows),
        'decision_days': int(windows.index.normalize().nunique()),
        'wet_windows_ge_10cm': int(wet.sum()), 'nonwet_windows_lt_10cm': int((~wet).sum()),
        'wet_days': int(wet_times.normalize().nunique()),
        'wet_temporal_groups_6h': temporal_groups(wet_times),
        'history_raw_proc_disagreement_fraction': float(windows['review_raw_proc_diff_ge_10cm_history'].mean()) if len(windows) else None,
        'history_proc_jump_fraction': float(windows['review_proc_jump_history'].mean()) if len(windows) else None,
        'history_raw_missing_fraction': float(windows['review_raw_missing_history'].mean()) if len(windows) else None,
        'current_source_age_seconds_p50': float(windows['source_age_lag_0_seconds'].median()) if len(windows) else None,
        'current_source_age_seconds_p95': float(windows['source_age_lag_0_seconds'].quantile(.95)) if len(windows) else None,
        'realized_target_lead_seconds_p05': float(windows['audit_realized_target_lead_seconds'].quantile(.05)) if len(windows) else None,
        'realized_target_lead_seconds_p50': float(windows['audit_realized_target_lead_seconds'].median()) if len(windows) else None,
        'realized_target_lead_seconds_p95': float(windows['audit_realized_target_lead_seconds'].quantile(.95)) if len(windows) else None,
        'first_decision_utc': str(windows.index.min()) if len(windows) else None,
        'last_decision_utc': str(windows.index.max()) if len(windows) else None,
    }


def classify(train, missing_fraction):
    if train['windows'] == 0:
        return 'no_train_windows', 'Không có cửa sổ train theo mốc cố định; có thể dùng cho khảo sát trạm mới.'
    enough = (train['windows'] >= POLICY['min_train_windows'] and
              train['decision_days'] >= POLICY['min_train_decision_days'])
    if not enough:
        return 'insufficient_train_coverage', 'Ít cửa sổ hoặc ít ngày có cửa sổ train.'
    flags = (train['history_raw_proc_disagreement_fraction'] > POLICY['max_train_history_disagreement_fraction'] or
             train['history_proc_jump_fraction'] > POLICY['max_train_history_jump_fraction'] or
             missing_fraction > POLICY['max_train_missing_grid_fraction'])
    if flags:
        return 'review_quality', 'Cần đối soát cờ chất lượng hoặc tỷ lệ bin thiếu; không tự kết luận lỗi.'
    events = (train['wet_windows_ge_10cm'] >= POLICY['min_train_wet_windows'] and
              train['nonwet_windows_lt_10cm'] >= POLICY['min_train_nonwet_windows'] and
              train['wet_days'] >= POLICY['min_train_wet_days'] and
              train['wet_temporal_groups_6h'] >= POLICY['min_train_wet_temporal_groups_6h'])
    if not events:
        return 'limited_wet_evidence', 'Độ phủ đạt nhưng chưa đủ bằng chứng cửa sổ có nước đa dạng theo chính sách.'
    return 'candidate_core', 'Đạt sàng lọc tự động TRAIN; chưa xác minh thực địa/bộ lọc causal.'


def cache_sources(station_ids):
    """Per-month/per-deployment cache, rebuilt when source fingerprint changes."""
    cache_root = OUT / 'local_cache'
    cache_root.mkdir(exist_ok=True)
    sources = sorted(ROOT.glob('*.csv'), key=month_key)
    directories = []
    for path in sources:
        fingerprint = {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                       'station_ids': sorted(station_ids), 'cache_schema': 1}
        digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
        folder = cache_root / f'{path.stem}_{digest}'
        directories.append(folder)
        marker = folder / 'complete.json'
        if marker.exists() and load_json(marker) == fingerprint:
            print(f'Cache {path.name}', flush=True)
            continue
        folder.mkdir(exist_ok=True)
        batches = []
        for chunk in pd.read_csv(path, usecols=['time', 'deployment_id', *DEPTHS], chunksize=250_000):
            selected = chunk.loc[chunk['deployment_id'].isin(station_ids)].copy()
            if len(selected):
                batches.append(selected)
        if batches:
            frame = pd.concat(batches, ignore_index=True)
            frame['time'] = pd.to_datetime(frame['time'], format='mixed', utc=True, errors='coerce')
            if frame['time'].isna().any():
                raise ValueError(f'Invalid time in {path.name}; audit before proceeding')
            for col in DEPTHS:
                frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            for station, group in frame.groupby('deployment_id', sort=False):
                group.drop(columns='deployment_id').set_index('time').to_parquet(folder / f'{station}.parquet')
        save_json(marker, fingerprint)
        print(f'Extracted {path.name}', flush=True)
    return sources, directories


def plot_train_review(station, frame, grid):
    """Training-only four-hour case: highest processed depth, never test-picked."""
    train_end = TRAIN_END - EMBARGO
    view_grid = grid.loc[grid.index + pd.Timedelta(minutes=10) < train_end]
    depth = view_grid['depth_proc_mm'].dropna()
    if depth.empty:
        return None
    center = depth.idxmax()
    start, end = center - pd.Timedelta(hours=2), min(center + pd.Timedelta(hours=2), train_end)
    view = frame.loc[(frame.index > start) & (frame.index < end)]
    # Scatter shows actual source times without drawing across unobserved gaps.
    fig, ax = plt.subplots(figsize=(12, 4), layout='constrained')
    for col, color in zip(DEPTHS, ['#777777', '#e69f00', '#0072b2']):
        ax.scatter(view.index, view[col] / 10, s=8, label=col, color=color, alpha=.65)
    ax.axhline(10, color='red', linestyle='--', linewidth=.8)
    ax.set_title(f'{station} | TRAIN-ONLY review | {center:%Y-%m-%d %H:%M} UTC\nNot a verified flood; no interpolated observations')
    ax.set_ylabel('Depth (cm)')
    ax.set_xlabel('Actual source time (UTC)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    folder = OUT / 'plots'
    folder.mkdir(exist_ok=True)
    path = folder / f'{station}_train_review.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path.relative_to(OUT))


def main():
    OUT.mkdir(exist_ok=True)
    inventory = pd.read_csv(ROOT / 'audit' / 'stations.csv')
    stations = inventory.loc[inventory['type'].str.strip().eq('pluvial')].set_index('deployment_id')
    station_ids = set(stations.index)
    if len(station_ids) != 58:
        print(f'Notice: current inventory has {len(station_ids)} pluvial deployments', flush=True)
    sources, directories = cache_sources(station_ids)
    window_dir = OUT / 'provisional_windows'
    window_dir.mkdir(exist_ok=True)
    station_rows, split_rows = [], []
    for station in sorted(station_ids):
        parts = [pd.read_parquet(folder / f'{station}.parquet') for folder in directories
                 if (folder / f'{station}.parquet').exists()]
        if not parts:
            raise ValueError(f'No observations found for {station}')
        frame = pd.concat(parts).sort_index(kind='stable')
        duplicate_count = int(frame.index.duplicated().sum())
        frame = frame.loc[~frame.index.duplicated(keep='first')]
        grid = make_grid(frame)
        features, eligible = make_features(grid, station)
        masks = split_masks(grid.index)
        per_split = {}
        for split, mask in masks.items():
            windows = features.loc[eligible & mask]
            windows.to_parquet(window_dir / f'{station}_{split}.parquet')
            per_split[split] = summarize_split(station, split, windows)
            split_rows.append(per_split[split])
        train_bins = grid.loc[masks['train']]
        missing_fraction = float(train_bins['depth_proc_mm'].isna().mean()) if len(train_bins) else None
        category, reason = classify(per_split['train'], missing_fraction)
        source_train = frame.loc[frame.index < TRAIN_END - EMBARGO]
        missing_source_fraction = float(source_train['depth_proc_mm'].isna().mean()) if len(source_train) else None
        plot = plot_train_review(station, frame, grid)
        station_rows.append({'deployment_id': station, 'name': stations.loc[station, 'name'],
            'category': category, 'reason': reason, 'selected_core': False,
            'train_missing_grid_fraction': missing_fraction,
            'train_missing_source_fraction': missing_source_fraction,
            'source_rows': len(frame), 'duplicate_times_excluded': duplicate_count,
            'first_source_utc': str(frame.index.min()), 'last_source_utc': str(frame.index.max()),
            'eligible_windows_before_split_purge': int(eligible.sum()),
            'windows_purged_or_embargoed': int(eligible.sum()) - sum(r['windows'] for r in per_split.values()),
            **{f'train_{k}': v for k, v in per_split['train'].items() if k not in ('deployment_id', 'split')},
            'train_review_plot': plot})
        print(f'{station}: {category}; train={per_split["train"]["windows"]:,}; wet={per_split["train"]["wet_windows_ge_10cm"]:,}', flush=True)
    candidates = sorted([r for r in station_rows if r['category'] == 'candidate_core'], key=lambda r: (
        -r['train_wet_temporal_groups_6h'], -r['train_wet_days'],
        r['train_history_proc_jump_fraction'], r['train_history_raw_proc_disagreement_fraction'],
        r['train_missing_grid_fraction'], r['deployment_id']))
    selected = [r['deployment_id'] for r in candidates[:POLICY['max_selected_stations']]]
    for row in station_rows:
        row['selected_core'] = row['deployment_id'] in selected
    pd.DataFrame(station_rows).to_csv(OUT / 'station_selection.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(split_rows).to_csv(OUT / 'window_counts.csv', index=False, encoding='utf-8-sig')
    # Availability and wet counts outside train are descriptive only, not ranking inputs.
    manifest = {'status': 'PROVISIONAL; no model trained; archive-filter causality unverified',
        'policy': POLICY, 'selection_inputs': 'TRAIN ONLY; no validation/test ranking or quality thresholds fitted',
        'selected_core': selected, 'eligible_core_candidates': len(candidates),
        'categories': pd.Series([r['category'] for r in station_rows]).value_counts().to_dict(),
        'stations_scanned': len(station_rows), 'source_files': [p.name for p in sources],
        'grid': '5min UTC (t-5min,t], last physical row, null-preserving, no interpolation/fill',
        'history': '3 valid nonnegative proc bins t-10,t-5,t; full support (t-15min,t]',
        'target': 'last physical observation in bin (t+5min,t+10min], NOT instantaneous depth at t+10',
        'split_boundaries_utc': [str(TRAIN_END), str(VALIDATION_END)], 'embargo_hours_each_side': 24,
        'ranking': 'train wet 6h groups desc, wet days desc, jump flags asc, mismatch flags asc, missing bins asc, ID',
        'feature_columns': [f'depth_lag_{lag}_cm' for lag in (0, 5, 10)] +
            [f'source_age_lag_{lag}_seconds' for lag in (0, 5, 10)] +
            ['slope_5min_cm_per_min', 'slope_10min_cm_per_min'] +
            [f'{stat}_three_bins_cm' for stat in ('mean', 'min', 'max', 'std')],
        'label_column': 'target_depth_bin_t_plus_10min_cm',
        'nonfeature_columns': 'deployment_id, decision timestamp, every review_* and audit_* column',
        'test_status': '2023 has prior exploratory inspection; NOT a pristine confirmatory holdout',
        'limits': ['All quality thresholds are proposed screening rules, not ground-truth flood verification.',
                   '6h wet groups are not independent verified storms; 24h embargo does not guarantee storm independence.',
                   'Source-age features capture differing sampling cadence; target bin ages may differ across stations.',
                   'Do not publish cache/windows; review FloodNet agreement before sharing derivatives.',
                   'Raw/proc disagreement may be effective noise removal rather than bad observations.',
                   'Causal row aggregation cannot make an offline noncausal archive filter causal.']}
    save_json(OUT / 'selection_manifest.json', manifest)
    write_report(manifest, station_rows, split_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def write_report(manifest, station_rows, split_rows):
    selected = [r for r in station_rows if r['selected_core']]
    no_train = [r for r in station_rows if r['category'] == 'no_train_windows']
    late = sum(pd.Timestamp(r['first_source_utc']) >= TRAIN_END for r in no_train)
    lines = ['# Sàng lọc 58 trạm pluvial với bước 5 phút', '',
        '**Trạng thái: bộ thăm dò, chưa train model và chưa xác minh tính nhân quả bộ lọc proc.**', '',
        f'Đã quét {manifest["stations_scanned"]} lần triển khai; {manifest["eligible_core_candidates"]} trạm đạt chính sách sàng lọc train; chọn {len(selected)} trạm core (tối đa 20, không ép đủ 10).', '',
        '## Phân nhóm', '', '| Nhóm | Số trạm |', '|---|---:|']
    for category, count in manifest['categories'].items():
        lines.append(f'| {category} | {count} |')
    lines += ['', 'candidate_core: đạt sàng lọc tự động; review_quality: cần đối soát; limited_wet_evidence: chưa đủ cửa sổ có nước đa dạng; insufficient_train_coverage: thiếu độ phủ train; no_train_windows: không có train theo mốc hiện tại. Không nhóm nào được coi là xác nhận ngập/lỗi hiện trường.', '',
        '## Quy tắc chọn — chỉ dùng train', '',
        'Train có >=1.000 cửa sổ, >=30 ngày có cửa sổ, >=20 nhãn >=10 cm, >=1.000 nhãn <10 cm, >=3 ngày có nhãn cao và >=3 nhóm nhãn cao cách nhau hơn 6 giờ.',
        'Cần tỷ lệ cửa sổ có cờ raw/proc lệch >=10 cm không quá 10%; cờ độ dốc proc >25,4 cm/phút không quá 0,5%; bin proc thiếu trong giai đoạn train quan sát không quá 30%. Đây là ngưỡng thiết kế sàng lọc, chưa được chứng minh tối ưu, không được chỉnh theo test.',
        'Thiếu bin không đồng nghĩa bản tin null: bao gồm khoảng không quan sát, chu kỳ gửi thưa và mất liên lạc. Không ép cân bằng bằng số ngày khô tổng hợp hoặc nội suy.', '',
        'Xếp hạng các trạm đạt theo số nhóm nhãn cao 6h, số ngày có nhãn cao, rồi cờ độ dốc, cờ raw/proc, bin thiếu và ID. Xếp hạng này chưa tối ưu đa dạng không gian/lưu vực; không mặc định nhiều nhóm là nhiều trận mưa thật.', '',
        '## Trạm core tạm thời', '', '| ID | Trạm | Cửa sổ train | Nhãn >=10 cm | Ngày có nhãn cao | Nhóm 6h |', '|---|---|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f'| {r["deployment_id"]} | {r["name"]} | {r["train_windows"]:,} | {r["train_wet_windows_ge_10cm"]:,} | {r["train_wet_days"]} | {r["train_wet_temporal_groups_6h"]} |')
    if not selected:
        lines.append('| Không có trạm đạt tất cả tiêu chí | | | | | |')
    notes_path = OUT / 'manual_review_notes.json'
    if notes_path.exists():
        notes = load_json(notes_path)
        lines += ['', '## Đối soát trực quan bổ sung', '',
            'Giữ nguyên kết quả sàng lọc tự động; không điều chỉnh ngưỡng theo ảnh hoặc test. Các ghi chú sau chỉ cho trường hợp đỉnh TRAIN đã xem, không xác minh toàn bộ sự kiện của trạm.']
        for note in notes['notes']:
            lines.append(f'- **{note["deployment_id"]}** ({note["status"]}, {note["case_utc"]}): {note["observation_vi"]}')
        lines += ['', '**Ba trạm core là ứng viên tạm thời, không phải ba trạm đã được xác nhận nhãn sạch.**']
    lines += ['', '## Cửa sổ theo tập', '', '| Phạm vi | Split | Cửa sổ | Nhãn >=10 cm |', '|---|---|---:|---:|']
    for scope, ids in [('Tất cả trạm', {r['deployment_id'] for r in station_rows}), ('Core tạm thời', set(manifest['selected_core']))]:
        for split in ('train', 'validation', 'test'):
            rows = [r for r in split_rows if r['deployment_id'] in ids and r['split'] == split]
            lines.append(f'| {scope} | {split} | {sum(r["windows"] for r in rows):,} | {sum(r["wet_windows_ge_10cm"] for r in rows):,} |')
    lines += ['', '## Ý nghĩa thời gian và giới hạn', '',
        '- Bin 5 phút lấy nguyên hàng quan trắc cuối, giữ null, không trộn ba cột từ ba thời điểm khác nhau; không nội suy/fill.',
        '- Lịch sử gồm 3 bin t-10/t-5/t; support thực tế 15 phút vì mỗi bin rộng 5 phút. Độ mới số đo được lưu làm đặc trưng.',
        '- Nhãn là bin ở t+10 phút, tức số đo cuối trong (t+5,t+10]; không đồng nhất với độ sâu tức thời đúng t+10.',
        '- Train trước 01/01/2023, validation từ 01/01 đến trước 01/07/2023, test từ 01/07/2023; embargo 24 giờ mỗi bên ranh giới, bao gồm toàn support lịch sử và nhãn.',
        '- Embargo 24 giờ là lựa chọn thận trọng để giảm liên hệ cùng đợt mưa/thoát nước qua ranh giới, không phải yêu cầu toán học của horizon 10 phút. Nó có thể làm mất cửa sổ ngập hiếm và chưa bảo đảm độc lập khí tượng.',
        '- Thay mốc chia so với bộ ba trạm trước để có thêm lịch sử train; mốc này cố định trước chạy sàng lọc, không chọn bằng điểm dự báo.',
        '- Nhóm trạm chỉ xuất hiện muộn được giữ riêng để khảo sát chuyển vị trí. Không xóa hoặc tự chia ngẫu nhiên cho đủ tập.',
        f'- Đã kiểm tra thêm: {late}/{len(no_train)} trạm no_train_windows bắt đầu có số đo từ mốc train kết thúc trở đi. Không thể đưa chúng vào train trước 2023 mà vẫn giữ cách chia này. Không hạ ngưỡng tự động để ép đủ 10–20 trạm.',
        '- Tỷ lệ thiếu bin là tiêu chí thận trọng cấp trạm; trạm bị xếp review_quality vẫn có thể có nhiều cửa sổ tốt. Không coi toàn bộ nhóm này là hỏng hoặc vĩnh viễn loại khỏi nghiên cứu.',
        '- Năm 2023 đã được khảo sát trước đó: KHÔNG gọi test này là holdout xác nhận chưa từng xem. Cần một bộ kiểm định độc lập sau khi chốt toàn pipeline.',
        '- Các cột review_* và metadata không tự đưa vào đặc trưng model. Không dùng nhãn tương lai làm đặc trưng.',
        '- Các cột audit_* chứa metadata tương lai của nhãn: TUYỆT ĐỐI không train bằng chúng. Manifest có whitelist feature_columns. window_counts.csv ghi độ mới đầu vào và phân vị lead time thực tế.',
        '- Các tập validation/test thuộc mùa khác nhau: khi đánh giá model cần tách thống kê theo mùa/sự kiện, không suy mọi chênh lệch là khả năng tổng quát.',
        '- Chưa so sánh XGBoost nhóm core với tất cả trạm đủ điều kiện. Không thể kết luận nhóm nào tốt hơn khi chưa train/validation.',
        '- Cần tiếp tục đối soát các trạm core và phương pháp lọc causal trước khi dùng dữ liệu vào đánh giá thời gian thực.', '',
        '## Đầu ra', '',
        '- `station_selection.csv`: danh sách 58 trạm, phân nhóm, lý do và thống kê train.',
        '- `window_counts.csv`: số cửa sổ từng trạm/split; validation/test chỉ dùng mô tả, không dùng xếp hạng.',
        '- `selection_manifest.json`: quy tắc, trạm core, nguồn và giới hạn.',
        '- `index.html`, `plots/`: biểu đồ kiểm tra chọn từ TRAIN, chưa xác minh thủ công tất cả.',
        '- `provisional_windows/`: cửa sổ tất cả trạm/split, kể cả file rỗng, chưa phải bộ train được duyệt.',
        '- `local_cache/`: dữ liệu trích xuất cục bộ theo tháng/trạm; fingerprint gồm kích thước, mtime và schema cache.', '',
        'Nguồn: FloodNet (New York University and The City University of New York). Không sửa CSV gốc, không đưa dữ liệu lên dịch vụ bên ngoài. Giữ cache/cửa sổ cục bộ. Phân tích/biểu diễn phái sinh được phép tuân CC BY-NC-SA 4.0 và Data Access License Agreement; cần rà quyền trước công khai/bàn giao.', '']
    (OUT / 'BAO_CAO_5_PHUT.md').write_text('\n'.join(lines), encoding='utf-8')
    cards = []
    for r in sorted(station_rows, key=lambda r: (not r['selected_core'], r['category'], r['deployment_id'])):
        img = f'<img loading="lazy" src="{html.escape(r["train_review_plot"])}" alt="Train review">' if r['train_review_plot'] else '<p>Không có dữ liệu train để vẽ.</p>'
        cards.append(f'<section><h2>{html.escape(r["deployment_id"])} — {html.escape(str(r["name"]))}</h2><p>{r["category"]} | core={r["selected_core"]}</p>{img}</section>')
    (OUT / 'index.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>FloodNet 5 phút</title><style>body{font:16px system-ui;max-width:1350px;margin:30px auto;padding:20px;background:#f4f6fa}section{background:white;padding:20px;margin:20px 0}img{width:100%}</style><h1>Đối soát TRAIN — 58 trạm pluvial</h1><p>Chưa xác nhận ngập; chưa train model. Chỉ dùng phần train để chọn biểu đồ và xếp nhóm.</p><p><a href="BAO_CAO_5_PHUT.md">Báo cáo</a> · <a href="station_selection.csv">Phân nhóm trạm</a></p>' + ''.join(cards) + '<footer>FloodNet (New York University and The City University of New York). Chỉ xem cục bộ; rà thỏa thuận trước chia sẻ.</footer></html>', encoding='utf-8')


if __name__ == '__main__':
    main()
